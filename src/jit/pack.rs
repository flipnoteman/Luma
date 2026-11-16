//! Arrays that live in one buffer, read through one binding.
//!
//! A device binds only so many storage buffers to a kernel (16 on llvmpipe, 29 on Metal), and a
//! program over a state of dozens of arrays (the 50 words of a Keccak state) needs one binding per
//! array. The arrays a kernel with many outputs writes are made to share one: the kernel is a
//! [`Tail::Stack`] that writes them into the one buffer of a *pack*, and each output is a
//! [window](Storage::window_of) of it. A kernel that reads windows of one pack binds the pack
//! once and reads each window through a translation, `pack[offset + i]`, which [`params`](super::params)
//! moves into the uniform: kernels that read the same arrays at other offsets are one kernel.
//!
//! Only the programs built by [`flatten`](super::program::flatten) are rewritten (by [`read_packs`]).
//! Anything else that needs a window's buffer (a readback, a write, a custom kernel, the placeholder
//! of an iteration) pins it, which gives the window a buffer of its own holding a copy of its range:
//! the arrays stay valid and independent, and only the ones that are used that way pay for the copy.
//!
//! The same happens to arrays that were not written together: when an expression reads more buffers
//! than a kernel binds, the evaluated arrays of one length and type it reads are copied into one
//! buffer (by [`repack`], from `create` and wherever a flattened program binds too many) that they
//! become windows of, instead of the expression being cut one operation at a time. Nothing is
//! packed while a step is being recorded (see `jit::replay::Trace`): the results of a recorded step
//! are the buffers its launches write, and a window would be copied out of a pack the recording
//! has not run yet.

use std::sync::Arc;

use crate::array::Storage;

use crate::context::Context;
use crate::gpu::hash::FxMap;

use super::program::{shape_of, Fused, Instr, Ref};
use super::{stack, Edge, ElemTy, JitState, Op, OpKind, ShiftKind, Src, ViewMap};

/// The program of `fused` with the inputs that are windows of one pack read through the pack, one
/// binding for all of them.
///
/// A window that a shift reads (a view of it, a circular shift) stays an input of its own, as does
/// every window of a program with sub-programs (the judgement of its staging refers to the
/// instructions that are there); pinned for the launch it is a plain buffer. The same goes for
/// windows of a type other than the pack's first window.
pub(super) fn read_packs(jit: &JitState, fused: &mut Fused) {
    if !fused.inputs.iter().any(|input| input.is_window()) || !fused.key.subs.is_empty() {
        return;
    }
    let key = &fused.key;
    let n = fused.inputs.len();
    let mut shifted = vec![false; n];
    for instr in key.instrs.iter() {
        if matches!(instr.op.kind, OpKind::Shift(_))
            && let Ref::Input(k) = instr.args[0]
            && let Some(slot) = shifted.get_mut(k as usize)
        {
            *slot = true;
        }
    }

    // Where every input goes: a new input of its own, or a range of a pack's input.
    enum Place {
        Own(u32),
        Window { input: u32, offset: usize, len: usize },
    }
    let mut inputs: Vec<Arc<Storage>> = Vec::with_capacity(n);
    let mut tys: Vec<ElemTy> = Vec::with_capacity(n);
    let mut places: Vec<Place> = Vec::with_capacity(n);
    let mut windows = 0;
    for ((input, &ty), &read_shifted) in fused.inputs.iter().zip(&key.input_tys).zip(&shifted) {
        if read_shifted {
            // Read through a view or shift, which cannot be told to read the pack instead.
            let _ = input.detach();
        }
        let window = if read_shifted || ty == ElemTy::Bool { None } else { input.window() };
        let Some((pack, offset)) = window else {
            inputs.push(Arc::clone(input));
            tys.push(ty);
            places.push(Place::Own(inputs.len() as u32 - 1));
            continue;
        };
        let at = match inputs.iter().position(|known| Arc::ptr_eq(known, &pack)) {
            Some(at) => at,
            None => {
                inputs.push(pack);
                tys.push(ty);
                inputs.len() - 1
            }
        };
        if tys[at] != ty {
            // A window of another type than the first one read from this pack: not through it.
            inputs.push(Arc::clone(input));
            tys.push(ty);
            places.push(Place::Own(inputs.len() as u32 - 1));
            continue;
        }
        windows += 1;
        places.push(Place::Window { input: at as u32, offset, len: input.len() });
    }
    if windows == 0 {
        return;
    }

    // One translation per window, ahead of the program; every read of the window is its value.
    let mut loads: Vec<Instr> = Vec::with_capacity(windows);
    let mut loaded: Vec<Option<u32>> = vec![None; n];
    for (k, place) in places.iter().enumerate() {
        if let Place::Window { input, offset, len } = *place {
            let total = inputs[input as usize].len();
            let mut map = ViewMap::identity([total as u32, 1, 1, 1]);
            map.out = [len as u32, 1, 1, 1];
            map.axes[0].start = offset as i32;
            let op = Op::new(OpKind::Shift(ShiftKind::View { map, edge: Edge::None }), tys[input as usize]);
            loaded[k] = Some(loads.len() as u32);
            loads.push(Instr { op, args: [Ref::Input(input), Ref::Unused, Ref::Unused] });
        }
    }
    let shift = loads.len() as u32;
    let rewrite = |arg: Ref| match arg {
        Ref::Input(k) => match places[k as usize] {
            Place::Own(input) => Ref::Input(input),
            Place::Window { .. } => Ref::Tmp(loaded[k as usize].unwrap_or(0)),
        },
        Ref::Tmp(t) => Ref::Tmp(t + shift),
        other => other,
    };
    let mut instrs = loads;
    instrs.extend(fused.key.instrs.iter().map(|instr| Instr { op: instr.op, args: instr.args.map(rewrite) }));
    let key = &mut fused.key;
    key.shape = shape_of(&instrs, &[]);
    key.instrs = Arc::new(instrs);
    key.input_tys = tys;
    for output in &mut key.outputs {
        *output += shift;
    }
    if !fused.instr_nodes.is_empty() {
        let mut nodes = vec![None; shift as usize];
        nodes.append(&mut fused.instr_nodes);
        fused.instr_nodes = nodes;
    }
    fused.inputs = inputs;
    jit.intern(&mut fused.key);
}

/// Fewest buffers of one kind that are worth copying into a pack.
const MIN_PACK: usize = 3;

/// Makes the buffers an expression reads (the operands `args` of `op`) windows of packs, so that
/// it binds fewer of them: the evaluated buffers of one length and type, read at the element
/// itself, are copied into one new buffer (by the kernels of [`stack::run`], at most as many
/// inputs as a kernel binds each), and the arrays that held them show it from then on. Windows
/// of several packs are moved into one pack like that too. Takes the groups of at least
/// [`MIN_PACK`] that save the most bindings first, until `excess` fewer buffers are bound.
///
/// What the arrays hold does not change, whoever holds them: pending operations still read the
/// contents they were made against, and a write gives its array a buffer of its own again (see
/// [`Storage::window_of`]). `false` if nothing was packed.
pub(super) fn repack(ctx: &Context, op: Op, args: &[Src], excess: usize) -> bool {
    pack_leaves(ctx, leaf_buffers(op, args), excess)
}

/// [`repack`] for the inputs of a program that binds too many: the buffers it reads (except the
/// ones a shift reads) are known exactly, with the types they are read as.
pub(super) fn repack_inputs(ctx: &Context, fused: &Fused, excess: usize) -> bool {
    let mut shifted = vec![false; fused.inputs.len()];
    for instr in fused.key.instrs.iter().chain(fused.key.subs.iter().flatten()) {
        if matches!(instr.op.kind, OpKind::Shift(_))
            && let Ref::Input(k) = instr.args[0]
            && let Some(slot) = shifted.get_mut(k as usize)
        {
            *slot = true;
        }
    }
    let leaves = fused.inputs.iter().zip(&fused.key.input_tys).zip(shifted).map(|((storage, &ty), shifted)| (Arc::clone(storage), ty, shifted)).collect();
    pack_leaves(ctx, leaves, excess)
}

fn pack_leaves(ctx: &Context, leaves: Vec<(Arc<Storage>, ElemTy, bool)>, excess: usize) -> bool {
    // A recorded step replays its launches over the buffers they were recorded with: the packs
    // would hold what the recording has not run yet.
    if ctx.recording_here() {
        return false;
    }
    type Kind = (usize, u64, ElemTy);
    let mut groups: FxMap<Kind, Vec<Arc<Storage>>> = FxMap::default();
    for (storage, ty, shifted) in leaves {
        if !shifted && ty != ElemTy::Bool && storage.len() > 0 && storage.can_adopt() {
            groups.entry((storage.len(), storage.layout().stride() as u64, ty)).or_default().push(storage);
        }
    }
    // What a group saves: the bindings of its members, but for the one of the pack they move to.
    let binding_ids = |members: &[Arc<Storage>]| {
        let mut ids: Vec<u64> = members.iter().map(|member| member.binding_id()).collect();
        ids.sort_unstable();
        ids.dedup();
        ids.len()
    };
    let mut groups: Vec<(Kind, usize, Vec<Arc<Storage>>)> = groups
        .into_iter()
        .filter(|(_, members)| members.len() >= MIN_PACK)
        .map(|(kind, members)| (kind, binding_ids(&members).saturating_sub(1), members))
        .filter(|&(_, saved, _)| saved > 0)
        .collect();
    groups.sort_by_key(|&((len, _, _), saved, _)| (std::cmp::Reverse(saved), len));
    let (mut saved_total, mut any) = (0, false);
    for ((len, stride, ty), saved, members) in groups {
        if saved_total >= excess {
            break;
        }
        // Packs hold what a buffer does.
        let per_pack = (ctx.segment_bytes() / (len as u64 * stride)) as usize;
        for chunk in members.chunks(per_pack.max(2)) {
            if chunk.len() < 2 {
                continue;
            }
            let parts: Vec<Src> = chunk.iter().map(|storage| Src::Buffer(Arc::clone(storage))).collect();
            let Ok(Some(pack)) = stack::run(ctx, &parts, ty, len, len, "pack", std::panic::Location::caller()) else { continue };
            let adopted = chunk.iter().enumerate().filter(|&(j, storage)| storage.adopt(&pack, j * len)).count();
            any |= adopted > 1;
        }
        saved_total += saved;
    }
    any
}

/// The evaluated buffers the expressions `args` (the operands of `op`) read, each with the type
/// it is read as and whether a shift reads it (or it is read as two types: then it is treated
/// as shifted).
fn leaf_buffers(op: Op, args: &[Src]) -> Vec<(Arc<Storage>, ElemTy, bool)> {
    let mut found: Vec<(Arc<Storage>, ElemTy, bool)> = Vec::new();
    let mut seen: FxMap<u64, ()> = FxMap::default();
    let shifts = |op: Op, i: usize| i == 0 && matches!(op.kind, OpKind::Shift(_));
    let mut work: Vec<(Src, ElemTy, bool)> = args.iter().enumerate().map(|(i, arg)| (arg.clone(), op.operand_ty(i), shifts(op, i))).collect();
    while let Some((src, ty, shifted)) = work.pop() {
        let storage = match &src {
            Src::Scalar(_) => continue,
            Src::Buffer(storage) => Some(storage),
            Src::Node(node) => node.evaluated(),
        };
        if let Some(storage) = storage {
            match found.iter_mut().find(|(known, ..)| Arc::ptr_eq(known, storage)) {
                Some(entry) => entry.2 |= shifted || entry.1 != ty,
                None => found.push((Arc::clone(storage), ty, shifted)),
            }
            continue;
        }
        let Src::Node(node) = src else { continue };
        if seen.insert(node.id, ()).is_some() {
            continue;
        }
        let kind = node.op();
        if let Some(operands) = node.operands() {
            for (i, operand) in operands.into_iter().take(kind.kind.arity()).enumerate() {
                work.push((operand, kind.operand_ty(i), shifts(kind, i)));
            }
        }
    }
    found
}
