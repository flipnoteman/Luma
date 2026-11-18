//! `kir`: the backend-neutral kernel IR, its builder, the WGSL front end and target rewrites.

pub mod builder;
pub mod from_wgsl;
pub mod ir;
pub mod opt;

pub use builder::*;
pub use from_wgsl::{PRELUDE, VIEW_PRELUDE, parse_wgsl, used_bindings};
pub use ir::*;
pub use opt::optimize_for_target;

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use super::*;
    use crate::hal::{BackendError, BindingKind, Features};

    type Parsed = (naga::Module, KernelModule, Vec<crate::hal::BindingInfo>, usize, Option<Box<[bool]>>);

    fn parse_with(src: &str, entry: &str, trusted: bool) -> Result<Parsed, BackendError> {
        parse_wgsl(src, entry, None, Features::SUBGROUP, [256, 256, 64], 256, 8, trusted)
    }

    fn parse(src: &str, entry: &str) -> KernelModule {
        parse_with(src, entry, false).expect("parses").1
    }

    fn stmts(m: &KernelModule) -> Vec<Stmt> {
        let mut out = Vec::new();
        for body in std::iter::once(&m.entry_body).chain(m.functions.iter().map(|f| &f.body)) {
            walk_stmts(body, &mut |s| out.push(s.clone()));
        }
        out
    }

    fn exprs(m: &KernelModule) -> Vec<Expr> {
        let mut out = Vec::new();
        for s in stmts(m) {
            s.for_each_expr(&mut |e| e.walk(&mut |x| out.push(x.clone())));
        }
        out
    }

    #[test]
    fn builder_makes_a_kernel() {
        let mut mb = ModuleBuilder::new("t", "main", [64, 1, 1]);
        let p = mb.add_uniform_u32_struct("P", &["n", "k"]);
        let params = mb.add_uniform("p", KirType::Struct(p));
        let out = mb.add_storage_read_write("out", KirType::U32);
        let unused = mb.add_storage_read("unused", KirType::F32);
        let hist = mb.add_shared("hist", KirType::Atomic(ScalarTy::U32), 4);
        let mut b = mb.entry_builder();
        let gid = b.builtin(BuiltinVar::GlobalInvocationId);
        let i = b.field(gid, 0, KirType::U32);
        let n = b.uniform_field(params, 0, KirType::U32);
        let oob = b.ge(i, n);
        b.if_then(oob, |b| b.ret(None));
        let hi = b.mul_hi_u32(i, n);
        let (zero, one) = (b.lit_u32(0), b.lit_u32(1));
        b.for_range("j", zero, n, one, |b, j| {
            let target = AtomicTarget::Shared { shared: hist, index: Some(j.expr().boxed()) };
            b.atomic_rmw(None, IrAtomicOp::Add, ScalarTy::U32, target, one);
        });
        let f = b.bitcast(hi, KirType::F32);
        let guarded = b.opaque_guard(f + f);
        let bits = b.bitcast(guarded, KirType::U32);
        b.store(out, i, bits ^ (hi + i));
        assert!(b.finish().is_none());
        let m = mb.finish();

        assert_eq!(m.structs[p as usize].size, 8);
        assert_eq!(m.structs[p as usize].fields[1].offset, 4);
        assert_eq!(m.bindings[params as usize].min_size, Some(8));
        assert_eq!(m.bindings[out as usize].element_stride, Some(4));
        assert_eq!(m.entry_builtins, vec![(BuiltinVar::GlobalInvocationId, LocalId(0))]);
        let all = exprs(&m);
        assert!(all.iter().any(|e| matches!(e, Expr::Binary { op: IrBinOp::MulHiU32, .. })));
        assert!(all.iter().any(|e| matches!(e, Expr::Unary { op: IrUnOp::OpaqueGuard, .. })));
        assert!(stmts(&m).iter().any(|s| matches!(s, Stmt::Loop { .. })));
        assert!(stmts(&m).iter().any(|s| matches!(s, Stmt::AtomicRmw { op: IrAtomicOp::Add, .. })));
        // `unused` is never touched.
        assert_eq!(used_bindings(&m), (3, Some(vec![true, true, false].into_boxed_slice())));
        assert_eq!(unused, 2);
    }

    #[test]
    fn select_operands_and_operators() {
        let mut mb = ModuleBuilder::new("t", "main", [1, 1, 1]);
        let b = mb.entry_builder();
        let (x, y) = (b.lit_f32(1.0), b.lit_f32(2.0));
        let cond = b.lt(x, y);
        let sel = b.select(cond, x, y);
        assert!(matches!(&sel.expr(), Expr::Ternary { op: IrTernaryOp::Select, a, b, .. }
            if matches!(**a, Expr::Binary { op: IrBinOp::Lt, .. }) && **b == Expr::LitF32(1.0)));
        assert_eq!(cond.ty(), KirType::BOOL);
        assert_eq!((-x).ty(), KirType::F32);
        assert!(matches!((!cond).expr(), Expr::Unary { op: IrUnOp::Not, .. }));
        let n = b.lit_u32(3);
        assert!(matches!((!n).expr(), Expr::Unary { op: IrUnOp::BitNot, .. }));
    }

    const SAXPY: &str = "
struct P { a: f32, n: u32 }
@group(1) @binding(0) var<storage, read> x: array<f32>;
@group(1) @binding(1) var<storage, read_write> y: array<f32>;
@group(1) @binding(2) var<uniform> p: P;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    y[i] = p.a * x[i] + y[i];
}";

    #[test]
    fn saxpy() {
        let (_, m, infos, declared, mask) = parse_with(SAXPY, "main", false).unwrap();
        assert_eq!((declared, mask), (3, None));
        assert_eq!(m.workgroup_size, [64, 1, 1]);
        assert!(m.functions.is_empty());
        assert!(!m.uses_exact_f32);
        let kinds: Vec<_> = infos.iter().map(|i| (i.name.as_str(), i.kind, i.element_stride, i.min_size)).collect();
        assert_eq!(
            kinds,
            vec![
                ("x", BindingKind::StorageRead, Some(4), None),
                ("y", BindingKind::StorageReadWrite, Some(4), None),
                ("p", BindingKind::Uniform, None, Some(8)),
            ]
        );
        assert_eq!(m.bindings.iter().map(BindingDecl::info).collect::<Vec<_>>(), infos);
        assert!(m.bindings[0].is_unsized_array && !m.bindings[2].is_unsized_array);
        assert_eq!(m.entry_builtins.len(), 1);
        assert_eq!(used_bindings(&m), (3, None));
        assert!(stmts(&m).iter().any(|s| matches!(s, Stmt::Store { binding: 1, .. })));
        assert!(exprs(&m).iter().any(|e| matches!(e, Expr::Load { binding: 0, .. })));
        assert!(exprs(&m).iter().any(|e| matches!(e, Expr::UniformRead { binding: 2, .. })));
    }

    #[test]
    fn padded_vec3_offsets() {
        let m = parse(
            "struct Padded { a: f32, v: vec3<f32>, c: f32 }
@group(1) @binding(0) var<storage, read_write> out: array<f32>;
@group(1) @binding(1) var<uniform> p: Padded;
@compute @workgroup_size(1) fn main() { out[0] = p.a + p.v.x + p.v.z + p.c; }",
            "main",
        );
        let s = &m.structs[0];
        assert_eq!(s.name, "Padded");
        assert_eq!(s.fields.iter().map(|f| f.offset).collect::<Vec<_>>(), [0, 16, 28]);
        assert_eq!((s.size, s.align), (32, 16));
        assert_eq!(s.fields[1].ty, KirType::vec(ScalarTy::F32, 3));
        assert_eq!(m.bindings[1].min_size, Some(32));
    }

    const MULTI: &str = "
@group(1) @binding(0) var<storage, read_write> a: array<f32>;
@group(1) @binding(1) var<storage, read_write> b: array<f32>;
@group(2) @binding(0) var<storage, read_write> unused: array<f32>;
@compute @workgroup_size(8) fn first() { a[0] = 1.0; }
@compute @workgroup_size(8) fn second() { b[0] = 2.0; }";

    #[test]
    fn entry_points_are_scoped() {
        let (_, m, infos, declared, _) = parse_with(MULTI, "first", false).unwrap();
        assert_eq!((declared, infos[0].name.as_str(), m.bindings.len()), (1, "a", 1));
        // `second` uses @binding(1) alone: not numbered from 0.
        assert!(matches!(parse_with(MULTI, "second", false), Err(BackendError::KernelBinding(_))));
        // Trusted: the arguments are all @group(1) variables, `second` reads only the last.
        let (_, m, infos, declared, mask) = parse_with(MULTI, "second", true).unwrap();
        assert_eq!((declared, m.bindings.len(), infos.len()), (2, 2, 1));
        assert_eq!(mask.as_deref(), Some(&[false, true][..]));
        assert_eq!(used_bindings(&m), (2, mask));
    }

    #[test]
    fn grid_uniform_and_helpers() {
        let m = parse(
            "@group(1) @binding(0) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    out[luma_idx2(g.x, g.y)] = luma.size.y;
}",
            "main",
        );
        assert_eq!(m.functions.len(), 1);
        assert_eq!(m.functions[0].name, "luma_idx2");
        assert!(exprs(&m).iter().any(|e| matches!(e, Expr::GridSize(0))));
        assert!(exprs(&m).iter().any(|e| matches!(e, Expr::GridSize(1))));
        assert_eq!(m.workgroup_size, [8, 8, 1]);
        assert!(m.entry_body.iter().any(|s| matches!(s, Stmt::Store { binding: 0, .. })));
    }

    #[test]
    fn subgroup_and_atomic_workgroup_array() {
        let m = parse(
            "var<workgroup> hist: array<atomic<u32>, 32>;
@group(1) @binding(0) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(32) fn main(@builtin(local_invocation_index) li: u32) {
    atomicStore(&hist[li], 0u);
    workgroupBarrier();
    let old = atomicAdd(&hist[li % 4u], 1u);
    let ballot = subgroupBallot(li < 5u);
    let sh = subgroupShuffle(old, 0u);
    let sum = subgroupAdd(old);
    out[li] = ballot.x + sh + sum + atomicLoad(&hist[0]);
}",
            "main",
        );
        assert_eq!(m.shared.len(), 1);
        assert_eq!(m.shared[0].len, Some(32));
        assert_eq!(m.shared[0].elem_ty, KirType::Atomic(ScalarTy::U32));
        let s = stmts(&m);
        assert!(s.iter().any(|s| matches!(s, Stmt::AtomicStore { target: AtomicTarget::Shared { .. }, .. })));
        assert!(s.iter().any(|s| matches!(s, Stmt::AtomicRmw { op: IrAtomicOp::Add, dst: Some(_), .. })));
        assert!(s.iter().any(|s| matches!(s, Stmt::Barrier(BarrierScope::Workgroup))));
        let e = exprs(&m);
        for op in [IrSubgroupOp::Ballot, IrSubgroupOp::Shuffle, IrSubgroupOp::Add] {
            assert!(e.iter().any(|x| matches!(x, Expr::Subgroup { op: o, .. } if *o == op)), "{op:?}");
        }
        assert!(e.iter().any(|x| matches!(x, Expr::AtomicLoad { .. })));
        // Without the device feature naga refuses the source.
        let no_sub = parse_wgsl("@compute @workgroup_size(1) fn main() { _ = subgroupAdd(1u); }", "main", None, Features::NONE, [256; 3], 256, 8, false);
        assert!(matches!(no_sub, Err(BackendError::ShaderCompile(_))));
    }

    #[test]
    fn compare_exchange_loop() {
        let m = parse(
            "@group(1) @binding(0) var<storage, read_write> acc: array<atomic<u32>>;
@compute @workgroup_size(1) fn main() {
    var old = atomicLoad(&acc[0]);
    loop {
        let new_v = bitcast<u32>(bitcast<f32>(old) + 1.0);
        let r = atomicCompareExchangeWeak(&acc[0], old, new_v);
        if (r.exchanged) { break; }
        old = r.old_value;
    }
}",
            "main",
        );
        let s = stmts(&m);
        assert!(s.iter().any(|s| matches!(s, Stmt::Loop { .. })));
        assert!(s.iter().any(|s| matches!(s, Stmt::AtomicCompareExchange { elem_ty: ScalarTy::U32, .. })));
        assert!(s.iter().any(|s| matches!(s, Stmt::Assign { dst: LValue::Local(_), .. })));
    }

    #[test]
    fn fixed_array_return_and_math() {
        let m = parse(
            "fn f(x: f32) -> array<f32, 4> { return array<f32, 4>(x, x + 1.0, x + 2.0, x + 3.0); }
@group(1) @binding(0) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(1) fn main() {
    let a = f(1.0);
    let v = vec3<f32>(1.0, 2.0, 3.0);
    out[0] = a[3] + dot(v, v) + select(1.0, 2.0, a[0] > 0.0) + f32(u32(a[1])) + tanh(a[2]);
}",
            "main",
        );
        assert_eq!(m.functions.len(), 1);
        assert_eq!(m.functions[0].ret_ty, Some(KirType::Array { elem: Box::new(KirType::F32), len: 4, stride: 4 }));
        let e = exprs(&m);
        assert!(e.iter().any(|x| matches!(x, Expr::Ternary { op: IrTernaryOp::Select, .. })));
        assert!(e.iter().any(|x| matches!(x, Expr::Cast { kind: CastKind::FloatToIntSat, .. })));
        assert!(e.iter().any(|x| matches!(x, Expr::Unary { op: IrUnOp::Tanh, .. })));
    }

    #[test]
    fn errors() {
        let e = parse_with("fn broken(", "main", false).unwrap_err();
        assert!(matches!(e, BackendError::ShaderCompile(_)), "{e:?}");
        let e = parse_with(SAXPY, "nope", false).unwrap_err();
        assert!(matches!(&e, BackendError::KernelEntryPoint(m) if m.contains("nope") && m.contains("main")), "{e:?}");
        let e = parse_with(
            "@group(0) @binding(1) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(1) fn main() { o[0] = 1.0; }",
            "main",
            false,
        )
        .unwrap_err();
        assert!(matches!(&e, BackendError::KernelBinding(m) if m.contains("@group(0)")), "{e:?}");
        let e = parse_with(
            "@group(2) @binding(0) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(1) fn main() { o[0] = 1.0; }",
            "main",
            false,
        )
        .unwrap_err();
        assert!(matches!(&e, BackendError::KernelBinding(m) if m.contains("@group(2)")), "{e:?}");
        let big = "@compute @workgroup_size(512) fn main() {}";
        assert!(matches!(parse_with(big, "main", false), Err(BackendError::KernelWorkgroup(_))));
        let r = parse_wgsl(SAXPY, "main", Some([32, 1, 1]), Features::NONE, [256; 3], 256, 8, false);
        assert!(matches!(r, Err(BackendError::KernelWorkgroup(_))));
        let r = parse_wgsl(SAXPY, "main", None, Features::NONE, [256; 3], 256, 1, false);
        assert!(matches!(r, Err(BackendError::KernelBinding(_))));
        let r = parse_wgsl("enable wgpu_cooperative_matrix; @compute @workgroup_size(1) fn main() {}", "main", None, Features::NONE, [256; 3], 256, 8, false);
        assert!(matches!(r, Err(BackendError::Unsupported(_))));
    }

    const DF: &str = "
@group(1) @binding(0) var<storage, read> x: array<f32>;
@group(1) @binding(1) var<storage, read_write> out: array<f32>;
@group(1) @binding(2) var<storage, read_write> hi: array<u32>;
fn luma_o(x: f32) -> f32 { return bitcast<f32>(bitcast<u32>(x) | 0u); }
fn luma_mulhi(a: u32, b: u32) -> u32 {
    let a0 = a & 0xFFFFu; let a1 = a >> 16u;
    let b0 = b & 0xFFFFu; let b1 = b >> 16u;
    let m1 = a1 * b0; let m2 = a0 * b1;
    let carry = (((a0 * b0) >> 16u) + (m1 & 0xFFFFu) + (m2 & 0xFFFFu)) >> 16u;
    return a1 * b1 + (m1 >> 16u) + (m2 >> 16u) + carry;
}
fn dead() -> u32 { return 7u; }
@compute @workgroup_size(1) fn main() {
    let a = x[0]; let b = x[1];
    out[0] = luma_o(a + b);
    out[1] = luma_o(luma_o(a * b) - a);
    out[2] = luma_o(a / b);
    hi[0] = luma_mulhi(3u, 5u);
}";

    #[test]
    fn optimizer_folds_helpers() {
        let m = parse(DF, "main");
        assert!(m.uses_exact_f32);
        // Naga drops `dead`: only what `main` reaches is lowered.
        assert_eq!(m.functions.iter().map(|f| f.name.as_str()).collect::<Vec<_>>(), ["luma_o", "luma_mulhi"]);

        let cuda = optimize_for_target(&m, true);
        assert!(matches!(cuda, Cow::Owned(_)));
        assert!(cuda.functions.is_empty());
        let e = exprs(&cuda);
        assert!(!e.iter().any(|x| matches!(x, Expr::Call { .. } | Expr::Unary { op: IrUnOp::OpaqueGuard, .. })));
        for op in [IrBinOp::AddExact, IrBinOp::MulExact, IrBinOp::SubExact, IrBinOp::MulHiU32] {
            assert!(e.iter().any(|x| matches!(x, Expr::Binary { op: o, .. } if *o == op)), "{op:?}");
        }
        // `luma_o(a / b)` is just `a / b`.
        assert!(e.iter().any(|x| matches!(x, Expr::Binary { op: IrBinOp::Div, .. })));

        let metal = optimize_for_target(&m, false);
        assert!(metal.functions.is_empty());
        let e = exprs(&metal);
        assert!(!e.iter().any(|x| matches!(x, Expr::Call { .. })));
        assert_eq!(e.iter().filter(|x| matches!(x, Expr::Unary { op: IrUnOp::OpaqueGuard, .. })).count(), 4);
        assert!(e.iter().any(|x| matches!(x, Expr::Binary { op: IrBinOp::MulHiU32, .. })));
        assert!(!e.iter().any(|x| matches!(x, Expr::Binary { op: IrBinOp::AddExact, .. })));
        // Bindings are untouched.
        assert_eq!(used_bindings(&cuda), used_bindings(&m));
    }

    #[test]
    fn optimizer_prunes_dead_helpers_and_borrows_otherwise() {
        let mut mb = ModuleBuilder::new("t", "main", [1, 1, 1]);
        let out = mb.add_storage_read_write("out", KirType::U32);
        let dead = mb.add_function("dead", &[], Some(KirType::U32), |b| {
            let v = b.lit_u32(9);
            b.ret(Some(v));
        });
        let live = mb.add_function("live", &[("x", KirType::U32)], Some(KirType::U32), |b| {
            let x = b.param(0);
            let one = b.lit_u32(1);
            b.ret(Some(x + one));
        });
        assert_eq!((dead, live), (FnId(0), FnId(1)));
        let mut b = mb.entry_builder();
        let (zero, five) = (b.lit_u32(0), b.lit_u32(5));
        let r = b.call(live, &[five], KirType::U32);
        b.store(out, zero, r);
        b.finish();
        let m = mb.finish();

        let pruned = optimize_for_target(&m, false);
        assert!(matches!(pruned, Cow::Owned(_)));
        assert_eq!(pruned.functions.len(), 1);
        assert_eq!(pruned.functions[0].name, "live");
        assert!(exprs(&pruned).iter().any(|x| matches!(x, Expr::Call { func: FnId(0), .. })));

        // Already minimal: nothing to do.
        assert!(matches!(optimize_for_target(&pruned, true), Cow::Borrowed(_)));
    }

    /// Evaluates straight-line scalar kernels: the `Store`s of `m`'s entry point, by
    /// `(binding, index)`, given the contents of each binding.
    fn run_scalar(m: &KernelModule, inputs: &[&[f64]]) -> std::collections::HashMap<(u32, usize), f64> {
        use std::collections::HashMap;
        fn eval(e: &Expr, env: &HashMap<LocalId, f64>, inputs: &[&[f64]]) -> f64 {
            match e {
                Expr::LitF32(v) => f64::from(*v),
                Expr::LitU32(v) => f64::from(*v),
                Expr::LitI32(v) => f64::from(*v),
                Expr::Local(l) => env[l],
                Expr::Load { binding, index, .. } => inputs[*binding as usize][eval(index, env, inputs) as usize],
                Expr::Binary { op, lhs, rhs, .. } => {
                    let (a, b) = (eval(lhs, env, inputs), eval(rhs, env, inputs));
                    match op {
                        IrBinOp::Add => a + b,
                        IrBinOp::Sub => a - b,
                        IrBinOp::Mul => a * b,
                        IrBinOp::Div => a / b,
                        other => panic!("{other:?}"),
                    }
                }
                Expr::Ternary { op: IrTernaryOp::Clamp, a, b, c, .. } => {
                    eval(a, env, inputs).clamp(eval(b, env, inputs), eval(c, env, inputs))
                }
                other => panic!("{other:?}"),
            }
        }
        let mut env = HashMap::new();
        let mut out = HashMap::new();
        for s in &m.entry_body {
            match s {
                Stmt::Let { local, init } => {
                    env.insert(*local, eval(init, &env, inputs));
                }
                Stmt::Store { binding, index, val } => {
                    out.insert((*binding, eval(index, &env, inputs) as usize), eval(val, &env, inputs));
                }
                Stmt::Return(None) => {}
                other => panic!("{other:?}"),
            }
        }
        out
    }

    #[test]
    fn smoothstep_takes_edges_then_value() {
        // Loaded operands: naga cannot fold the call away.
        let m = parse(
            "@group(1) @binding(0) var<storage, read> x: array<f32>;
@group(1) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(1) fn main() { out[0] = smoothstep(x[0], x[1], x[2]); }",
            "main",
        );
        // t = (0.5 - 0) / (2 - 0) = 0.25; t * t * (3 - 2 * t) = 0.15625.
        let got = run_scalar(&m, &[&[0.0, 2.0, 0.5], &[]])[&(1, 0)];
        assert!((got - 0.15625).abs() < 1e-12, "{got}");
        // Outside the edges it saturates.
        assert_eq!(run_scalar(&m, &[&[0.0, 2.0, 5.0], &[]])[&(1, 0)], 1.0);
        assert_eq!(run_scalar(&m, &[&[0.0, 2.0, -1.0], &[]])[&(1, 0)], 0.0);
    }

    /// The initialiser of the immutable local called `name` in the entry point.
    fn let_init<'a>(m: &'a KernelModule, name: &str) -> &'a Expr {
        m.entry_body
            .iter()
            .find_map(|s| match s {
                Stmt::Let { local, init } if m.entry_locals[local.0 as usize].0 == name => Some(init),
                _ => None,
            })
            .unwrap_or_else(|| panic!("no `let {name}`"))
    }

    fn count_binary(m: &KernelModule, op: IrBinOp) -> usize {
        exprs(m).iter().filter(|x| matches!(x, Expr::Binary { op: o, .. } if *o == op)).count()
    }

    #[test]
    fn guard_of_a_shared_local_makes_its_definition_exact() {
        let m = parse(
            "@group(1) @binding(0) var<storage, read> x: array<f32>;
@group(1) @binding(1) var<storage, read_write> out: array<f32>;
fn luma_o(v: f32) -> f32 { return bitcast<f32>(bitcast<u32>(v) | 0u); }
@compute @workgroup_size(1) fn main() {
    let a = x[0]; let b = x[1]; let c = x[2];
    let p = a * b;
    out[0] = luma_o(p);
    out[1] = p + c;
    let s = a + c;
    out[2] = luma_o(luma_o(s) - b);
    out[3] = s * c;
    out[4] = a - c;
}",
            "main",
        );
        // Named values: `p` and `s` are used twice, so they are `let`s the guards only see by name.
        assert!(matches!(let_init(&m, "p"), Expr::Binary { op: IrBinOp::Mul, .. }));
        assert!(matches!(let_init(&m, "s"), Expr::Binary { op: IrBinOp::Add, .. }));

        let cuda = optimize_for_target(&m, true);
        assert!(matches!(let_init(&cuda, "p"), Expr::Binary { op: IrBinOp::MulExact, .. }));
        assert!(matches!(let_init(&cuda, "s"), Expr::Binary { op: IrBinOp::AddExact, .. }));
        // The nested guard: `luma_o(luma_o(s) - b)` is an exact difference of the exact sum.
        assert_eq!(count_binary(&cuda, IrBinOp::SubExact), 1);
        assert_eq!((count_binary(&cuda, IrBinOp::AddExact), count_binary(&cuda, IrBinOp::MulExact)), (1, 1));
        // Arithmetic nobody guarded is untouched: `p + c`, `s * c`, `a - c`.
        assert_eq!(
            (count_binary(&cuda, IrBinOp::Add), count_binary(&cuda, IrBinOp::Mul), count_binary(&cuda, IrBinOp::Sub)),
            (1, 1, 1)
        );
        assert!(cuda.functions.is_empty());

        // A target without exact operations keeps the arithmetic and guards the values.
        let metal = optimize_for_target(&m, false);
        assert!(matches!(let_init(&metal, "p"), Expr::Binary { op: IrBinOp::Mul, .. }));
        assert_eq!(count_binary(&metal, IrBinOp::MulExact) + count_binary(&metal, IrBinOp::AddExact), 0);
        assert_eq!(exprs(&metal).iter().filter(|x| matches!(x, Expr::Unary { op: IrUnOp::OpaqueGuard, .. })).count(), 3);
    }

    #[test]
    fn guard_follows_local_aliases_but_not_variables() {
        let mut mb = ModuleBuilder::new("t", "main", [1, 1, 1]);
        let x = mb.add_storage_read("x", KirType::F32);
        let out = mb.add_storage_read_write("out", KirType::F32);
        let o = mb.add_function("luma_o", &[("v", KirType::F32)], Some(KirType::F32), |b| {
            let v = b.param(0);
            b.ret(Some(v));
        });
        let mut b = mb.entry_builder();
        let (i0, i1, i2) = (b.lit_u32(0), b.lit_u32(1), b.lit_u32(2));
        let a = b.load(x, i0);
        let a = b.let_val("a", a);
        let c = b.load(x, i1);
        let c = b.let_val("c", c);
        let product = b.let_val("p", a * c);
        let alias = b.let_val("q", product);
        // A mutable variable holding a product is not a value a guard can pin down.
        let var = b.var_decl("v", KirType::F32, Some(a * c));
        let guarded = b.call(o, &[alias], KirType::F32);
        b.store(out, i0, guarded);
        let var_val = b.read_var(var);
        let guarded_var = b.call(o, &[var_val], KirType::F32);
        b.store(out, i2, guarded_var);
        b.store(out, i1, product + a);
        b.finish();
        let m = mb.finish();

        let cuda = optimize_for_target(&m, true);
        assert!(matches!(let_init(&cuda, "p"), Expr::Binary { op: IrBinOp::MulExact, .. }));
        assert!(matches!(let_init(&cuda, "q"), Expr::Local(_)));
        let var_init = cuda.entry_body.iter().find_map(|s| match s {
            Stmt::Var { init: Some(e), .. } => Some(e),
            _ => None,
        });
        assert!(matches!(var_init, Some(Expr::Binary { op: IrBinOp::Mul, .. })));
        // `product + a` stays a plain add.
        assert_eq!((count_binary(&cuda, IrBinOp::Add), count_binary(&cuda, IrBinOp::MulExact)), (1, 1));
    }

    /// A value of the small interpreter below: scalar, vector or array (a matrix is an array of
    /// its columns).
    #[derive(Clone, Debug)]
    enum V {
        S(f64),
        Vec(Vec<f64>),
        Arr(Vec<V>),
    }

    type Env = std::collections::HashMap<LocalId, V>;

    fn flat(v: &V, out: &mut Vec<f64>) {
        match v {
            V::S(x) => out.push(*x),
            V::Vec(xs) => out.extend(xs),
            V::Arr(items) => items.iter().for_each(|i| flat(i, out)),
        }
    }

    fn zip(a: &V, b: &V, f: &dyn Fn(f64, f64) -> f64) -> V {
        match (a, b) {
            (V::S(x), V::S(y)) => V::S(f(*x, *y)),
            (V::Vec(x), V::S(y)) => V::Vec(x.iter().map(|x| f(*x, *y)).collect()),
            (V::S(x), V::Vec(y)) => V::Vec(y.iter().map(|y| f(*x, *y)).collect()),
            (V::Vec(x), V::Vec(y)) => V::Vec(x.iter().zip(y).map(|(x, y)| f(*x, *y)).collect()),
            _ => panic!("{a:?} {b:?}"),
        }
    }

    /// Evaluates `e` with `x` as the contents of binding 0.
    fn eval_v(e: &Expr, env: &Env, x: &[f64]) -> V {
        let index = |e: &Expr| match eval_v(e, env, x) {
            V::S(i) => i as usize,
            other => panic!("{other:?}"),
        };
        match e {
            Expr::LitF32(v) => V::S(f64::from(*v)),
            Expr::LitU32(v) => V::S(f64::from(*v)),
            Expr::LitI32(v) => V::S(f64::from(*v)),
            Expr::Local(l) => env[l].clone(),
            Expr::Load { binding: 0, index: i, .. } => V::S(x[index(i)]),
            Expr::Compose { ty, comps } => {
                let vals: Vec<V> = comps.iter().map(|c| eval_v(c, env, x)).collect();
                if matches!(ty, KirType::Array { .. }) {
                    V::Arr(vals)
                } else {
                    let mut out = Vec::new();
                    vals.iter().for_each(|v| flat(v, &mut out));
                    V::Vec(out)
                }
            }
            Expr::Access { base, step, .. } => {
                let k = match step {
                    AccessStep::Field(k) => *k as usize,
                    AccessStep::Index(i) => index(i),
                };
                match eval_v(base, env, x) {
                    V::Arr(items) => items[k].clone(),
                    V::Vec(xs) => V::S(xs[k]),
                    other => panic!("{other:?}"),
                }
            }
            Expr::Unary { op: IrUnOp::Neg, arg, .. } => zip(&eval_v(arg, env, x), &V::S(0.0), &|a, _| -a),
            Expr::Binary { op, lhs, rhs, .. } => {
                let (a, b) = (eval_v(lhs, env, x), eval_v(rhs, env, x));
                match op {
                    IrBinOp::Add => zip(&a, &b, &|p, q| p + q),
                    IrBinOp::Sub => zip(&a, &b, &|p, q| p - q),
                    IrBinOp::Mul => zip(&a, &b, &|p, q| p * q),
                    IrBinOp::Div => zip(&a, &b, &|p, q| p / q),
                    other => panic!("{other:?}"),
                }
            }
            other => panic!("{other:?}"),
        }
    }

    /// Runs the straight-line entry point of `m`, returning what it stores to binding 1.
    fn run_values(m: &KernelModule, x: &[f64]) -> Vec<f64> {
        let mut env = Env::new();
        let mut out = std::collections::BTreeMap::new();
        for s in &m.entry_body {
            match s {
                Stmt::Let { local, init } => {
                    let v = eval_v(init, &env, x);
                    env.insert(*local, v);
                }
                Stmt::Store { binding: 1, index, val } => {
                    let (V::S(i), V::S(v)) = (eval_v(index, &env, x), eval_v(val, &env, x)) else { panic!("scalar store") };
                    out.insert(i as usize, v);
                }
                Stmt::Return(None) => {}
                other => panic!("{other:?}"),
            }
        }
        assert!(out.keys().copied().eq(0..out.len()), "stores are not contiguous");
        out.into_values().collect()
    }

    /// Column-major matrices of the reference below: one `Vec` per column.
    type Cols = Vec<Vec<f64>>;

    fn cols(x: &[f64], n: usize, rows: usize) -> Cols {
        (0..n).map(|c| x[c * rows..(c + 1) * rows].to_vec()).collect()
    }

    fn m_vec(m: &Cols, v: &[f64]) -> Vec<f64> {
        (0..m[0].len()).map(|r| (0..m.len()).map(|c| m[c][r] * v[c]).sum()).collect()
    }

    fn m_mul(a: &Cols, b: &Cols) -> Cols {
        b.iter().map(|c| m_vec(a, c)).collect()
    }

    fn flat_cols(m: &Cols) -> Vec<f64> {
        m.iter().flatten().copied().collect()
    }

    /// Determinant by elimination with partial pivoting, independent of the Laplace expansion
    /// the lowering emits.
    fn det(m: &Cols) -> f64 {
        let n = m.len();
        let mut a: Vec<Vec<f64>> = (0..n).map(|r| (0..n).map(|c| m[c][r]).collect()).collect();
        let mut det = 1.0;
        for i in 0..n {
            let p = (i..n).max_by(|&p, &q| a[p][i].abs().total_cmp(&a[q][i].abs())).expect("rows");
            if a[p][i] == 0.0 {
                return 0.0;
            }
            if p != i {
                a.swap(p, i);
                det = -det;
            }
            det *= a[i][i];
            for r in i + 1..n {
                let f = a[r][i] / a[i][i];
                for c in i..n {
                    let t = a[i][c];
                    a[r][c] -= f * t;
                }
            }
        }
        det
    }

    fn vec_outs(name: &str, lanes: usize) -> Vec<String> {
        (0..lanes).map(|i| format!("{name}[{i}]")).collect()
    }

    fn mat_outs(name: &str, cols: usize, rows: usize) -> Vec<String> {
        (0..cols).flat_map(|c| (0..rows).map(move |r| format!("{name}[{c}][{r}]"))).collect()
    }

    fn matrix_kernel(lets: &str, outs: &[String]) -> String {
        let stores: String = outs.iter().enumerate().map(|(i, e)| format!("    out[{i}] = {e};\n")).collect();
        format!(
            "@group(1) @binding(0) var<storage, read> x: array<f32>;
@group(1) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(1) fn main() {{
{lets}
{stores}}}"
        )
    }

    fn assert_close(got: &[f64], want: &[f64]) {
        assert_eq!(got.len(), want.len(), "{got:?} vs {want:?}");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!((g - w).abs() < 1e-9, "out[{i}] = {g}, expected {w}\n{got:?}\n{want:?}");
        }
    }

    /// Matrices are lowered away: no arithmetic on a whole matrix survives.
    fn assert_no_matrix_ops(m: &KernelModule) {
        let whole = |ty: &KirType| matches!(ty, KirType::Array { .. });
        for e in exprs(m) {
            match e {
                Expr::Binary { ty, .. } | Expr::Unary { ty, .. } => assert!(!whole(&ty), "{ty:?}"),
                _ => {}
            }
        }
    }

    const X: [f64; 16] = [2., 3., 1., 0., 4., 5., 6., 1., 2., 3., 7., 2., 1., 0., 2., 5.];

    #[test]
    fn matrix_products_match_a_reference() {
        // a: 3 columns of 2 rows, b: 2 columns of 3 rows; every product has a different shape.
        let lets = "let a = mat3x2<f32>(vec2<f32>(x[0], x[1]), vec2<f32>(x[2], x[3]), vec2<f32>(x[4], x[5]));
    let b = mat2x3<f32>(vec3<f32>(x[0], x[1], x[2]), vec3<f32>(x[3], x[4], x[5]));
    let v = vec3<f32>(x[6], x[7], x[8]);
    let w = vec2<f32>(x[9], x[10]);
    let mv = a * v;
    let vm = w * a;
    let ab = a * b;
    let ba = b * a;";
        let outs: Vec<String> =
            [vec_outs("mv", 2), vec_outs("vm", 3), mat_outs("ab", 2, 2), mat_outs("ba", 3, 3)].into_iter().flatten().collect();
        let m = parse(&matrix_kernel(lets, &outs), "main");
        assert_no_matrix_ops(&m);

        let (a, b) = (cols(&X, 3, 2), cols(&X, 2, 3));
        let (v, w) = (&X[6..9], &X[9..11]);
        let mut want = m_vec(&a, v);
        want.extend((0..3).map(|j| w.iter().zip(&a[j]).map(|(p, q)| p * q).sum::<f64>()));
        want.extend(flat_cols(&m_mul(&a, &b)));
        want.extend(flat_cols(&m_mul(&b, &a)));
        assert_close(&run_values(&m, &X), &want);
    }

    #[test]
    fn matrix_arithmetic_transpose_and_determinant() {
        let lets = "let a = mat3x2<f32>(vec2<f32>(x[0], x[1]), vec2<f32>(x[2], x[3]), vec2<f32>(x[4], x[5]));
    let c = 3.0 * a - a;
    let d = a * -0.5;
    let t = transpose(a);
    let m2 = mat2x2<f32>(vec2<f32>(x[0], x[1]), vec2<f32>(x[2], x[3]));
    let m3 = mat3x3<f32>(vec3<f32>(x[0], x[1], x[2]), vec3<f32>(x[3], x[4], x[5]), vec3<f32>(x[6], x[7], x[8]));
    let m4 = mat4x4<f32>(vec4<f32>(x[0], x[1], x[2], x[3]), vec4<f32>(x[4], x[5], x[6], x[7]), vec4<f32>(x[8], x[9], x[10], x[11]), vec4<f32>(x[12], x[13], x[14], x[15]));
    let d2 = determinant(m2);
    let d3 = determinant(m3);
    let d4 = determinant(m4);";
        let mut outs = mat_outs("c", 3, 2);
        outs.extend(mat_outs("d", 3, 2));
        outs.extend(mat_outs("t", 2, 3));
        outs.extend(["d2", "d3", "d4"].map(String::from));
        let m = parse(&matrix_kernel(lets, &outs), "main");
        assert_no_matrix_ops(&m);

        let a = cols(&X, 3, 2);
        let mut want: Vec<f64> = flat_cols(&a).iter().map(|v| 2.0 * v).collect();
        want.extend(flat_cols(&a).iter().map(|v| -0.5 * v));
        let t: Cols = (0..2).map(|r| (0..3).map(|c| a[c][r]).collect()).collect();
        want.extend(flat_cols(&t));
        want.extend([det(&cols(&X, 2, 2)), det(&cols(&X, 3, 3)), det(&cols(&X, 4, 4))]);
        assert!(want[18..].iter().all(|d| d.abs() > 1.0), "{:?}", &want[18..]);
        assert_close(&run_values(&m, &X), &want);
    }

    #[test]
    fn matrix_types_lower_to_column_arrays() {
        let m = parse(
            "struct U { m: mat3x3<f32>, k: f32 }
@group(1) @binding(0) var<uniform> u: U;
@group(1) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(1) fn main() { out[0] = (u.m * vec3<f32>(1.0, 0.0, 0.0)).y + u.k; }",
            "main",
        );
        let col = KirType::vec(ScalarTy::F32, 3);
        let s = &m.structs[0];
        assert_eq!(s.fields[0].ty, KirType::Array { elem: Box::new(col), len: 3, stride: 16 });
        assert_eq!(s.fields.iter().map(|f| f.offset).collect::<Vec<_>>(), [0, 48]);
        assert_eq!((s.size, s.align), (64, 16));
        assert_eq!(m.bindings[0].min_size, Some(64));
        assert_no_matrix_ops(&m);
        // The uniform's matrix is read once and named, then taken apart by column.
        assert!(exprs(&m).iter().any(|e| matches!(e, Expr::UniformRead { binding: 0, .. })));
    }
}
