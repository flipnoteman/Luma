//! Analysis for workgroup staging of sub-programs read through translation views.
//!
//! An expression read through views at several offsets (a face flux read by the two cells that
//! share the face) is recomputed at every read when the kernel runs the lowered program of
//! `jit::inline`. When the reads are plain translations of 2-D (or 1-D) arrays, a workgroup can
//! instead compute the expression once per cell of the region its tile reads (the tile plus a
//! halo) into workgroup memory and let the reads come from there. This module decides which
//! sub-programs can be staged ([`candidates`]) and lays the staging out ([`plan`]): the tile,
//! the region of every staged sub-program relative to the tile origin, and the order in which
//! sub-programs that read each other are computed. `codegen::staged_main` writes the kernel,
//! `inline::lower_staged` lowers the programs it runs and `inline::judge` weighs it against the
//! other ways of running the program. [`check`] validates a layout against its key.
//!
//! Everything is expressed in one translated frame: the cell of a staged sub-program at local
//! coordinate `(x, y)` is the cell `tile origin + (x, y)` of its own array, and a reader at
//! local `(x, y)` that reads it at offset `o` needs the local cell `(x, y) + o`. The main
//! program's region is the tile, `[0, tx - 1] x [0, ty - 1]`.

use crate::tuning::Tuning;

use super::program::{FusedKey, Instr, Ref, Tail};
use super::{Edge, OpKind, ShiftKind};

/// How a store kernel stages some of its sub-programs in workgroup memory. Part of the
/// [`FusedKey`] because the generated source bakes the layout in.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub(crate) struct Staging {
    /// Output dims (x, y); y = 1 for 1-D.
    pub(crate) dims: [u32; 2],
    /// Output cells per workgroup (x, y).
    pub(crate) tile: [u32; 2],
    /// Staged sub-programs, dependencies first.
    pub(crate) subs: Vec<StagedSub>,
    /// `subs[levels[l]..levels[l + 1]]` are computed between the same two barriers.
    pub(crate) levels: Vec<u32>,
}

/// One staged sub-program.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub(crate) struct StagedSub {
    pub(crate) sub: u32,
    /// The sub-program's own dims (x, y).
    pub(crate) dims: [u32; 2],
    /// Region, relative to the tile origin in output coordinates, inclusive: x in `lo[0]..=hi[0]`,
    /// y in `lo[1]..=hi[1]`.
    pub(crate) lo: [i32; 2],
    pub(crate) hi: [i32; 2],
}

impl StagedSub {
    /// Cells of the region along x.
    pub(crate) fn width(&self) -> u32 {
        (self.hi[0] - self.lo[0] + 1) as u32
    }

    /// Cells of the region.
    pub(crate) fn cells(&self) -> u32 {
        self.width() * (self.hi[1] - self.lo[1] + 1) as u32
    }
}

impl Staging {
    /// The staged sub-program `sub`.
    pub(crate) fn find(&self, sub: u32) -> Option<&StagedSub> {
        self.subs.iter().find(|staged| staged.sub == sub)
    }

    /// Workgroups of a launch.
    pub(crate) fn tiles(&self) -> u32 {
        self.dims[0].div_ceil(self.tile[0]) * self.dims[1].div_ceil(self.tile[1])
    }
}

/// The program a read happens in.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub(crate) enum Reader {
    Main,
    Sub(u32),
}

/// A read of a staged sub-program by the main program or by another staged one. A sub-program
/// that is not staged is computed inside the programs that read it, at the cell they read it
/// at, so what it reads is read from there: the offsets of the translations on the way add up.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub(crate) struct Read {
    pub(crate) reader: Reader,
    pub(crate) sub: u32,
    /// Cell read from the cell at `(x, y)`: `(x, y) + offset`.
    pub(crate) offset: [i32; 2],
}

/// The sub-programs that can be staged together, and how they are read.
#[derive(Debug)]
pub(crate) struct Candidates {
    /// Output dims (x, y) of the kernel.
    pub(crate) dims: [u32; 2],
    /// Per sub-program, its dims if it is a candidate.
    pub(crate) sub_dims: Vec<Option<[u32; 2]>>,
    /// The reads of candidates.
    pub(crate) reads: Vec<Read>,
}

/// What a qualifying read looks like: see [`translation`].
struct Translation {
    src: [u32; 2],
    out: [u32; 2],
    offset: [i32; 2],
}

/// The read `kind` is a translation of a 2-D array: x and y map to themselves with unit step,
/// no edge handling, nothing in the other two axes.
fn translation(kind: ShiftKind) -> Option<Translation> {
    let ShiftKind::View { map, edge: Edge::None } = kind else { return None };
    let [x, y, ..] = map.axes;
    let plain = x.from == 0 && x.step == 1 && y.from == 1 && y.step == 1;
    let flat = map.src[2] == 1 && map.src[3] == 1 && map.out[2] == 1 && map.out[3] == 1;
    (plain && flat && map.fits() && map.in_range()).then_some(Translation {
        src: [map.src[0], map.src[1]],
        out: [map.out[0], map.out[1]],
        offset: [x.start, y.start],
    })
}

/// What the reads of sub-programs in the programs of a key show before anything is staged.
struct Scan {
    /// Output dims (x, y) of the main program's views.
    main: [u32; 2],
    /// Per sub-program, its dims as its readers see it, if they agree.
    dims: Vec<Option<[u32; 2]>>,
    /// Per sub-program, whether it cannot be staged: some read of it is not a translation of
    /// its dims, or it only views another sub-program (staging it would copy that one).
    bad: Vec<bool>,
}

/// The reads of `key` if it is a store kernel without segments or circular-shift tiling that
/// reads sub-programs through views, and the main program's views all have the same output dims.
fn scan(key: &FusedKey) -> Option<Scan> {
    if key.tail != Tail::Store || !key.spans.is_empty() || key.tile.is_some() || !key.inlines() {
        return None;
    }
    let n = key.subs.len();
    let mut bad = vec![false; n];
    let mut dims: Vec<Option<[u32; 2]>> = vec![None; n];
    let mut main_out: Option<[u32; 2]> = None;
    let programs = std::iter::once((Reader::Main, key.instrs.as_slice()))
        .chain(key.subs.iter().enumerate().map(|(t, program)| (Reader::Sub(t as u32), program.as_slice())));
    for (reader, program) in programs {
        for instr in program {
            let OpKind::Shift(kind) = instr.op.kind else { continue };
            let Ref::Sub(s) = instr.args[0] else { continue };
            let s = s as usize;
            if s >= n || reader == Reader::Sub(s as u32) {
                return None;
            }
            if reader == Reader::Main
                && let ShiftKind::View { map, .. } = kind
            {
                let out = [map.out[0], map.out[1]];
                if map.out[2] != 1 || map.out[3] != 1 || main_out.is_some_and(|known| known != out) {
                    return None;
                }
                main_out = Some(out);
            }
            match translation(kind) {
                Some(read) if dims[s].is_none_or(|known| known == read.src) => dims[s] = Some(read.src),
                _ => bad[s] = true,
            }
        }
    }
    for (s, program) in key.subs.iter().enumerate() {
        if let [instr] = program.as_slice()
            && matches!(instr.op.kind, OpKind::Shift(_))
            && matches!(instr.args[0], Ref::Sub(_))
        {
            bad[s] = true;
        }
    }
    let [w, h] = main_out.filter(|&[w, h]| w > 0 && h > 0)?;
    Some(Scan { main: [w, h], dims, bad })
}

/// The reads of the staged sub-programs (`staged`) that the main program and every staged
/// sub-program make, with the sub-programs that are not staged followed into: `broken[s]` is set
/// for a staged `s` read at a cell that is not a translation of its reader's (through edge
/// handling, a gather or a stride on the way), which cannot be served from workgroup memory.
fn collect(key: &FusedKey, staged: &[bool], scan: &Scan) -> (Vec<Read>, Vec<bool>) {
    struct Walk<'a> {
        key: &'a FusedKey,
        staged: &'a [bool],
        dims: &'a [Option<[u32; 2]>],
        reads: Vec<Read>,
        broken: Vec<bool>,
        /// The programs walked from the current reader, by the offset they stand at.
        done: Vec<(u32, Option<[i32; 2]>)>,
    }
    impl Walk<'_> {
        /// Walks `program` (computing cells of `extent`) standing at `offset` cells from the
        /// cell of `frame`, the program the code is generated in.
        fn walk(&mut self, frame: Reader, program: &[Instr], extent: Option<[u32; 2]>, offset: Option<[i32; 2]>) {
            let key = self.key;
            for instr in program {
                let OpKind::Shift(kind) = instr.op.kind else { continue };
                let Ref::Sub(s) = instr.args[0] else { continue };
                let s = s as usize;
                let Some(&dims) = self.dims.get(s) else { continue };
                let hop = translation(kind).filter(|hop| Some(hop.out) == extent && Some(hop.src) == dims);
                let at = offset.zip(hop).map(|(from, hop)| [from[0] + hop.offset[0], from[1] + hop.offset[1]]);
                if self.staged[s] {
                    match at {
                        Some(offset) => self.reads.push(Read { reader: frame, sub: s as u32, offset }),
                        None => self.broken[s] = true,
                    }
                } else if !self.done.contains(&(s as u32, at))
                    && let Some(inner) = key.subs.get(s)
                {
                    self.done.push((s as u32, at));
                    self.walk(frame, inner, dims, at);
                }
            }
        }
    }
    let n = key.subs.len();
    let mut walk = Walk { key, staged, dims: &scan.dims, reads: Vec::new(), broken: vec![false; n], done: Vec::new() };
    walk.walk(Reader::Main, &key.instrs, Some(scan.main), Some([0, 0]));
    for (s, program) in key.subs.iter().enumerate().filter(|&(s, _)| staged[s]) {
        walk.done.clear();
        walk.walk(Reader::Sub(s as u32), program, scan.dims[s], Some([0, 0]));
    }
    let Walk { mut reads, broken, .. } = walk;
    reads.sort();
    reads.dedup();
    (reads, broken)
}

/// The sub-programs of `key` that can be staged, if any: `key` is a store kernel without
/// segments or circular-shift tiling whose sub-programs are read through views, the main
/// program's views all have the same output dims, and a candidate is read only at cells that are
/// translations of the reader's (a read by a program that is not staged counts as a read by the
/// programs that compute it: see [`Read`]), by the main program or by candidates, at several
/// offsets or by a candidate (one offset read only by the main program shares nothing).
pub(crate) fn candidates(key: &FusedKey) -> Option<Candidates> {
    let scan = scan(key)?;
    let n = key.subs.len();
    let mut staged: Vec<bool> = (0..n).map(|s| !scan.bad[s] && scan.dims[s].is_some()).collect();
    // Dropping a candidate turns it into a program its readers compute: what it reads is then
    // read from where they are, so the reads are collected again.
    let reads = loop {
        let (reads, broken) = collect(key, &staged, &scan);
        let mut changed = false;
        for s in 0..n {
            if !staged[s] {
                continue;
            }
            let mut mine = reads.iter().filter(|read| read.sub as usize == s);
            let first = mine.next();
            let shares = first.is_some_and(|first| first.reader != Reader::Main || mine.any(|read| read.reader != Reader::Main || read.offset != first.offset));
            if broken[s] || !shares {
                staged[s] = false;
                changed = true;
            }
        }
        if !changed {
            break reads;
        }
    };
    if reads.is_empty() {
        return None;
    }
    let sub_dims = (0..n).map(|s| scan.dims[s].filter(|_| staged[s])).collect();
    Some(Candidates { dims: scan.main, sub_dims, reads })
}

/// Inclusive corners (low, high) of a rectangle of cells.
type Region = ([i32; 2], [i32; 2]);

/// The region (inclusive corners) of every candidate for tiles of `tile` output cells.
fn regions(c: &Candidates, tile: [u32; 2]) -> Vec<Option<Region>> {
    /// Grows the region of the sub-program `read` reads to cover what a reader at `from` reaches.
    fn reach(region: &mut [Option<Region>], from: Region, read: &Read) {
        let lo = [from.0[0] + read.offset[0], from.0[1] + read.offset[1]];
        let hi = [from.1[0] + read.offset[0], from.1[1] + read.offset[1]];
        let slot = &mut region[read.sub as usize];
        *slot = Some(match *slot {
            None => (lo, hi),
            Some((l, h)) => ([l[0].min(lo[0]), l[1].min(lo[1])], [h[0].max(hi[0]), h[1].max(hi[1])]),
        });
    }
    let n = c.sub_dims.len();
    let mut region: Vec<Option<Region>> = vec![None; n];
    let main = ([0, 0], [tile[0] as i32 - 1, tile[1] as i32 - 1]);
    for read in c.reads.iter().filter(|read| read.reader == Reader::Main) {
        reach(&mut region, main, read);
    }
    // Sub-programs only read lower-numbered ones: a reader's region is final before it is used.
    for t in (0..n).rev() {
        if let Some(own) = region[t] {
            for read in c.reads.iter().filter(|read| read.reader == Reader::Sub(t as u32)) {
                reach(&mut region, own, read);
            }
        }
    }
    region
}

/// Lays out the staging of every candidate of `c`: the tile (the tuning's stencil tile, halved
/// until the regions fit its workgroup-memory budget; `None` below 8 cells a side), the regions
/// and the levels. `elem_bytes(s)` is the size of one value of sub-program `s`.
pub(crate) fn plan(c: &Candidates, tuning: &Tuning, elem_bytes: impl Fn(u32) -> u64) -> Option<Staging> {
    let n = c.sub_dims.len();
    let one_d = c.dims[1] == 1;
    let (mut tx, mut ty) = if one_d { (tuning.stencil_tile_1d, 1) } else { tuning.stencil_tile_2d };
    let region = loop {
        let region = regions(c, [tx, ty]);
        let bytes: u64 = region
            .iter()
            .enumerate()
            .filter_map(|(s, r)| r.map(|(lo, hi)| (hi[0] - lo[0] + 1) as u64 * (hi[1] - lo[1] + 1) as u64 * elem_bytes(s as u32)))
            .sum();
        if bytes <= u64::from(tuning.stencil_tile_bytes) {
            break region;
        }
        if one_d || tx >= ty {
            tx /= 2;
        } else {
            ty /= 2;
        }
        if tx < 8 || (!one_d && ty < 8) {
            return None;
        }
    };
    // A sub-program is computed one level after the staged sub-programs it reads.
    let mut level = vec![0u32; n];
    for s in 0..n {
        if region[s].is_some() {
            level[s] = c.reads.iter().filter(|read| read.reader == Reader::Sub(s as u32)).map(|read| level[read.sub as usize] + 1).max().unwrap_or(0);
        }
    }
    let mut order: Vec<usize> = (0..n).filter(|&s| region[s].is_some()).collect();
    order.sort_by_key(|&s| (level[s], s));
    let last = order.last().map_or(0, |&s| level[s]);
    let mut levels = vec![0u32];
    for l in 0..=last {
        levels.push(levels[l as usize] + order.iter().filter(|&&s| level[s] == l).count() as u32);
    }
    let subs = order
        .iter()
        .filter_map(|&s| {
            let (lo, hi) = region[s]?;
            Some(StagedSub { sub: s as u32, dims: c.sub_dims[s]?, lo, hi })
        })
        .collect();
    Some(Staging { dims: c.dims, tile: [tx, ty], subs, levels })
}

/// Verifies that `staging` serves every read `key` makes of a staged sub-program (see
/// [`Read`]): the reader is the main program or a staged sub-program computed in an earlier
/// level, every read is at a cell that is a translation of the reader's, and the cells it
/// reaches from every cell of the reader's region lie in the region of the sub-program.
#[cfg(any(debug_assertions, test))]
pub(crate) fn check(staging: &Staging, key: &FusedKey) -> Result<(), String> {
    let level_of = |sub: u32| {
        let at = staging.subs.iter().position(|staged| staged.sub == sub)?;
        staging.levels.iter().skip(1).position(|&end| at < end as usize)
    };
    let found = scan(key).ok_or("not a store kernel that reads sub-programs through views")?;
    let mut staged = vec![false; key.subs.len()];
    for entry in &staging.subs {
        let Some(flag) = staged.get_mut(entry.sub as usize) else { return Err(format!("sub-program {} does not exist", entry.sub)) };
        *flag = true;
        if found.dims[entry.sub as usize] != Some(entry.dims) {
            return Err(format!("sub-program {} has dims {:?}, staged as {:?}", entry.sub, found.dims[entry.sub as usize], entry.dims));
        }
    }
    if found.main != staging.dims {
        return Err(format!("the kernel has dims {:?}, staged as {:?}", found.main, staging.dims));
    }
    let (reads, broken) = collect(key, &staged, &found);
    if let Some(s) = (0..staged.len()).find(|&s| staged[s] && broken[s]) {
        return Err(format!("staged sub-program {s} is read at a cell that is not a translation of its reader's"));
    }
    let main = ([0i32, 0], [staging.tile[0] as i32 - 1, staging.tile[1] as i32 - 1]);
    for read in reads {
        let here = format!("{:?} reads staged sub-program {} at {:?}", read.reader, read.sub, read.offset);
        let Some(staged) = staging.find(read.sub) else { return Err(format!("{here}: not staged")) };
        let ((lo, hi), level) = match read.reader {
            Reader::Main => (main, None),
            Reader::Sub(t) => match staging.find(t) {
                Some(reader) => ((reader.lo, reader.hi), level_of(t)),
                None => return Err(format!("{here} from a program that is not staged")),
            },
        };
        if read.reader != Reader::Main && level.zip(level_of(read.sub)).is_none_or(|(reader_level, level)| level >= reader_level) {
            return Err(format!("{here} in a level that is not after it"));
        }
        for axis in 0..2 {
            let (need_lo, need_hi) = (lo[axis] + read.offset[axis], hi[axis] + read.offset[axis]);
            if need_lo < staged.lo[axis] || need_hi > staged.hi[axis] {
                return Err(format!("{here} outside its region on axis {axis}: needs {need_lo}..={need_hi}, has {}..={}", staged.lo[axis], staged.hi[axis]));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtin::BinOp;
    use crate::jit::program::{Instr, Tail};
    use crate::jit::{ElemTy, Op, ViewMap};
    use crate::tuning::Tuning;

    fn instr(kind: OpKind, args: [Ref; 3]) -> Instr {
        Instr { op: Op::new(kind, ElemTy::F32), args }
    }

    fn key(instrs: Vec<Instr>, subs: Vec<Vec<Instr>>) -> FusedKey {
        let outputs = vec![instrs.len() as u32 - 1];
        FusedKey {
            shape: crate::jit::program::shape_of(&instrs, &subs),
            instrs: std::sync::Arc::new(instrs),
            input_tys: vec![ElemTy::F32],
            scalar_tys: vec![ElemTy::F32],
            outputs,
            subs: std::sync::Arc::new(subs),
            tail: Tail::Store,
            in_place: Vec::new(),
            spans: Vec::new(),
            tile: None,
            staging: None,
            guard: None,
        }
    }

    /// The view of a `src` array that reads `out` cells from `start`.
    fn view(src: [u32; 2], out: [u32; 2], start: [i32; 2], edge: Edge) -> ShiftKind {
        let mut map = ViewMap::identity([src[0], src[1], 1, 1]);
        map.out[0] = out[0];
        map.out[1] = out[1];
        map.axes[0].start = start[0];
        map.axes[1].start = start[1];
        ShiftKind::View { map, edge }
    }

    fn read(kind: ShiftKind, sub: u32) -> Instr {
        instr(OpKind::Shift(kind), [Ref::Sub(sub), Ref::Unused, Ref::Unused])
    }

    fn add(a: u32, b: u32) -> Instr {
        instr(OpKind::Binary(BinOp::Add), [Ref::Tmp(a), Ref::Tmp(b), Ref::Unused])
    }

    /// `e = x * s0`.
    fn expression() -> Vec<Instr> {
        vec![instr(OpKind::Binary(BinOp::Mul), [Ref::Input(0), Ref::Scalar(0), Ref::Unused])]
    }

    fn tuning(bytes: u32) -> Tuning {
        Tuning { stencil_tile_bytes: bytes, stencil_tile_2d: (32, 16), stencil_tile_1d: 512, ..Tuning::for_tests() }
    }

    #[test]
    fn a_three_slice_read_of_a_1d_array_stages_the_tile_and_a_halo() {
        let (n, out) = (100, 98);
        let main = vec![
            read(view([n, 1], [out, 1], [0, 0], Edge::None), 0),
            read(view([n, 1], [out, 1], [1, 0], Edge::None), 0),
            read(view([n, 1], [out, 1], [2, 0], Edge::None), 0),
            add(0, 1),
            add(3, 2),
        ];
        let key = key(main, vec![expression()]);
        let found = candidates(&key).expect("candidate");
        assert_eq!(found.dims, [out, 1]);
        let staging = plan(&found, &tuning(32 * 1024), |_| 4).expect("plan");
        assert_eq!(staging.tile, [512, 1]);
        assert_eq!(staging.subs, vec![StagedSub { sub: 0, dims: [n, 1], lo: [0, 0], hi: [513, 0] }]);
        assert_eq!(staging.levels, vec![0, 1]);
        assert_eq!(check(&staging, &key), Ok(()));
    }

    #[test]
    fn a_five_point_read_of_a_2d_array_covers_the_tile_plus_a_cell_on_each_side() {
        let (src, out) = ([34, 18], [32, 16]);
        let main = vec![
            read(view(src, out, [0, 1], Edge::None), 0),
            read(view(src, out, [2, 1], Edge::None), 0),
            read(view(src, out, [1, 0], Edge::None), 0),
            read(view(src, out, [1, 2], Edge::None), 0),
            read(view(src, out, [1, 1], Edge::None), 0),
            add(0, 1),
            add(5, 2),
            add(6, 3),
            add(7, 4),
        ];
        let key = key(main, vec![expression()]);
        let staging = plan(&candidates(&key).expect("candidate"), &tuning(32 * 1024), |_| 4).expect("plan");
        assert_eq!(staging.tile, [32, 16]);
        assert_eq!(staging.subs.len(), 1);
        assert_eq!((staging.subs[0].lo, staging.subs[0].hi), ([0, 0], [33, 17]));
        assert_eq!(check(&staging, &key), Ok(()));
    }

    #[test]
    fn a_staged_sub_program_read_by_another_one_is_computed_a_level_earlier() {
        // f = g(e)[0..] - g(e)[1..] as sub 1 over e (sub 0); r = f[0..] + f[1..].
        let f = vec![
            read(view([40, 1], [39, 1], [0, 0], Edge::None), 0),
            read(view([40, 1], [39, 1], [1, 0], Edge::None), 0),
            add(0, 1),
        ];
        let main = vec![read(view([39, 1], [38, 1], [0, 0], Edge::None), 1), read(view([39, 1], [38, 1], [1, 0], Edge::None), 1), add(0, 1)];
        let key = key(main, vec![expression(), f]);
        let staging = plan(&candidates(&key).expect("candidate"), &tuning(32 * 1024), |_| 4).expect("plan");
        assert_eq!(staging.subs.iter().map(|staged| staged.sub).collect::<Vec<_>>(), vec![0, 1]);
        assert_eq!(staging.levels, vec![0, 1, 2]);
        // `f` covers the tile and one cell, `e` that and one more.
        assert_eq!((staging.subs[1].lo, staging.subs[1].hi), ([0, 0], [staging.tile[0] as i32, 0]));
        assert_eq!((staging.subs[0].lo, staging.subs[0].hi), ([0, 0], [staging.tile[0] as i32 + 1, 0]));
        assert_eq!(check(&staging, &key), Ok(()));
    }

    #[test]
    fn a_sub_program_read_through_an_edge_is_not_a_candidate() {
        let main = vec![
            read(view([100, 1], [98, 1], [0, 0], Edge::Clamp), 0),
            read(view([100, 1], [98, 1], [1, 0], Edge::Clamp), 0),
            add(0, 1),
        ];
        assert!(candidates(&key(main, vec![expression()])).is_none());
    }

    #[test]
    fn a_sub_program_read_once_by_the_main_program_is_dropped() {
        let main = vec![read(view([100, 1], [98, 1], [1, 0], Edge::None), 0), read(view([100, 1], [98, 1], [1, 0], Edge::None), 0), add(0, 1)];
        assert!(candidates(&key(main, vec![expression()])).is_none());
    }

    #[test]
    fn the_tile_shrinks_to_fit_the_workgroup_memory() {
        let (src, out) = ([34, 18], [32, 16]);
        let main = vec![read(view(src, out, [0, 1], Edge::None), 0), read(view(src, out, [2, 1], Edge::None), 0), add(0, 1)];
        let key = key(main, vec![expression()]);
        let found = candidates(&key).expect("candidate");
        // 34 x 16 cells of 4 bytes do not fit 2 KiB; 18 x 16 (a 16 x 16 tile) do.
        let staging = plan(&found, &tuning(2 * 1024), |_| 4).expect("plan");
        assert_eq!(staging.tile, [16, 16]);
        assert_eq!(check(&staging, &key), Ok(()));
        assert!(plan(&found, &tuning(64), |_| 4).is_none());
    }

    #[test]
    fn what_a_sub_program_read_once_reads_is_read_from_the_cell_it_is_computed_at() {
        // f = e[0..] + e[1..] (sub 1, over e = sub 0); the main program reads f once, from 1.
        let f = vec![read(view([40, 1], [39, 1], [0, 0], Edge::None), 0), read(view([40, 1], [39, 1], [1, 0], Edge::None), 0), add(0, 1)];
        let main = vec![read(view([39, 1], [38, 1], [1, 0], Edge::None), 1)];
        let key = key(main, vec![expression(), f]);
        let found = candidates(&key).expect("candidate");
        // `f` is computed inside the main program, so the main program reads `e` at 1 and 2.
        assert_eq!(found.sub_dims, vec![Some([40, 1]), None]);
        assert_eq!(found.reads, vec![Read { reader: Reader::Main, sub: 0, offset: [1, 0] }, Read { reader: Reader::Main, sub: 0, offset: [2, 0] }]);
        let staging = plan(&found, &tuning(32 * 1024), |_| 4).expect("plan");
        assert_eq!(staging.subs.len(), 1);
        assert_eq!((staging.subs[0].lo, staging.subs[0].hi), ([1, 0], [staging.tile[0] as i32 + 1, 0]));
        assert_eq!(check(&staging, &key), Ok(()));
    }

    #[test]
    fn a_sub_program_computed_through_an_edge_cannot_serve_what_it_reads_from_workgroup_memory() {
        let f = vec![read(view([40, 1], [39, 1], [0, 0], Edge::None), 0), read(view([40, 1], [39, 1], [1, 0], Edge::None), 0), add(0, 1)];
        let main = vec![read(view([39, 1], [38, 1], [1, 0], Edge::Clamp), 1)];
        assert!(candidates(&key(main, vec![expression(), f])).is_none());
    }

    #[test]
    fn a_sub_program_that_only_views_another_one_is_never_staged() {
        // v = e[2..] (sub 1); the main program reads v at 0, 1 and 2: `e` is what is staged.
        let v = vec![read(view([42, 1], [40, 1], [2, 0], Edge::None), 0)];
        let main = vec![
            read(view([40, 1], [38, 1], [0, 0], Edge::None), 1),
            read(view([40, 1], [38, 1], [1, 0], Edge::None), 1),
            read(view([40, 1], [38, 1], [2, 0], Edge::None), 1),
            add(0, 1),
            add(3, 2),
        ];
        let key = key(main, vec![expression(), v]);
        let found = candidates(&key).expect("candidate");
        assert_eq!(found.sub_dims, vec![Some([42, 1]), None]);
        assert_eq!(found.reads.iter().map(|read| read.offset[0]).collect::<Vec<_>>(), vec![2, 3, 4]);
        let staging = plan(&found, &tuning(32 * 1024), |_| 4).expect("plan");
        assert_eq!(staging.subs.len(), 1);
        assert_eq!(check(&staging, &key), Ok(()));
    }
}
