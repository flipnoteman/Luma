//! The bit-packed layout of `Array<bool>`: element `i` is bit `i % 32` of word `i / 32` of
//! `ceil(n / 32)` words, and the bits past `n` are 0. Every way of reading and writing a boolean
//! array is checked against the host at the sizes around the word boundaries (and, on a context
//! with tiny buffers, around the segment boundaries).

use crate::common;

use common::values;
use luma::*;

const SIZES: [usize; 10] = [1, 31, 32, 33, 63, 64, 65, 1000, 1 << 20, (1 << 20) + 7];

/// A context whose buffers hold 1024 bytes: 256 elements per segment, 8 words of bits.
fn chunked() -> Context {
    Context::with_options(ContextOptions::from_env().chunk_bytes(1024)).expect("a device is available")
}

fn floats(n: usize, seed: u64) -> Vec<f32> {
    values(n, seed, -1.0, 1.0)
}

fn bits_of(x: &[f32]) -> Vec<bool> {
    x.iter().map(|&v| v > 0.0).collect()
}

/// The words of `bits`: 32 to a word, the first element in the lowest bit.
fn pack(bits: &[bool]) -> Vec<u32> {
    bits.chunks(32).map(|chunk| chunk.iter().enumerate().fold(0u32, |word, (i, &b)| word | u32::from(b) << i)).collect()
}

const WORDS: &str = r#"
@group(1) @binding(0) var<storage, read> src: array<u32>;
@group(1) @binding(1) var<storage, read_write> dst: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= luma.size.x) { return; }
    dst[gid.x] = src[gid.x];
}
"#;

const UNPACK: &str = r#"
@group(1) @binding(0) var<storage, read> src: array<u32>;
@group(1) @binding(1) var<storage, read_write> dst: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= luma.size.x) { return; }
    dst[i] = (src[i >> 5u] >> (i & 31u)) & 1u;
}
"#;

/// The words a boolean array holds on the GPU, as a custom kernel sees them.
fn words(m: &Array<bool>) -> Vec<u32> {
    let count = m.len().div_ceil(32);
    let mut out = Array::<u32>::zeros_on(m.context(), count).unwrap();
    let kernel = m.context().kernel(WORDS).build().unwrap();
    kernel.launch([count as u32], args![m, &mut out]).unwrap();
    out.to_vec().unwrap()
}

/// `m` holds exactly `want`: through the host, through the words a kernel reads (the unused bits
/// of the last word are 0), and through the counts that read the words.
fn check(m: &Array<bool>, want: &[bool], what: &str) {
    assert_eq!(m.len(), want.len(), "{what}: length");
    assert_eq!(m.to_vec().unwrap(), want, "{what}: to_vec");
    assert_eq!(words(m), pack(want), "{what}: words");
    let count = want.iter().filter(|&&b| b).count();
    assert_eq!(m.count().item().unwrap() as usize, count, "{what}: count");
    assert_eq!(m.any().item().unwrap(), count > 0, "{what}: any");
    assert_eq!(m.all().item().unwrap(), count == want.len(), "{what}: all");
}

/// `m` read back through [`Array::read_into`] too.
fn check_read_into(m: &Array<bool>, want: &[bool]) {
    let mut out = vec![!want[0]; want.len()];
    m.read_into(&mut out).unwrap();
    assert_eq!(out, want);
}

fn bools_on(ctx: &Context, bits: &[bool]) -> Array<bool> {
    Array::from_slice_on(ctx, bits, bits.len()).unwrap()
}

fn f32s_on(ctx: &Context, data: &[f32]) -> Array<f32> {
    Array::from_slice_on(ctx, data, data.len()).unwrap()
}

/// A boolean array that has been evaluated into a buffer of its own.
fn evaluated(m: Array<bool>) -> Array<bool> {
    m.to_vec().unwrap();
    m
}

#[test]
fn comparisons_write_whole_words() {
    let ctx = Context::new().unwrap();
    for n in SIZES {
        let x = floats(n, n as u64);
        let want = bits_of(&x);
        let m = f32s_on(&ctx, &x).gt(0.0);
        check(&m, &want, &format!("gt, n = {n}"));
        check_read_into(&m, &want);
        // Several boolean outputs of one kernel are packed one by one.
        let (y, z) = (f32s_on(&ctx, &x).lt(0.25), f32s_on(&ctx, &x).ne(0.0));
        eval_all(&[&y, &z]).unwrap();
        check(&y, &x.iter().map(|&v| v < 0.25).collect::<Vec<_>>(), &format!("lt, n = {n}"));
        check(&z, &x.iter().map(|&v| v != 0.0).collect::<Vec<_>>(), &format!("ne, n = {n}"));
    }
}

#[test]
fn nothing_outside_the_array_is_set() {
    // Stale bits in a recycled buffer must not leak into the unused bits of the last word.
    let ctx = Context::new().unwrap();
    for n in [1usize, 5, 31, 33, 65] {
        // A buffer full of ones goes back to the pool for the arrays below to take.
        let words = vec![u32::MAX; n.div_ceil(32)];
        Array::from_slice_on(&ctx, &words, words.len()).unwrap().to_vec().unwrap();
        let m = f32s_on(&ctx, &vec![1.0; n]).gt(0.0);
        check(&m, &vec![true; n], &format!("all true, n = {n}"));
        let full = Array::full_on(&ctx, n, true).unwrap();
        check(&full, &vec![true; n], &format!("full, n = {n}"));
        check(&!&full, &vec![false; n], &format!("not full, n = {n}"));
        let zeros = Array::<bool>::zeros_on(&ctx, n).unwrap();
        check(&!&zeros, &vec![true; n], &format!("not zeros, n = {n}"));
    }
}

#[test]
fn host_upload_and_readback_round_trip() {
    let ctx = Context::new().unwrap();
    for n in SIZES {
        let want = bits_of(&floats(n, 7 * n as u64));
        let a = bools_on(&ctx, &want);
        check(&a, &want, &format!("from_slice, n = {n}"));
        check_read_into(&a, &want);
        let mut b = Array::<bool>::zeros_on(&ctx, n).unwrap();
        assert_eq!(b.to_vec().unwrap(), vec![false; n]);
        b.write(&want).unwrap();
        check(&b, &want, &format!("write, n = {n}"));
        let c = a.copy();
        check(&c, &want, &format!("copy, n = {n}"));
        for value in [true, false] {
            check(&Array::full_on(&ctx, n, value).unwrap(), &vec![value; n], &format!("full({value}), n = {n}"));
        }
    }
}

#[test]
fn counts_any_and_all_read_the_words() {
    let ctx = Context::new().unwrap();
    for n in SIZES {
        let patterns: Vec<(&str, Vec<bool>)> = vec![
            ("all true", vec![true; n]),
            ("all false", vec![false; n]),
            ("only the first", (0..n).map(|i| i == 0).collect()),
            ("only the last", (0..n).map(|i| i + 1 == n).collect()),
            ("all but the last", (0..n).map(|i| i + 1 != n).collect()),
            ("alternating", (0..n).map(|i| i % 2 == 1).collect()),
            ("random", bits_of(&floats(n, 3 * n as u64))),
        ];
        for (name, bits) in patterns {
            // An uploaded array and a computed one take the same path (reading words).
            let uploaded = bools_on(&ctx, &bits);
            check(&uploaded, &bits, &format!("{name}, n = {n}"));
            let x: Vec<f32> = bits.iter().map(|&b| if b { 1.0 } else { -1.0 }).collect();
            let computed = f32s_on(&ctx, &x).gt(0.0);
            // A pending comparison is reduced inside the kernel that computes it.
            let count = bits.iter().filter(|&&b| b).count();
            assert_eq!(computed.count().item().unwrap() as usize, count, "{name}, n = {n}: pending count");
            assert_eq!(computed.sum().item().unwrap() as usize, count);
            assert_eq!(computed.any().item().unwrap(), count > 0);
            assert_eq!(computed.all().item().unwrap(), count == n);
            assert_eq!(computed.min().item().unwrap(), count == n);
            assert_eq!(computed.max().item().unwrap(), count > 0);
        }
    }
}

#[test]
fn consumers_of_evaluated_bools_read_the_bits() {
    let ctx = Context::new().unwrap();
    for n in SIZES {
        let (x, y) = (floats(n, 11), floats(n, 12));
        let (hm, hn) = (bits_of(&x), bits_of(&y));
        let (m, k) = (evaluated(f32s_on(&ctx, &x).gt(0.0)), bools_on(&ctx, &hn));
        let xs = f32s_on(&ctx, &x);
        let selected = select(&m, &xs, 7.0).to_vec().unwrap();
        assert_eq!(selected, (0..n).map(|i| if hm[i] { x[i] } else { 7.0 }).collect::<Vec<_>>(), "select, n = {n}");
        let sum = (m.to_f32() * 3.0 + &xs).to_vec().unwrap();
        assert_eq!(sum, (0..n).map(|i| f32::from(u8::from(hm[i])) * 3.0 + x[i]).collect::<Vec<_>>(), "to_f32, n = {n}");
        check(&(&m & &k), &(0..n).map(|i| hm[i] & hn[i]).collect::<Vec<_>>(), &format!("and, n = {n}"));
        check(&(&m | &k), &(0..n).map(|i| hm[i] | hn[i]).collect::<Vec<_>>(), &format!("or, n = {n}"));
        check(&(&m ^ &k), &(0..n).map(|i| hm[i] ^ hn[i]).collect::<Vec<_>>(), &format!("xor, n = {n}"));
        check(&!&m, &hm.iter().map(|b| !b).collect::<Vec<_>>(), &format!("not, n = {n}"));
        let counted = (0..n).filter(|&i| hm[i] && !hn[i]).count();
        assert_eq!((&m & &!&k).count().item().unwrap() as usize, counted, "n = {n}");
        let masked = (&m * &xs).to_vec().unwrap();
        assert_eq!(masked, (0..n).map(|i| if hm[i] { x[i] } else { 0.0 }).collect::<Vec<_>>());
        let positive: f32 = x.iter().zip(&hm).filter(|&(_, &b)| b).map(|(&v, _)| v).sum();
        assert!((xs.sum_where(&m).item().unwrap() - positive).abs() <= 1e-3 * positive.abs().max(1.0) * (n as f32).sqrt(), "sum_where, n = {n}");
        let words_as_condition = m.to_u32();
        assert_eq!(select(&words_as_condition, &xs, 7.0).to_vec().unwrap(), selected, "u32 condition, n = {n}");
    }
}

/// The host models of the views.
fn flipped(b: &[bool]) -> Vec<bool> {
    b.iter().rev().copied().collect()
}

fn shifted_fill(b: &[bool], k: isize, fill: bool) -> Vec<bool> {
    (0..b.len() as isize).map(|i| if (0..b.len() as isize).contains(&(i - k)) { b[(i - k) as usize] } else { fill }).collect()
}

fn shifted_clamp(b: &[bool], k: isize) -> Vec<bool> {
    (0..b.len() as isize).map(|i| b[(i - k).clamp(0, b.len() as isize - 1) as usize]).collect()
}

/// `w * h` with `w <= h` closest to a square.
fn factors(n: usize) -> (usize, usize) {
    let w = (1..=(n as f64).sqrt() as usize).rev().find(|w| n.is_multiple_of(*w)).unwrap();
    (w, n / w)
}

#[test]
fn views_of_evaluated_bools() {
    let ctx = Context::new().unwrap();
    for n in SIZES {
        let want = bits_of(&floats(n, 21 + n as u64));
        let m = bools_on(&ctx, &want);
        let tag = |what: &str| format!("{what}, n = {n}");
        // Slices at odd offsets: the first element of the view is in the middle of a word.
        for (lo, hi) in [(0, n), (1, n), ((n / 2) | 1, n), (n.min(3) / 3, n - n / 4), (n / 3, n / 3 + 1)] {
            if lo < hi {
                check(&m.slice(0, lo..hi), &want[lo..hi], &tag(&format!("slice {lo}..{hi}")));
            }
        }
        check(&m.flip(0), &flipped(&want), &tag("flip"));
        check(&m.slice(0, n.min(5) / 2..).flip(0), &flipped(&want[n.min(5) / 2..]), &tag("flipped slice"));
        for k in [1isize, 3, -2, 33, -64, n as isize / 2] {
            check(&m.shift_fill(k, true), &shifted_fill(&want, k, true), &tag(&format!("shift_fill({k}, true)")));
            check(&m.shift_fill(k, false), &shifted_fill(&want, k, false), &tag(&format!("shift_fill({k}, false)")));
            check(&m.shift_clamp(k), &shifted_clamp(&want, k), &tag(&format!("shift_clamp({k})")));
            let circular = (0..n as isize).map(|i| want[(i - k).rem_euclid(n as isize) as usize]).collect::<Vec<_>>();
            check(&m.shift(k), &circular, &tag(&format!("shift({k})")));
        }
        let padded = m.pad([(3, 5)], PadMode::Zero);
        let mut expect = vec![false; 3];
        expect.extend(&want);
        expect.extend([false; 5]);
        check(&padded, &expect, &tag("pad zero"));
        let edge = m.pad([(2, 7)], PadMode::Edge);
        let mut expect = vec![want[0]; 2];
        expect.extend(&want);
        expect.extend(vec![want[n - 1]; 7]);
        check(&edge, &expect, &tag("pad edge"));
        let wrapped = m.pad([(4, 4)], PadMode::Wrap);
        let expect: Vec<bool> = (-4..n as isize + 4).map(|i| want[i.rem_euclid(n as isize) as usize]).collect();
        check(&wrapped, &expect, &tag("pad wrap"));
        // A matrix: transposes and slices along the second axis.
        let (w, h) = factors(n);
        let grid = m.reshape([w, h]);
        let mut transposed = vec![false; n];
        for y in 0..h {
            for x in 0..w {
                transposed[y + h * x] = want[x + w * y];
            }
        }
        let t = grid.transpose();
        assert_eq!(t.shape().dims()[..2], [h, w]);
        check(&t.reshape(n), &transposed, &tag("transpose"));
        if h > 2 {
            check(&grid.slice(1, 1..h - 1).reshape(w * (h - 2)), &want[w..w * (h - 1)], &tag("row slice"));
            check(&grid.flip(1).reshape(n), &(0..h).rev().flat_map(|y| want[w * y..w * (y + 1)].to_vec()).collect::<Vec<_>>(), &tag("row flip"));
        }
        // Broadcast: one row repeated.
        let row = m.slice(0, 0..w.min(n));
        let rows = row.reshape([row.len(), 1]).broadcast_to([row.len(), 3]);
        let expect: Vec<bool> = (0..3).flat_map(|_| want[..row.len()].to_vec()).collect();
        check(&rows.reshape(3 * row.len()), &expect, &tag("broadcast"));
    }
}

#[test]
fn gathers_read_single_bits() {
    let ctx = Context::new().unwrap();
    for n in SIZES {
        let want = bits_of(&floats(n, 31 + n as u64));
        let m = bools_on(&ctx, &want);
        let count = n.min(1000);
        let idx: Vec<u32> = (0..count).map(|i| ((i * 7919 + 13) % (n + 4)) as u32).collect();
        let indices = Array::from_slice_on(&ctx, &idx, count).unwrap();
        // Indices past the end read the last element.
        let expect: Vec<bool> = idx.iter().map(|&i| want[(i as usize).min(n - 1)]).collect();
        check(&m.take(&indices), &expect, &format!("take, n = {n}"));
        // Gathers of an expression computed at the gathered indices.
        let x = floats(n, 31 + n as u64);
        let computed = f32s_on(&ctx, &x).gt(0.0);
        check(&computed.take(&indices), &expect, &format!("take of a comparison, n = {n}"));
    }
}

#[test]
fn assigning_into_slices_writes_single_bits() {
    let ctx = Context::new().unwrap();
    for n in [40, 64, 65, 1000, (1 << 16) + 7] {
        let want = bits_of(&floats(n, 41));
        let mut model = want.clone();
        let mut a = bools_on(&ctx, &want);
        // Odd ranges: the written bits share words with bits that stay.
        let ranges = [(1, 6), (6, 38), (38, 39), ((n / 2) | 1, n - 1), (0, 1), (n - 1, n)];
        for (step, &(lo, hi)) in ranges.iter().enumerate() {
            let value = step % 2 == 0;
            a.slice_mut(0, lo..hi).assign(Array::full_on(&ctx, hi - lo, value).unwrap()).unwrap();
            model[lo..hi].fill(value);
            check(&a, &model, &format!("assign constant {value}, {lo}..{hi}, n = {n}"));
        }
        // An expression of the right values, adjacent writes: `true` and `false` mixed in words.
        let y = floats(hi_len(n), 42);
        let rhs = f32s_on(&ctx, &y).lt(0.0);
        let (lo, hi) = (5, 5 + hi_len(n));
        a.slice_mut(0, lo..hi).assign(&rhs).unwrap();
        for (k, &v) in y.iter().enumerate() {
            model[lo + k] = v < 0.0;
        }
        check(&a, &model, &format!("assign comparison, n = {n}"));
        // A right-hand side that reads the cells it writes through a shift.
        let source = a.slice(0, 0..n - 3);
        a.slice_mut(0, 3..n).assign(&source).unwrap();
        let old = model.clone();
        model[3..n].copy_from_slice(&old[..n - 3]);
        check(&a, &model, &format!("overlapping assign, n = {n}"));
        // Whole-array writes, the bits flipped in place.
        let flipped_in_place = !&a;
        a.slice_mut(0, 0..n).assign(&flipped_in_place).unwrap();
        check(&a, &model.iter().map(|b| !b).collect::<Vec<_>>(), &format!("whole assign, n = {n}"));
    }
}

fn hi_len(n: usize) -> usize {
    (n / 3).clamp(2, 700)
}

#[test]
fn assigning_into_records_writes_every_field_in_one_kernel() {
    let ctx = Context::new().unwrap();
    let n = 70;
    let (a, b) = (bits_of(&floats(n, 51)), bits_of(&floats(n, 52)));
    let mut rec = Flags { a: bools_on(&ctx, &a), b: bools_on(&ctx, &b) };
    let rhs = Flags { a: Array::full_on(&ctx, 20, true).unwrap(), b: Array::full_on(&ctx, 20, false).unwrap() };
    rec.slice_mut(0, 11..31).assign(rhs).unwrap();
    let (mut ma, mut mb) = (a.clone(), b.clone());
    ma[11..31].fill(true);
    mb[11..31].fill(false);
    check(&rec.a, &ma, "field a");
    check(&rec.b, &mb, "field b");
    let sliced = rec.slice(0, 5..65);
    check(&sliced.a, &ma[5..65], "sliced field a");
    check(&sliced.b, &mb[5..65], "sliced field b");
    let selected = rec.map(|field| !field);
    check(&selected.a, &ma.iter().map(|b| !b).collect::<Vec<_>>(), "mapped field a");
}

/// A record with boolean fields (`fields!` is for arithmetic: its operators do not exist for them).
struct Flags {
    a: Array<bool>,
    b: Array<bool>,
}

impl Fields for Flags {
    type Elem = bool;
    const COUNT: usize = 2;

    fn each<'a>(&'a self, f: &mut dyn FnMut(&'a Array<bool>)) {
        f(&self.a);
        f(&self.b);
    }

    fn each_mut<'a>(&'a mut self, f: &mut dyn FnMut(&'a mut Array<bool>)) {
        f(&mut self.a);
        f(&mut self.b);
    }

    fn build(next: &mut dyn FnMut() -> Array<bool>) -> Self {
        let a = next();
        Flags { a, b: next() }
    }
}

#[test]
fn puts_scatter_single_bits() {
    let ctx = Context::new().unwrap();
    for n in [33, 64, 65, 1000, (1 << 16) + 7] {
        let want = bits_of(&floats(n, 61));
        let mut model = want.clone();
        let mut a = bools_on(&ctx, &want);
        // Adjacent indices, duplicates carrying one value, and indices past the end (skipped).
        let mut idx: Vec<u32> = (3..40).collect();
        idx.extend([40, 40, 41, 41, 41, n as u32 - 1, n as u32 - 1, n as u32, n as u32 + 9]);
        idx.extend((0..300).map(|i| (i * 37 % n) as u32));
        let value = |i: u32| i.is_multiple_of(3) || i % 7 == 1;
        let vals: Vec<bool> = idx.iter().map(|&i| value(i)).collect();
        let indices = Array::from_slice_on(&ctx, &idx, idx.len()).unwrap();
        a.put(&indices, bools_on(&ctx, &vals)).unwrap();
        for &i in &idx {
            if (i as usize) < n {
                model[i as usize] = value(i);
            }
        }
        check(&a, &model, &format!("put, n = {n}"));
        // A single value, and values computed in the kernel.
        a.put(&indices, true).unwrap();
        for &i in &idx {
            if (i as usize) < n {
                model[i as usize] = true;
            }
        }
        check(&a, &model, &format!("put true, n = {n}"));
        // Values computed in the kernel, at distinct indices (so the winner of duplicates does not matter).
        let y = floats(64, 62);
        let distinct: Vec<u32> = (0..64).map(|i| (i * 5 + 1) as u32).collect();
        let first_values: Vec<bool> = y.iter().map(|&v| v > 0.0).collect();
        let first = f32s_on(&ctx, &y).gt(0.0);
        a.put(&Array::from_slice_on(&ctx, &distinct, 64).unwrap(), &first).unwrap();
        for (k, &i) in distinct.iter().enumerate() {
            if (i as usize) < n {
                model[i as usize] = first_values[k];
            }
        }
        check(&a, &model, &format!("put comparison, n = {n}"));
    }
}

#[test]
fn compound_assignments_write_shared_buffers_in_place() {
    let ctx = Context::new().unwrap();
    for n in [1, 33, 65, 1000, (1 << 16) + 7] {
        let (hm, hn) = (bits_of(&floats(n, 71)), bits_of(&floats(n, 72)));
        let other = bools_on(&ctx, &hn);
        for op in 0..3 {
            let mut m = bools_on(&ctx, &hm);
            // A second handle on the buffer: the update cannot take over the array, it runs in place.
            let alias = m.clone();
            match op {
                0 => m &= &other,
                1 => m |= &other,
                _ => m ^= &other,
            }
            let want: Vec<bool> = (0..n)
                .map(|i| match op {
                    0 => hm[i] & hn[i],
                    1 => hm[i] | hn[i],
                    _ => hm[i] ^ hn[i],
                })
                .collect();
            check(&m, &want, &format!("compound {op}, n = {n}"));
            check(&alias, &want, &format!("alias of compound {op}, n = {n}"));
        }
    }
}

#[test]
fn stacks_and_concatenations_join_bits() {
    let ctx = Context::new().unwrap();
    for n in [1, 31, 33, 65, 1000, (1 << 16) + 7] {
        let parts: Vec<Vec<bool>> = (0..3).map(|k| bits_of(&floats(n, 81 + k))).collect();
        // Evaluated, and pending (the comparison is computed inside the join).
        let evaluated_parts: Vec<Array<bool>> = parts.iter().map(|p| bools_on(&ctx, p)).collect();
        let pending_parts: Vec<Array<bool>> = (0..3).map(|k| f32s_on(&ctx, &floats(n, 81 + k)).gt(0.0)).collect();
        for (name, arrays) in [("evaluated", &evaluated_parts), ("pending", &pending_parts)] {
            let stacked = stack(0, [&arrays[0], &arrays[1], &arrays[2]]);
            let mut expect = Vec::new();
            for ((&a, &b), &c) in parts[0].iter().zip(&parts[1]).zip(&parts[2]) {
                expect.extend([a, b, c]);
            }
            check(&stacked, &expect, &format!("{name} stack axis 0, n = {n}"));
            let stacked = stack(1, [&arrays[0], &arrays[1], &arrays[2]]);
            check(&stacked, &parts.concat(), &format!("{name} stack axis 1, n = {n}"));
            if n > 1 {
                let joined = concat(0, [&arrays[0], &arrays[1].slice(0, 1..), &arrays[2]]);
                let mut expect = parts[0].clone();
                expect.extend(&parts[1][1..]);
                expect.extend(&parts[2]);
                check(&joined, &expect, &format!("{name} concat, n = {n}"));
            }
        }
    }
}

#[test]
fn custom_kernels_see_the_words() {
    let ctx = Context::new().unwrap();
    for n in SIZES {
        let want = bits_of(&floats(n, 91));
        let m = bools_on(&ctx, &want);
        // The documented way to read an element.
        let unpack = ctx.kernel(UNPACK).build().unwrap();
        let mut out = Array::<u32>::zeros_on(&ctx, n).unwrap();
        unpack.launch([n as u32], args![&m, &mut out]).unwrap();
        assert_eq!(out.to_vec().unwrap(), want.iter().map(|&b| u32::from(b)).collect::<Vec<_>>(), "n = {n}");
        // A kernel writing the words of a boolean array.
        let mut dst = Array::<bool>::zeros_on(&ctx, n).unwrap();
        let copy = ctx.kernel(WORDS).build().unwrap();
        copy.launch([n.div_ceil(32) as u32], args![&m, &mut dst]).unwrap();
        check(&dst, &want, &format!("copied words, n = {n}"));
    }
}

#[test]
fn loops_with_a_boolean_state() {
    let ctx = Context::new().unwrap();
    for n in [33, 1000] {
        let (hs, hm) = (bits_of(&floats(n, 101)), bits_of(&floats(n, 102)));
        let (s, m) = (bools_on(&ctx, &hs), evaluated(bools_on(&ctx, &hm)));
        for steps in [0, 1, 2, 5, 40] {
            let [out] = iterate_all([&s], steps, |[s]| [s ^ &m]);
            let want: Vec<bool> = (0..n).map(|i| hs[i] ^ (steps % 2 == 1 && hm[i])).collect();
            check(&out, &want, &format!("xor loop, {steps} steps, n = {n}"));
        }
        // Two states and an evaluated bool read by the step.
        let t = bools_on(&ctx, &hm.iter().rev().copied().collect::<Vec<_>>());
        for steps in [1, 3, 10] {
            let [x, y] = iterate_all([&s, &t], steps, |[x, y]| [x ^ y, x & &m]);
            let (mut hx, mut hy) = (hs.clone(), t.to_vec().unwrap());
            for _ in 0..steps {
                let next = ((0..n).map(|i| hx[i] ^ hy[i]).collect::<Vec<_>>(), (0..n).map(|i| hx[i] & hm[i]).collect::<Vec<_>>());
                (hx, hy) = next;
            }
            check(&x, &hx, &format!("two states, {steps} steps, n = {n}"));
            check(&y, &hy, &format!("two states, {steps} steps, n = {n}"));
        }
        // The same loop on the host, one step at a time.
        let mut by_hand = s.clone();
        for _ in 0..7 {
            by_hand = &by_hand ^ &m;
        }
        let want: Vec<bool> = (0..n).map(|i| hs[i] ^ hm[i]).collect();
        check(&by_hand, &want, &format!("host loop, n = {n}"));
    }
}

#[test]
fn chunked_arrays_split_on_word_boundaries() {
    let ctx = chunked();
    for n in [1, 255, 256, 257, 511, 513, 1000, 5000 + 7] {
        let x = floats(n, 111);
        let want = bits_of(&x);
        let xs = f32s_on(&ctx, &x);
        let m = xs.gt(0.0);
        assert_eq!(m.to_vec().unwrap(), want, "comparison, n = {n}");
        let uploaded = bools_on(&ctx, &want);
        assert_eq!(uploaded.to_vec().unwrap(), want, "upload, n = {n}");
        check_read_into(&uploaded, &want);
        let count = want.iter().filter(|&&b| b).count();
        for array in [&m, &uploaded] {
            assert_eq!(array.count().item().unwrap() as usize, count, "count, n = {n}");
            assert_eq!(array.any().item().unwrap(), count > 0);
            assert_eq!(array.all().item().unwrap(), count == n);
        }
        // A tail of true bits is not counted past the end of the array.
        let ones = Array::full_on(&ctx, n, true).unwrap();
        assert_eq!(ones.count().item().unwrap() as usize, n);
        assert!(ones.all().item().unwrap());
        assert_eq!(Array::<bool>::zeros_on(&ctx, n).unwrap().count().item().unwrap(), 0);
        let mut written = Array::<bool>::zeros_on(&ctx, n).unwrap();
        written.write(&want).unwrap();
        assert_eq!(written.to_vec().unwrap(), want, "write, n = {n}");
        // Elementwise consumers run segment by segment.
        let selected = select(&uploaded, &xs, 5.0).to_vec().unwrap();
        assert_eq!(selected, (0..n).map(|i| if want[i] { x[i] } else { 5.0 }).collect::<Vec<_>>(), "select, n = {n}");
        let other = bits_of(&floats(n, 112));
        let together = &uploaded ^ &bools_on(&ctx, &other);
        assert_eq!(together.to_vec().unwrap(), (0..n).map(|i| want[i] ^ other[i]).collect::<Vec<_>>(), "xor, n = {n}");
        assert_eq!((!&uploaded).to_vec().unwrap(), want.iter().map(|b| !b).collect::<Vec<_>>(), "not, n = {n}");
        // A circular shift reads neighbouring segments.
        let shifted = uploaded.shift(3);
        assert_eq!(shifted.to_vec().unwrap(), (0..n as isize).map(|i| want[(i - 3).rem_euclid(n as isize) as usize]).collect::<Vec<_>>(), "shift, n = {n}");
        if n > 1 {
            let sliced = uploaded.slice(0, 1..n);
            assert_eq!(sliced.to_vec().unwrap(), want[1..].to_vec(), "slice, n = {n}");
        }
    }
}

#[test]
fn spilled_boolean_arrays_come_back() {
    let ctx = Context::with_options(ContextOptions::from_env().chunk_bytes(1 << 16).memory_budget(1 << 20)).unwrap();
    let n = (1 << 20) + 7;
    let arrays: Vec<(Vec<bool>, Array<bool>)> = (0..24)
        .map(|k| {
            let bits = bits_of(&floats(n, 200 + k));
            let array = bools_on(&ctx, &bits);
            (bits, array)
        })
        .collect();
    // Arrays of 128 KiB each: 24 of them do not fit the budget with the data below.
    let big: Vec<Array<f32>> = (0..6).map(|k| f32s_on(&ctx, &floats(1 << 18, 300 + k))).collect();
    for array in &big {
        array.sum().item().unwrap();
    }
    for (bits, array) in &arrays {
        assert_eq!(&array.to_vec().unwrap(), bits);
        assert_eq!(array.count().item().unwrap() as usize, bits.iter().filter(|&&b| b).count());
    }
}

fn region_of<'a>(base: &'a mut Array<bool>, xs: &std::ops::Range<usize>, ys: &std::ops::Range<usize>, flip_rows: bool) -> ViewMut<'a, Array<bool>> {
    let view = base.slice_mut(0, xs.clone()).slice(1, ys.clone());
    if flip_rows { view.flip(1) } else { view }
}

/// A region of a matrix of bits: rows `ys` of the columns `xs`, in the order the region has.
fn region_assign(ctx: &Context, w: usize, h: usize, xs: std::ops::Range<usize>, ys: std::ops::Range<usize>, flip_rows: bool, tag: &str) {
    let want = bits_of(&floats(w * h, (w * 31 + h) as u64));
    let (rw, rh) = (xs.len(), ys.len());
    let new = bits_of(&floats(rw * rh, 5 + (w + h) as u64));
    let mut base = Array::from_slice_on(ctx, &want, [w, h]).unwrap();
    let rhs = Array::from_slice_on(ctx, &new, [rw, rh]).unwrap();
    // The same, from an expression that is computed inside the write.
    let from_x = floats(rw * rh, 5 + (w + h) as u64);
    let pending = Array::from_slice_on(ctx, &from_x, [rw, rh]).unwrap().gt(0.0);
    let mut model = want.clone();
    for y in 0..rh {
        let row = if flip_rows { ys.start + rh - 1 - y } else { ys.start + y };
        for x in 0..rw {
            model[xs.start + x + w * row] = new[x + rw * y];
        }
    }
    region_of(&mut base, &xs, &ys, flip_rows).assign(&rhs).unwrap();
    check(&base.reshape(w * h), &model, &format!("{tag}: assign of an array"));
    let mut again = Array::from_slice_on(ctx, &want, [w, h]).unwrap();
    region_of(&mut again, &xs, &ys, flip_rows).assign(&pending).unwrap();
    check(&again.reshape(w * h), &model, &format!("{tag}: assign of a comparison"));
    // Constants, so that every word of the run is all ones or all zeros.
    for value in [true, false] {
        let mut constant = Array::from_slice_on(ctx, &want, [w, h]).unwrap();
        region_of(&mut constant, &xs, &ys, flip_rows).assign(Array::full_on(ctx, [rw, rh], value).unwrap()).unwrap();
        let mut expect = want.clone();
        for y in ys.clone() {
            expect[xs.start + w * y..xs.end + w * y].fill(value);
        }
        check(&constant.reshape(w * h), &expect, &format!("{tag}: assign of {value}"));
    }
}

#[test]
fn region_writes_of_runs_keep_the_bits_around_them() {
    let ctx = Context::new().unwrap();
    // Long rows at every alignment, rows that are not a multiple of 32 apart, whole rows (a run
    // of several rows), rows shorter than a word (written bit by bit), flipped rows.
    for (w, h, xs, ys) in [
        (100, 7, 3..99, 0..7),
        (100, 7, 33..97, 1..6),
        (96, 9, 0..96, 2..8),
        (96, 9, 1..96, 0..9),
        (97, 9, 0..97, 3..7),
        (64, 8, 0..64, 0..8),
        (200, 5, 31..95, 0..5),
        (130, 70, 1..129, 1..69),
        (40, 40, 3..37, 2..30),
        (1000, 3, 5..995, 1..2),
        (4093, 4, 3..4004, 0..4),
    ] {
        region_assign(&ctx, w, h, xs.clone(), ys.clone(), false, &format!("{w}x{h} columns {xs:?} rows {ys:?}"));
        region_assign(&ctx, w, h, xs.clone(), ys.clone(), true, &format!("{w}x{h} columns {xs:?} rows {ys:?} flipped rows"));
    }
}

#[test]
fn region_writes_of_runs_in_three_dimensions_and_transposes() {
    let ctx = Context::new().unwrap();
    let (a, b, c) = (70usize, 5usize, 4usize);
    let want = bits_of(&floats(a * b * c, 77));
    let rhs_bits = bits_of(&floats(a * 3 * 2, 78));
    let mut base = Array::from_slice_on(&ctx, &want, [a, b, c]).unwrap();
    base.slice_mut(1, 1..4).slice(2, 1..3).assign(Array::from_slice_on(&ctx, &rhs_bits, [a, 3, 2]).unwrap()).unwrap();
    let mut model = want.clone();
    for z in 0..2 {
        for y in 0..3 {
            for x in 0..a {
                model[x + a * (1 + y) + a * b * (1 + z)] = rhs_bits[x + a * y + a * 3 * z];
            }
        }
    }
    check(&base.reshape(a * b * c), &model, "3-D region of whole rows");
    // A transposed region writes columns: no runs.
    let mut square = Array::from_slice_on(&ctx, &want[..64 * 64.min(want.len() / 64)], [64, want.len().min(64 * 64) / 64]).unwrap();
    let (w, h) = (64usize, want.len().min(64 * 64) / 64);
    let new = bits_of(&floats(w * h, 79));
    square.transpose_mut().assign(Array::from_slice_on(&ctx, &new, [h, w]).unwrap()).unwrap();
    let mut model = want[..w * h].to_vec();
    for y in 0..h {
        for x in 0..w {
            model[x + w * y] = new[y + h * x];
        }
    }
    check(&square.reshape(w * h), &model, "transposed region");
}

#[test]
fn updates_that_take_over_their_buffer() {
    let ctx = Context::new().unwrap();
    for n in [1, 33, 1000, (1 << 16) + 7] {
        let (hm, hn) = (bits_of(&floats(n, 301)), bits_of(&floats(n, 302)));
        let k = bools_on(&ctx, &hn);
        // The only handle on the buffer goes in: the result is written over it, bit by word.
        let m = bools_on(&ctx, &hm);
        check(&!m, &hm.iter().map(|b| !b).collect::<Vec<_>>(), &format!("not of the only owner, n = {n}"));
        let m = evaluated(f32s_on(&ctx, &floats(n, 301)).gt(0.0));
        check(&(m ^ &k), &(0..n).map(|i| hm[i] ^ hn[i]).collect::<Vec<_>>(), &format!("xor of the only owner, n = {n}"));
        let mut m = bools_on(&ctx, &hm);
        m &= &k;
        check(&m, &(0..n).map(|i| hm[i] & hn[i]).collect::<Vec<_>>(), &format!("and-assign, n = {n}"));
    }
}

#[test]
fn kernels_with_boolean_and_number_outputs() {
    let ctx = Context::new().unwrap();
    for n in [1, 33, 1000, (1 << 16) + 7] {
        let x = floats(n, 311);
        let xs = f32s_on(&ctx, &x);
        let y = &xs * 2.0;
        let m = y.gt(0.0);
        // The comparison and the product it reads are computed together, and the shift reads both.
        let shifted = m.shift(1);
        let ys = y.shift(-1);
        let want: Vec<bool> = x.iter().map(|&v| v * 2.0 > 0.0).collect();
        check(&shifted, &(0..n as isize).map(|i| want[(i - 1).rem_euclid(n as isize) as usize]).collect::<Vec<_>>(), &format!("shifted comparison, n = {n}"));
        assert_eq!(ys.to_vec().unwrap(), (0..n as isize).map(|i| x[(i + 1).rem_euclid(n as isize) as usize] * 2.0).collect::<Vec<_>>());
        check(&m, &want, &format!("comparison, n = {n}"));
        assert_eq!(y.to_vec().unwrap(), x.iter().map(|v| v * 2.0).collect::<Vec<_>>());
    }
}

/// A stencil with a packed boolean fixed input binds the input's real (packed) byte length.
#[test]
fn stencil_with_packed_mask_input() {
    let ctx = Context::new().unwrap();
    let n = 64usize;
    let init: Vec<f32> = (0..n * n).map(|i| ((i * 37) % 101) as f32).collect();
    let inner: Vec<bool> = (0..n * n).map(|i| (1..n - 1).contains(&(i % n)) && (1..n - 1).contains(&(i / n))).collect();
    let u = Array::from_slice_on(&ctx, &init, [n, n]).unwrap();
    let m = Array::from_slice_on(&ctx, &inner, [n, n]).unwrap();
    m.to_vec().unwrap();
    let got = u
        .iterate(10, |u| select(&m, (u.shift([1, 0]) + u.shift([-1, 0]) + u.shift([0, 1]) + u.shift([0, -1])) * 0.25, u))
        .to_vec()
        .unwrap();
    let mut want = init;
    for _ in 0..10 {
        let prev = want.clone();
        for y in 1..n - 1 {
            for x in 1..n - 1 {
                want[y * n + x] = (prev[y * n + x + 1] + prev[y * n + x - 1] + prev[(y + 1) * n + x] + prev[(y - 1) * n + x]) * 0.25;
            }
        }
    }
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!((g - w).abs() <= 1e-3 * w.abs().max(1.0), "element {i}: {g} vs {w}");
    }
}

/// `x *= &mask` is `select(mask, x, 0)`: a masked-out NaN becomes +0.0, as in `&x * &mask`.
#[test]
fn mul_assign_mask_selects() {
    let ctx = Context::new().unwrap();
    let mut x = Array::from_slice_on(&ctx, &[f32::NAN, -1.0, 2.0], 3).unwrap();
    let m = Array::from_slice_on(&ctx, &[false, false, true], 3).unwrap();
    let product = (&x * &m).to_vec().unwrap();
    x *= &m;
    let got = x.to_vec().unwrap();
    let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&got), bits(&[0.0, 0.0, 2.0]));
    assert_eq!(bits(&got), bits(&product));
}
