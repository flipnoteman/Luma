//! Bitwise operations on `u32` arrays against the host, and `u32` states in records and loops.

use luma::*;

fn u32s(data: &[u32]) -> Array<u32> {
    Array::from_slice(data, data.len()).unwrap()
}

/// Edge values and a pseudo-random spread.
fn words(n: usize, seed: u32) -> Vec<u32> {
    let mut s = seed | 1;
    let mut out = vec![0, 1, 0x8000_0000, 0xFFFF_FFFF, 0x8000_0001, 0x1234_5678];
    while out.len() < n {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        out.push(s);
    }
    out.truncate(n);
    out
}

const AMOUNTS: [u32; 9] = [0, 1, 5, 31, 32, 33, 63, 64, 0xFFFF_FFE1];

#[test]
fn and_or_xor_not_match_the_host() {
    let n = 1000;
    let (xs, ys) = (words(n, 1), words(n, 2));
    let (a, b) = (u32s(&xs), u32s(&ys));
    let zip = |f: fn(u32, u32) -> u32| -> Vec<u32> { xs.iter().zip(&ys).map(|(&x, &y)| f(x, y)).collect() };

    assert_eq!((&a & &b).to_vec().unwrap(), zip(|x, y| x & y));
    assert_eq!((&a | &b).to_vec().unwrap(), zip(|x, y| x | y));
    assert_eq!((&a ^ &b).to_vec().unwrap(), zip(|x, y| x ^ y));
    assert_eq!((!&a).to_vec().unwrap(), xs.iter().map(|x| !x).collect::<Vec<_>>());
    // Owned operands, scalars on either side, a fused chain.
    assert_eq!((a.clone() ^ b.clone()).to_vec().unwrap(), zip(|x, y| x ^ y));
    assert_eq!((&a & 0xFF00_FF00u32).to_vec().unwrap(), xs.iter().map(|x| x & 0xFF00_FF00).collect::<Vec<_>>());
    assert_eq!((0x0F0F_0F0Fu32 | &a).to_vec().unwrap(), xs.iter().map(|x| 0x0F0F_0F0F | x).collect::<Vec<_>>());
    assert_eq!((0xDEAD_BEEFu32 ^ a.clone()).to_vec().unwrap(), xs.iter().map(|x| 0xDEAD_BEEF ^ x).collect::<Vec<_>>());
    let chain = !((&a ^ &b) & (&a | 0x1111u32)) ^ &b;
    let want: Vec<u32> = xs.iter().zip(&ys).map(|(&x, &y)| !((x ^ y) & (x | 0x1111)) ^ y).collect();
    assert_eq!(chain.to_vec().unwrap(), want);
    // Broadcasting a one-element array and the `Scalar` handle.
    let one = u32s(&[0x00FF_00FF]);
    assert_eq!((&a & &one).to_vec().unwrap(), xs.iter().map(|x| x & 0x00FF_00FF).collect::<Vec<_>>());
    let s = Scalar::<u32>::new(0xF0F0_F0F0);
    assert_eq!((&a ^ &s).to_vec().unwrap(), xs.iter().map(|x| x ^ 0xF0F0_F0F0).collect::<Vec<_>>());
    assert_eq!((&s | &a).to_vec().unwrap(), xs.iter().map(|x| 0xF0F0_F0F0 | x).collect::<Vec<_>>());
}

#[test]
fn compound_assignments_match_the_host() {
    let xs = words(300, 3);
    let ys = words(300, 4);
    let b = u32s(&ys);
    let mut a = u32s(&xs);
    a &= &b;
    a |= 0x8000_0001u32;
    a ^= &b;
    a <<= 3u32;
    a >>= &u32s(&vec![2; 300]);
    let want: Vec<u32> = xs.iter().zip(&ys).map(|(&x, &y)| ((((x & y) | 0x8000_0001) ^ y) << 3) >> 2).collect();
    assert_eq!(a.to_vec().unwrap(), want);
    // A clone is another handle on the same data: it sees the write.
    let mut c = u32s(&xs);
    let keep = c.clone();
    c ^= 0xFFFF_FFFFu32;
    assert_eq!(c.to_vec().unwrap(), xs.iter().map(|x| !x).collect::<Vec<_>>());
    assert_eq!(keep.to_vec().unwrap(), c.to_vec().unwrap());
}

#[test]
fn shifts_and_rotations_take_the_amount_modulo_32() {
    let xs = words(500, 5);
    let x = u32s(&xs);
    for amount in AMOUNTS {
        let m = amount & 31;
        let host = |f: fn(u32, u32) -> u32| -> Vec<u32> { xs.iter().map(|&v| f(v, m)).collect() };
        assert_eq!(x.shl(amount).to_vec().unwrap(), host(|v, m| v << m), "shl {amount}");
        assert_eq!(x.shr(amount).to_vec().unwrap(), host(|v, m| v >> m), "shr {amount}");
        assert_eq!(x.rotate_left(amount).to_vec().unwrap(), host(u32::rotate_left), "rotate_left {amount}");
        assert_eq!(x.rotate_right(amount).to_vec().unwrap(), host(u32::rotate_right), "rotate_right {amount}");
        assert_eq!((&x << amount).to_vec().unwrap(), host(|v, m| v << m), "<< {amount}");
        assert_eq!((&x >> amount).to_vec().unwrap(), host(|v, m| v >> m), ">> {amount}");
    }
}

#[test]
fn array_amounts_and_scalar_left_forms() {
    let xs = words(700, 6);
    let amounts: Vec<u32> = (0..700u32).map(|i| AMOUNTS[i as usize % AMOUNTS.len()] ^ ((i / 9) * 3)).collect();
    let (x, n) = (u32s(&xs), u32s(&amounts));
    let zip = |f: fn(u32, u32) -> u32| -> Vec<u32> { xs.iter().zip(&amounts).map(|(&v, &m)| f(v, m & 31)).collect() };
    assert_eq!(x.shl(&n).to_vec().unwrap(), zip(|v, m| v << m));
    assert_eq!(x.shr(&n).to_vec().unwrap(), zip(|v, m| v >> m));
    assert_eq!(x.rotate_left(&n).to_vec().unwrap(), zip(u32::rotate_left));
    assert_eq!(x.rotate_right(&n).to_vec().unwrap(), zip(u32::rotate_right));
    assert_eq!((&x << &n).to_vec().unwrap(), zip(|v, m| v << m));
    assert_eq!((x.clone() >> n.clone()).to_vec().unwrap(), zip(|v, m| v >> m));
    // The value on the left, the amounts in the array: `1 << n`.
    assert_eq!((0x8000_0001u32 << &n).to_vec().unwrap(), amounts.iter().map(|&m| 0x8000_0001u32 << (m & 31)).collect::<Vec<_>>());
    assert_eq!((0x8000_0001u32 >> &n).to_vec().unwrap(), amounts.iter().map(|&m| 0x8000_0001u32 >> (m & 31)).collect::<Vec<_>>());
    // Fused with arithmetic around it.
    let fused = ((&x + 1u32).rotate_left(&n) ^ (&x << 7u32)) + 3u32;
    let want: Vec<u32> = xs.iter().zip(&amounts).map(|(&v, &m)| (v.wrapping_add(1).rotate_left(m & 31) ^ (v << 7)).wrapping_add(3)).collect();
    assert_eq!(fused.to_vec().unwrap(), want);
}

#[test]
fn a_64_bit_rotation_from_32_bit_halves() {
    let (lo, hi) = (words(200, 7), words(200, 8));
    let (l, h) = (u32s(&lo), u32s(&hi));
    for r in [1u32, 5, 31, 32, 33, 63] {
        // (lo, hi) rotated left by r: swap the halves for r >= 32, then funnel-shift.
        let (a, b) = if r >= 32 { (&h, &l) } else { (&l, &h) };
        let k = r & 31;
        let (nl, nh) = if k == 0 { (a.clone(), b.clone()) } else { ((a << k) | (b >> (32 - k)), (b << k) | (a >> (32 - k))) };
        let want: Vec<u64> = lo.iter().zip(&hi).map(|(&l, &h)| (u64::from(h) << 32 | u64::from(l)).rotate_left(r)).collect();
        let got: Vec<u64> = nl.to_vec().unwrap().iter().zip(nh.to_vec().unwrap()).map(|(&l, h)| u64::from(h) << 32 | u64::from(l)).collect();
        assert_eq!(got, want, "rotl {r}");
    }
}

#[test]
fn bitwise_operations_are_fused_into_one_kernel() {
    let ctx = Context::new().unwrap();
    let xs = words(4096, 9);
    let x = Array::from_slice_on(&ctx, &xs, xs.len()).unwrap();
    let mut y = x.clone();
    for k in 0..20u32 {
        y = (y.rotate_left(k + 1) ^ (&y << 3u32)) & 0xFFFF_FFF0u32 | (&y >> 7u32);
    }
    let got = if ctx.profiling_supported() {
        let (got, profile) = ctx.profile(|| y.to_vec().unwrap()).unwrap();
        assert_eq!(profile.launches(), 1, "{profile}");
        got
    } else {
        y.to_vec().unwrap()
    };
    let want: Vec<u32> = xs
        .iter()
        .map(|&v| {
            let mut y = v;
            for k in 0..20u32 {
                y = ((y.rotate_left(k + 1) ^ (y << 3)) & 0xFFFF_FFF0) | (y >> 7);
            }
            y
        })
        .collect();
    assert_eq!(got, want);
}

// -- u32 records and loops --

#[derive(Fields)]
struct State {
    lanes: [Array<u32>; 4],
    carry: Array<u32>,
}

#[derive(Fields)]
struct Pair {
    a: Array<u32>,
    b: Array<u32>,
}

#[test]
fn a_record_of_u32_arrays_does_arithmetic_and_views() {
    let n = 64;
    let xs = words(n, 10);
    let rec = Pair { a: u32s(&xs), b: u32s(&xs) };
    let sum = &rec + &rec;
    assert_eq!(sum.a.to_vec().unwrap(), xs.iter().map(|x| x.wrapping_mul(2)).collect::<Vec<_>>());
    let halves = rec.slice(0, 0..32);
    assert_eq!(halves.a.to_vec().unwrap(), xs[..32].to_vec());
    let bumped = &rec + 1u32;
    assert_eq!(bumped.b.to_vec().unwrap(), xs.iter().map(|x| x.wrapping_add(1)).collect::<Vec<_>>());
}

/// An xorshift-style step on every lane of a state, with the lanes mixed.
fn mix(s: &[Array<u32>; 4]) -> [Array<u32>; 4] {
    let [a, b, c, d] = s;
    [(a ^ (b << 13u32)).rotate_left(5u32), b ^ (c >> 7u32), (c + d) ^ a, !d ^ a.rotate_right(3u32)]
}

fn mix_host(s: [u32; 4]) -> [u32; 4] {
    let [a, b, c, d] = s;
    [(a ^ (b << 13)).rotate_left(5), b ^ (c >> 7), c.wrapping_add(d) ^ a, !d ^ a.rotate_right(3)]
}

#[test]
fn iterate_all_runs_u32_states() {
    let n = 1500;
    let init: Vec<Vec<u32>> = (0..4).map(|k| words(n, 20 + k)).collect();
    let arrays: Vec<Array<u32>> = init.iter().map(|v| u32s(v)).collect();
    for steps in [0usize, 1, 5, 40] {
        let got = iterate_all([&arrays[0], &arrays[1], &arrays[2], &arrays[3]], steps, mix);
        for (k, g) in got.iter().enumerate() {
            let want: Vec<u32> = (0..n)
                .map(|i| {
                    let mut s = [init[0][i], init[1][i], init[2][i], init[3][i]];
                    for _ in 0..steps {
                        s = mix_host(s);
                    }
                    s[k]
                })
                .collect();
            assert_eq!(g.to_vec().unwrap(), want, "state {k} after {steps} steps");
        }
    }
}

#[test]
fn iterate_until_all_runs_a_u32_record() {
    let n = 256;
    let init: Vec<Vec<u32>> = (0..4).map(|k| words(n, 30 + k)).collect();
    let state = State { lanes: std::array::from_fn(|k| u32s(&init[k])), carry: Array::<u32>::full(n, 0).unwrap() };
    let run = iterate_until_all(
        state,
        Until::new(50)
            .record("done", |s: &State| select(&s.carry.ge(5u32), 1.0f32, 0.0f32).max())
            .stop_when(|h| h["done"].last().ge(1.0)),
        |s, _| State {
            lanes: [
                &s.lanes[1] ^ &s.carry,
                s.lanes[2].rotate_left(&(&s.carry & 31u32)),
                &s.lanes[3] + &s.lanes[0],
                s.lanes[0].shr(1u32),
            ],
            carry: &s.carry + 1u32,
        },
    )
    .unwrap();
    assert_eq!(run.steps, 5);
    let mut want = init.clone();
    for carry in 0..5u32 {
        let l = want.clone();
        for i in 0..n {
            want[0][i] = l[1][i] ^ carry;
            want[1][i] = l[2][i].rotate_left(carry & 31);
            want[2][i] = l[3][i].wrapping_add(l[0][i]);
            want[3][i] = l[0][i] >> 1;
        }
    }
    for k in 0..4 {
        assert_eq!(run.state.lanes[k].to_vec().unwrap(), want[k], "lane {k}");
    }
    assert_eq!(run.state.carry.to_vec().unwrap(), vec![5; n]);
}
