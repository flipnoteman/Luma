# Integer and bit operations

`Array<u32>` is the integer type: indices, masks, counters, and the lanes of bit-sliced
cryptography. A `u32` array fuses, views, reduces, scans and loops like an `f32` one, and has the
bit operations of a hash function. There is no signed integer type; `u32` arithmetic wraps.

## Arithmetic

`+ - * / %` work on `u32` arrays and `u32` scalars, and wrap modulo `2^32` where a Rust
`wrapping_*` would. Division and remainder by an element that is zero follow WGSL: `x / 0` is `x`
(the dividend), so check divisors where a zero is possible.

```rust
use luma::*;

let x = Array::from_slice(&[7u32, u32::MAX, 0, 100], 4)?;
assert_eq!((&x + 1u32).to_vec()?, vec![8, 0, 1, 101]);              // wraps
assert_eq!((&x - 1u32).to_vec()?, vec![6, u32::MAX - 1, u32::MAX, 99]);
assert_eq!((&x * 3u32).to_vec()?, vec![21, u32::MAX - 2, 0, 300]);
assert_eq!((&x % 10u32).to_vec()?, vec![7, 5, 0, 0]);
assert_eq!((&x / 0u32).to_vec()?, vec![7, u32::MAX, 0, 100]);       // WGSL: x / 0 == x
# Ok::<(), luma::LumaError>(())
```

## Bit operations

| Operation | Notes |
| --- | --- |
| `&`, `\|`, `^`, `!` | bitwise and, or, xor, complement (`!&x`) |
| `<<`, `>>` / `shl`, `shr` | logical shifts; the amount is taken **modulo 32**: a shift by 32 is no shift |
| `rotate_left`, `rotate_right` | rotation of the 32 bits, amount modulo 32 |
| `&=`, `\|=`, `^=`, `<<=`, `>>=` | compound forms, in place |

The right-hand side of every binary operation is an array of the same shape (with the usual
[broadcasting](arrays.md#shapes-and-broadcasting)) or a `u32` scalar. The shifts and rotations also
take an array of per-element amounts.

```rust
use luma::*;

let x = Array::from_slice(&[0x8000_0001u32, 0x1234_5678], 2)?;
assert_eq!((&x & 0xFFu32).to_vec()?,       vec![0x01, 0x78]);
assert_eq!((&x ^ &x).to_vec()?,            vec![0, 0]);
assert_eq!((!&x).to_vec()?,                vec![0x7FFF_FFFE, 0xEDCB_A987]);
assert_eq!((&x << 4u32).to_vec()?,         vec![0x0000_0010, 0x2345_6780]);
assert_eq!((&x >> 4u32).to_vec()?,         vec![0x0800_0000, 0x0123_4567]);
assert_eq!(x.rotate_left(4u32).to_vec()?,  vec![0x0000_0018, 0x2345_6781]);
assert_eq!(x.shl(33u32).to_vec()?,         (&x << 1u32).to_vec()?);  // 33 mod 32 == 1

// Per-element amounts:
let amounts = Array::from_slice(&[4u32, 8], 2)?;
assert_eq!(x.rotate_right(&amounts).to_vec()?, vec![0x1800_0000, 0x7812_3456]);
# Ok::<(), luma::LumaError>(())
```

**Constant amounts are cheaper.** When the amount is a host scalar (`x << 13u32`,
`x.rotate_left(7u32)`) the amount is compiled into the kernel as a constant: no masking, and one
instruction. Hash functions are written with constant amounts, so this is the path they use. An
array of amounts costs a mask and a variable shift per element.

## Comparisons, masks and `select`

Comparisons (`lt`, `le`, `gt`, `ge`, `eq`, `ne`) of any element type return `Array<u32>` masks of
`0` or `1`. Masks combine with `&`, `|` and `^`. Negate a mask with `^ 1u32`: `!` flips all 32 bits, and
`!1` is not `0`. `select(&mask, a, b)` takes `a` where the mask is non-zero and `b`
elsewhere, for any element type; `a` and `b` are arrays or scalars.

```rust
use luma::*;

let x = Array::from_slice(&[3u32, 9, 12, 5], 4)?;
let in_range = x.ge(4u32) & x.lt(10u32);               // 1 where 4 <= x < 10
assert_eq!(in_range.to_vec()?, vec![0, 1, 0, 1]);
assert_eq!((&in_range ^ 1u32).to_vec()?, vec![1, 0, 1, 0]);

// Clamp to 10, keep the rest.
let clamped = select(&x.gt(10u32), 10u32, &x);
assert_eq!(clamped.to_vec()?, vec![3, 9, 10, 5]);

// Counting is summing a mask; `sum_where` sums values under a mask.
assert_eq!(in_range.sum().item()?, 2);
assert_eq!(x.sum_where(&in_range).item()?, 14);
# Ok::<(), luma::LumaError>(())
```

`u32` has no unary minus and no implicit conversion to `f32`; to use a mask in float arithmetic
write `select(&mask, 1.0, 0.0)`, which fuses.

## 64-bit lanes as pairs of `u32`

A 64-bit lane is a pair `(lo, hi)` of `u32` arrays, one element per independent instance. A
rotation by a constant swaps the halves from 32 and then funnels bits across them. This is the
`rotl64` of [`examples/keccak.rs`](https://github.com/flipnoteman/Luma/blob/main/examples/keccak.rs):

```rust
use luma::*;

/// `(lo, hi)` rotated left by `r` bits, `r` a constant.
fn rotl64(lo: &Array<u32>, hi: &Array<u32>, r: u32) -> (Array<u32>, Array<u32>) {
    let (lo, hi, k) = if r % 64 >= 32 { (hi, lo, r % 64 - 32) } else { (lo, hi, r % 64) };
    if k == 0 {
        return (lo.clone(), hi.clone());
    }
    ((lo << k) | (hi >> (32 - k)), (hi << k) | (lo >> (32 - k)))
}

let lanes = [0x0123_4567_89AB_CDEFu64, 0x8000_0000_0000_0001];
let lo = Array::from_slice(&lanes.map(|l| l as u32), 2)?;
let hi = Array::from_slice(&lanes.map(|l| (l >> 32) as u32), 2)?;

for r in [1u32, 28, 32, 45] {
    let (rl, rh) = rotl64(&lo, &hi, r);
    let (rl, rh) = (rl.to_vec()?, rh.to_vec()?);
    for (i, lane) in lanes.iter().enumerate() {
        assert_eq!(u64::from(rl[i]) | u64::from(rh[i]) << 32, lane.rotate_left(r));
    }
}
# Ok::<(), luma::LumaError>(())
```

Keccak-f[1600] is 25 such lanes, 24 rounds of `^ & | !` and rotations on 50 arrays.
`examples/keccak.rs` writes the permutation as a Rust function on `[Array<u32>; 50]` and runs a whole
permutation per [`iterate_all`](loops.md) step; `examples/mlkem` (`keccak_f_loop`) runs one round
per step, with the round constant from an LFSR carried as a 51st state word.
[Worked examples](examples.md#keccak) shows how to run them and what to look at.

## Lookup tables with `take`

`table.take(&indices)` gathers: element `i` of the result is `table[indices[i]]`, with the result
shaped like `indices`. A table is just an array, and `indices` may be a lazy expression with
broadcasting, so a lookup is a fused gather. An index past the end is clamped to the last element.

The hypothesis matrix of a correlation power analysis (CPA) of a block cipher is exactly that: for
every trace's plaintext byte `pt` and every key guess `k`, the Hamming weight of the S-box output
`HW[SBOX[pt ^ k]]`. A `[traces, 1]` array of plaintexts xored with a `[1, 256]` array of key guesses
is a `[traces, 256]` array of indices; two gathers give the hypotheses; a few reductions turn them
into 256 Pearson correlations against the measured traces.

```rust
use luma::*;

// A stand-in S-box (any permutation of 0..256; use the AES table for the real thing)
// and the table of Hamming weights, as f32 so the lookup gives a float directly.
let sbox: Vec<u32> = (0..256u32).map(|i| (i * 167 + 13) & 255).collect();
let weights: Vec<f32> = (0..256u32).map(|i| i.count_ones() as f32).collect();
let sbox_t = Array::from_slice(&sbox, 256)?;
let hw_t = Array::from_slice(&weights, 256)?;

// 200 traces: plaintext bytes, and a synthetic leakage of the secret key byte 0x2b.
let n = 200usize;
let pts: Vec<u32> = (0..n as u32).map(|i| (i * 37 + 11) & 255).collect();
let secret = 0x2bu32;
let leak: Vec<f32> = pts.iter().map(|&p| weights[sbox[(p ^ secret) as usize] as usize]).collect();

let pt = Array::from_slice(&pts, [n, 1])?;           // [traces, 1]
let t = Array::from_slice(&leak, [n, 1])?;           // measured samples
let guesses = Array::<u32>::range([1, 256], 1)?;     // [1, 256]: the key guess along axis 1

// Hypotheses for every (trace, guess): HW[SBOX[pt ^ k]], a [200, 256] f32 array.
let h = hw_t.take(&sbox_t.take(&(&pt ^ &guesses)));
assert_eq!(h.shape(), Shape::from([n, 256]));

// Pearson correlation of every column with the trace, along the trace axis.
let nf = n as f32;
let hc = &h - &(h.sum_axis(0) / nf);                 // centre each column: [200, 256] - [1, 256]
let tc = &t - &(t.sum_axis(0) / nf);                 // [200, 1] - [1, 1]
let num = (&hc * &tc).sum_axis(0);                   // [1, 256]
let den = ((&hc * &hc).sum_axis(0) * (&tc * &tc).sum_axis(0)).sqrt();
let corr = (num / den).to_vec()?;

let best = (0..256).max_by(|&a, &b| corr[a].total_cmp(&corr[b])).unwrap();
assert_eq!(best, 0x2b);
assert!(corr[best] > 0.999);
# Ok::<(), luma::LumaError>(())
```

The gathers and the elementwise arithmetic fuse, but `sum_axis` does not fuse its operand
([Reductions](reductions.md#scans)): the arrays it reduces (`hc * tc`, ...) are computed into
`[traces, 256]` buffers first. Wrap the analysis in `ctx.profile(..)` to see the kernels.

## States in `iterate_all`

`u32` arrays are a valid state for [`iterate_all`](loops.md#elementwise-loops-iterate_all). An
integer step that is elementwise, such as a hash round or a random-number generator, runs as a
kernel that keeps the state in registers, in launches of about 40 ms of GPU time:

```rust
use luma::*;

// xorshift32, 1000 steps per element.
let seed = Array::from_slice(&[1u32, 2, 3], 3)?;
let [x] = iterate_all([&seed], 1000, |[x]| {
    let a = x ^ (x << 13u32);
    let b = &a ^ (&a >> 17u32);
    [&b ^ (&b << 5u32)]
});

let host = |mut s: u32| {
    for _ in 0..1000 {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
    }
    s
};
assert_eq!(x.to_vec()?, vec![host(1), host(2), host(3)]);
# Ok::<(), luma::LumaError>(())
```

For a stream of random numbers, `Array::uniform(shape, seed)` and `Array::normal` are counter-based
generators that fuse into the kernel that reads them; use your own recurrence when you need exact
integer sequences.
