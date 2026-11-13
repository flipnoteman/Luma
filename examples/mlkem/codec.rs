//! ByteEncode / ByteDecode, Compress / Decompress and the sampling of noise (FIPS 203 Algorithms 5,
//! 6, 7 and 8) as lazy array operations.
//!
//! Byte strings are little-endian `u32` words. A batch of byte strings is a `[n, words]` array
//! (word `w` of instance `i` is element `i + n * w`, so threads of neighbouring instances read
//! neighbouring words); a polynomial batch is `[rows, 256]` (see `field`).

use luma::*;

use super::field::Q;

/// The `bits` bits (`bits <= 25`) starting at bit `bit` of the little-endian bit string whose
/// word `a` is `word(a)`: two words are read and the bits of the second one that are needed are
/// ORed in. `bit` is an array of bit positions (`>= 0`).
fn window(bit: &Array<u32>, bits: u32, word: impl Fn(&Array<u32>) -> Array<u32>) -> Array<u32> {
    let index = bit >> 5u32;
    let shift = bit & 31u32;
    let low = word(&index).shr(&shift);
    // The next word contributes when the field runs over the end of the first one.
    let high = word(&(&index + 1u32)).shl(&(32u32 - &shift));
    let spills = (&shift + bits).gt(32u32);
    (low | select(&spills, &high, 0u32)) & ((1u32 << bits) - 1)
}

/// `ByteDecode_d` of one polynomial per instance: coefficient `kappa` is the `d` bits at bit
/// `d * kappa` of the words `first..first + 8 d` of `words`, a `[n, w]` array. The result is `[n, 256]`.
pub fn unpack(words: &Array<u32>, first: usize, d: u32) -> Array<u32> {
    let ctx = words.context();
    let n = words.shape().dims()[0];
    let row = Array::<u32>::range_on(ctx, [n, 256], 0).expect("index array");
    let kappa = Array::<u32>::range_on(ctx, [n, 256], 1).expect("index array");
    let bit = &kappa * d;
    window(&bit, d, |index| words.take(&(&row + (index + first as u32) * n as u32)))
}

/// `ByteEncode_d` of one polynomial per instance (`coefficients` is `[n, 256]` with values below
/// `2^d`): the result is `[n, 8 d]` words.
pub fn pack(coefficients: &Array<u32>, d: u32) -> Array<u32> {
    let ctx = coefficients.context();
    let n = coefficients.shape().dims()[0];
    let row = Array::<u32>::range_on(ctx, [n, 8 * d as usize], 0).expect("index array");
    let word = Array::<u32>::range_on(ctx, [n, 8 * d as usize], 1).expect("index array");
    let start = &word * 32u32;
    // The coefficients that have bits in word `w`: `32 w / d` up to `(32 w + 31) / d`.
    let first = &start / d;
    let mut acc: Option<Array<u32>> = None;
    for t in 0..(31 / d + 2) {
        let kappa = &first + t;
        let c = coefficients.take(&(&row + &kappa * n as u32));
        let at = &kappa * d;
        // Bit offset of the coefficient in the word: negative when it began in the word before.
        let from_left = at.ge(&start);
        let up = select(&from_left, &(&at - &start), 0u32);
        let down = select(&from_left, 0u32, &(&start - &at));
        let inside = up.lt(32u32) | (1u32 - &from_left);
        let placed = select(&from_left, &c.shl(&up), &c.shr(&down));
        let term = select(&inside, &placed, 0u32);
        acc = Some(match acc {
            Some(sum) => sum | term,
            None => term,
        });
    }
    acc.expect("at least one term")
}

/// `Compress_d` of values below q: `round(2^d x / q) mod 2^d`.
pub fn compress(x: &Array<u32>, d: u32) -> Array<u32> {
    if d == 1 {
        // 833 ..= 2496 round to 1.
        return x.gt(832u32) * x.lt(2497u32);
    }
    (((x << d) + (Q / 2)) / Q) & ((1u32 << d) - 1)
}

/// `Decompress_d`: `round(q y / 2^d)`.
pub fn decompress(y: &Array<u32>, d: u32) -> Array<u32> {
    (y * Q + (1u32 << (d - 1))) >> d
}

/// `CBD_eta` of the PRF output of every row: `words` is `[rows, 16 eta]`; the result `[rows, 256]`
/// has coefficient `x - y mod q` with `x`, `y` the sums of `eta` bits each (Algorithm 8).
pub fn cbd(words: &Array<u32>, eta: u32) -> Array<u32> {
    let ctx = words.context();
    let rows = words.shape().dims()[0];
    let row = Array::<u32>::range_on(ctx, [rows, 256], 0).expect("index array");
    let kappa = Array::<u32>::range_on(ctx, [rows, 256], 1).expect("index array");
    let bits = 2 * eta;
    let v = window(&(&kappa * bits), bits, |index| words.take(&(&row + index * rows as u32)));
    let count = |from: u32| (0..eta).map(|b| (&v >> (from + b)) & 1u32).reduce(|a, b| a + b).expect("eta >= 1");
    let (x, y) = (count(0), count(eta));
    let c = (x + Q) - y;
    c.minimum(&(&c - Q))
}
