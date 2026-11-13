//! Keccak-f[1600] as a loop of 24 rounds (one traced round, so the host builds about 700 nodes per
//! permutation instead of 10 000) and the hash functions of FIPS 203 on batches of rows.

use luma::*;

use super::keccak::{keccak_round_with, sponge_with, Iota, State};

/// The round constants of a round as lanes, from the 8-bit LFSR of the Keccak specification:
/// `lfsr` is the register at the start of the round; the register after the round is returned too.
fn round_constant(lfsr: &Array<u32>) -> (Array<u32>, Array<u32>, Array<u32>) {
    let mut r = lfsr.clone();
    // Bit `j` of the LFSR output lands on bit `2^j - 1` of the 64-bit constant.
    let mut lo: Option<Array<u32>> = None;
    let mut hi: Option<Array<u32>> = None;
    for j in 0..7u32 {
        let bit = &r & 1u32;
        let position = (1u32 << j) - 1;
        let placed = if position < 32 { &bit << position } else { &bit << (position - 32) };
        if position < 32 {
            lo = Some(match lo {
                Some(acc) => acc | placed,
                None => placed,
            });
        } else {
            hi = Some(placed);
        }
        // x^8 + x^6 + x^5 + x^4 + 1: shift left, and xor 0x71 when bit 7 was set.
        r = ((&r << 1u32) & 0xFFu32) ^ (((&r >> 7u32) & 1u32) * 0x71u32);
    }
    (lo.expect("seven bits"), hi.expect("bit 63"), r)
}

/// Keccak-f[1600]: the 24 rounds as a register loop of one traced round.
pub fn keccak_f_loop(a: &State) -> State {
    let one = Array::<u32>::full_on(a[0].context(), a[0].shape(), 1).expect("a register");
    // The absorbed words are one kernel, not one per word: requested together, then recorded (and
    // the GPU started on everything before this) rather than postponed while it is busy.
    eval_all(&a.iter().collect::<Vec<_>>()).expect("absorbed state");
    a[0].context().flush().expect("flush");
    let init: [&Array<u32>; 51] = std::array::from_fn(|i| if i < 50 { &a[i] } else { &one });
    let out = iterate_all(init, 24, |s: &[Array<u32>; 51]| {
        let state: State = std::array::from_fn(|i| s[i].clone());
        let (lo, hi, next) = round_constant(&s[50]);
        let mut rounded = keccak_round_with(&state, Iota::Lanes(&lo, &hi)).into_iter();
        std::array::from_fn(|i| if i < 50 { rounded.next().expect("50 words") } else { next.clone() })
    });
    let mut out = out.into_iter();
    std::array::from_fn(|_| out.next().expect("50 words"))
}

/// The sponge of every row with the looped permutation.
#[allow(clippy::too_many_arguments)]
pub fn sponge(ctx: &Context, rows: impl Into<Shape>, inputs: &[Array<u32>], len: usize, rate: usize, suffix: u8, out_len: usize) -> Result<Vec<Array<u32>>> {
    sponge_with(ctx, rows, inputs, len, rate, suffix, out_len, &keccak_f_loop)
}

const SHA3: u8 = 0x06;
const SHAKE: u8 = 0x1F;

/// SHA3-256 of `len`-byte messages: 8 words per row.
pub fn sha3_256(ctx: &Context, rows: impl Into<Shape>, input: &[Array<u32>], len: usize) -> Result<Vec<Array<u32>>> {
    sponge(ctx, rows, input, len, 136, SHA3, 32)
}

/// SHA3-512 of `len`-byte messages: 16 words per row.
pub fn sha3_512(ctx: &Context, rows: impl Into<Shape>, input: &[Array<u32>], len: usize) -> Result<Vec<Array<u32>>> {
    sponge(ctx, rows, input, len, 72, SHA3, 64)
}

/// SHAKE256 of `len`-byte messages, `out_len` bytes per row.
pub fn shake256(ctx: &Context, rows: impl Into<Shape>, input: &[Array<u32>], len: usize, out_len: usize) -> Result<Vec<Array<u32>>> {
    sponge(ctx, rows, input, len, 136, SHAKE, out_len)
}

/// SHAKE128 of `len`-byte messages, `out_len` bytes per row.
pub fn shake128(ctx: &Context, rows: impl Into<Shape>, input: &[Array<u32>], len: usize, out_len: usize) -> Result<Vec<Array<u32>>> {
    sponge(ctx, rows, input, len, 168, SHAKE, out_len)
}
