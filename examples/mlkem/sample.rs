//! SampleNTT (FIPS 203 Algorithm 7): rejection sampling of the coefficients of a polynomial in the
//! NTT domain from the SHAKE128 stream, without a data-dependent loop.
//!
//! A fixed number of 168-byte blocks is squeezed for every polynomial; the stream is read as 12-bit
//! candidates, eight per three words (a group). Per group the number of accepted candidates is
//! counted, and together with the accepted candidates before each candidate of the group packed in
//! one word; an inclusive scan over the groups gives each group's first slot, and one scatter
//! writes every accepted candidate to its slot (the ones past slot 255 are dropped). The blocks
//! suffice with overwhelming probability, not always: the caller checks the returned minimum
//! number of accepted candidates and squeezes more blocks when some polynomial got fewer than 256.

use luma::*;

use super::field::Q;

/// Candidates per group and words per group.
const PER_GROUP: usize = 8;
const GROUP_WORDS: usize = 3;
/// Words of a SHAKE128 block (rate 168 bytes).
pub const BLOCK_WORDS: usize = 42;

/// The polynomials sampled from `words` (`[rows, 42 * blocks]`, the squeezed streams) as a
/// `[256, rows]` array, and the smallest number of candidates any row accepted (at least 256 when
/// every row is complete).
pub fn sample_ntt(words: &Array<u32>) -> (Array<u32>, Array<u32>) {
    let ctx = words.context();
    let [rows, total_words, ..] = words.shape().dims();
    let groups = total_words / GROUP_WORDS;

    // Per group: the 12-bit fields of its three words.
    let by_word = words.reshape([rows, GROUP_WORDS, groups]);
    let w: Vec<Array<u32>> = (0..GROUP_WORDS).map(|t| by_word.slice(1, t..t + 1)).collect();
    let field = |c: usize| -> Array<u32> {
        let (a, shift) = ((12 * c) / 32, (12 * c) % 32);
        let low = &w[a] >> shift as u32;
        let v = if shift > 20 { low | (&w[a + 1] << (32 - shift as u32)) } else { low };
        v & 0xFFFu32
    };
    let accepted: Vec<Array<u32>> = (0..PER_GROUP).map(|c| field(c).lt(Q)).collect();
    // The accepted candidates before candidate `c` of the group, three bits each for c = 1..8.
    let mut before = accepted[0].clone();
    let mut packed_before: Option<Array<u32>> = None;
    for (c, a) in accepted.iter().enumerate().skip(1) {
        let term = &before << (3 * (c as u32 - 1));
        packed_before = Some(match packed_before {
            Some(p) => p | term,
            None => term,
        });
        // A new lazy array, not `+=`: that would write into the buffer `before` may share.
        #[allow(clippy::assign_op_pattern)]
        {
            before = before + a;
        }
    }
    let count = before; // accepted candidates of the whole group
    let first_slot = count.cumsum(2) - &count;
    let ends = &first_slot + &count;
    // Slots from 256 on are dropped, so the count is clamped to fit its 9 bits however many blocks there are.
    let info = (first_slot.minimum(256u32) | (packed_before.expect("eight candidates") << 9u32)).cache();
    let accepted_in_row = ends.slice(2, groups - 1..groups);
    let least = accepted_in_row.min();

    // Every candidate: `c` fastest, so that neighbouring threads write neighbouring slots of one
    // polynomial, then a few rows, so that the threads that read the words of a group read
    // neighbouring words.
    let near = [8usize, 4, 2, 1].into_iter().find(|&n| rows.is_multiple_of(n)).expect("1 divides every row count");
    let shape = [PER_GROUP, near, groups, rows / near];
    let coordinate = |axis: usize| Array::<u32>::range_on(ctx, shape, axis).expect("index array");
    let (c, g) = (coordinate(0), coordinate(2));
    let row = coordinate(1) + coordinate(3) * near as u32;
    let word = |a: Array<u32>| words.take(&(&row + a * rows as u32));
    let bit = &c * 12u32;
    let a = (&bit >> 5u32) + &g * 3u32;
    let shift = &bit & 31u32;
    let low = word(a.clone()).shr(&shift);
    let high = word(a + 1u32).shl(&(32u32 - &shift));
    let value = (low | select(&shift.gt(20u32), &high, 0u32)) & 0xFFFu32;
    let packed = info.take(&(&row + &g * rows as u32));
    let before_in_group = (packed.clone() >> (&c * 3u32 + 6u32)) & 7u32;
    let slot = (&packed & 0x1FFu32) + before_in_group * c.gt(0u32);
    // Rejected candidates and the ones past slot 255 go beyond the array.
    let valid = value.lt(Q) * slot.lt(256u32);
    let index = select(&valid, &(&slot + &row * 256u32), rows as u32 * 256);
    let polynomials = Array::<u32>::zeros_on(ctx, [256, rows]).expect("polynomials");
    polynomials.put(&index, &value).expect("scatter");
    (polynomials, least)
}
