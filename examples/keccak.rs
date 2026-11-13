//! Keccak-f[1600], SHA3-256 and SHAKE128 over a batch of independent inputs, written only with
//! lazy array operations: no custom kernel, no WGSL. A 64-bit lane is a `(lo, hi)` pair of `u32`
//! arrays with one element per instance, the permutation is a Rust loop over 24 rounds of
//! `^ & | ! << >>` on those arrays, and luma fuses it into kernels.
//!
//! `cargo run --release --example keccak -- [--n <instances>] [--reps <permutations per loop>] [--quick] [--no-verify]`
//!
//! The results are checked against the known answers (the permutation of the zero state, SHA3-256
//! and SHAKE128 of the empty message) and against a plain-Rust reference on random messages, then
//! the throughput of one permutation per `eval` and of `iterate_all` over many permutations is
//! printed with the kernels launched per permutation and where the expression was cut. Every row
//! is the best of five calls of the same code: the later calls of `iterate_all` run from the
//! program of the first (see `iterate_all`), which is why `host ms`, the time until the call
//! returns, is only the tracing of the step.

use luma::*;
use std::time::Instant;

const RC: [u64; 24] = [
    0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000, 0x000000000000808B, 0x0000000080000001,
    0x8000000080008081, 0x8000000000008009, 0x000000000000008A, 0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
    0x000000008000808B, 0x800000000000008B, 0x8000000000008089, 0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
    0x000000000000800A, 0x800000008000000A, 0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
];

/// Rotation of lane `x + 5 y`.
const RHO: [u32; 25] = [0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43, 25, 39, 41, 45, 15, 21, 8, 18, 2, 61, 56, 14];

/// Where lane `i = x + 5 y` goes in the rho-pi step: `y + 5 (2 x + 3 y mod 5)`.
fn pi(i: usize) -> usize {
    let (x, y) = (i % 5, i / 5);
    y + 5 * ((2 * x + 3 * y) % 5)
}

// =================================================================================================
// Plain Rust reference
// =================================================================================================

fn keccak_f_host(a: &mut [u64; 25]) {
    for rc in RC {
        let mut c = [0u64; 5];
        for x in 0..5 {
            c[x] = a[x] ^ a[x + 5] ^ a[x + 10] ^ a[x + 15] ^ a[x + 20];
        }
        for x in 0..5 {
            let d = c[(x + 4) % 5] ^ c[(x + 1) % 5].rotate_left(1);
            for y in 0..5 {
                a[x + 5 * y] ^= d;
            }
        }
        let mut b = [0u64; 25];
        for i in 0..25 {
            b[pi(i)] = a[i].rotate_left(RHO[i]);
        }
        for y in 0..5 {
            for x in 0..5 {
                a[x + 5 * y] = b[x + 5 * y] ^ (!b[(x + 1) % 5 + 5 * y] & b[(x + 2) % 5 + 5 * y]);
            }
        }
        a[0] ^= rc;
    }
}

/// The sponge of one message: `rate` bytes absorbed per permutation, `suffix` the domain
/// separation byte (0x06 for SHA3, 0x1F for SHAKE), `out_len` bytes squeezed.
pub fn sponge_host(message: &[u8], rate: usize, suffix: u8, out_len: usize) -> Vec<u8> {
    let mut state = [0u64; 25];
    let xor_block = |state: &mut [u64; 25], block: &[u8]| {
        for (i, byte) in block.iter().enumerate() {
            state[i / 8] ^= u64::from(*byte) << (8 * (i % 8));
        }
    };
    let mut chunks = message.chunks_exact(rate);
    for block in &mut chunks {
        xor_block(&mut state, block);
        keccak_f_host(&mut state);
    }
    let mut last = vec![0u8; rate];
    last[..chunks.remainder().len()].copy_from_slice(chunks.remainder());
    last[chunks.remainder().len()] ^= suffix;
    last[rate - 1] ^= 0x80;
    xor_block(&mut state, &last);
    keccak_f_host(&mut state);
    let mut out = Vec::with_capacity(out_len);
    loop {
        for i in 0..rate {
            out.push((state[i / 8] >> (8 * (i % 8))) as u8);
        }
        if out.len() >= out_len {
            out.truncate(out_len);
            return out;
        }
        keccak_f_host(&mut state);
    }
}

// =================================================================================================
// The same with lazy array operations
// =================================================================================================

/// The 25 lanes of every instance: word `2 i` is the low half of lane `i`, word `2 i + 1` the high
/// half. Every array has one element per instance.
type State = [Array<u32>; 50];

/// `(lo, hi)` rotated left by `r` bits, `r` a constant: halves swap from 32, then funnel shifts.
fn rotl64(lo: &Array<u32>, hi: &Array<u32>, r: u32) -> (Array<u32>, Array<u32>) {
    let (lo, hi, k) = if r % 64 >= 32 { (hi, lo, r % 64 - 32) } else { (lo, hi, r % 64) };
    if k == 0 {
        return (lo.clone(), hi.clone());
    }
    ((lo << k) | (hi >> (32 - k)), (hi << k) | (lo >> (32 - k)))
}

fn keccak_round(a: &State, rc: u64) -> State {
    // Word `h` (0: low, 1: high) of lane `i`.
    let word = |i: usize, h: usize| &a[2 * i + h];
    // theta
    let c: Vec<(Array<u32>, Array<u32>)> = (0..5)
        .map(|x| {
            let fold = |h: usize| (word(x, h) ^ word(x + 5, h)) ^ (word(x + 10, h) ^ word(x + 15, h)) ^ word(x + 20, h);
            (fold(0), fold(1))
        })
        .collect();
    let d: Vec<(Array<u32>, Array<u32>)> = (0..5)
        .map(|x| {
            let (rl, rh) = rotl64(&c[(x + 1) % 5].0, &c[(x + 1) % 5].1, 1);
            (&c[(x + 4) % 5].0 ^ rl, &c[(x + 4) % 5].1 ^ rh)
        })
        .collect();
    // rho and pi: lane i, xor D, rotated, lands at pi(i)
    let mut b: Vec<Option<(Array<u32>, Array<u32>)>> = vec![None; 25];
    for i in 0..25 {
        let (l, h) = (word(i, 0) ^ &d[i % 5].0, word(i, 1) ^ &d[i % 5].1);
        b[pi(i)] = Some(rotl64(&l, &h, RHO[i]));
    }
    let b: Vec<(Array<u32>, Array<u32>)> = b.into_iter().map(|lane| lane.expect("pi is a permutation")).collect();
    // chi, then iota
    let mut out = Vec::with_capacity(50);
    for y in 0..5 {
        for x in 0..5 {
            let (this, next, after) = (&b[x + 5 * y], &b[(x + 1) % 5 + 5 * y], &b[(x + 2) % 5 + 5 * y]);
            let mut l = &this.0 ^ ((!&next.0) & &after.0);
            let mut h = &this.1 ^ ((!&next.1) & &after.1);
            if x + 5 * y == 0 {
                l ^= rc as u32;
                h ^= (rc >> 32) as u32;
            }
            out.push(l);
            out.push(h);
        }
    }
    out.try_into().unwrap_or_else(|_| unreachable!("25 lanes of two words"))
}

/// Keccak-f[1600] on every instance: 24 rounds, about 10 000 elementwise operations.
fn keccak_f(a: &State) -> State {
    let mut state = keccak_round(a, RC[0]);
    for rc in &RC[1..] {
        state = keccak_round(&state, *rc);
    }
    state
}

/// The sponge on every instance: `inputs` are the words (little-endian) of the messages of
/// `len` bytes each, `inputs[w]` holding word `w` of every instance (bytes past `len` are
/// ignored); the result is `out_len` bytes per instance as `out_len / 4` rounded up words.
pub fn sponge(ctx: &Context, n: usize, inputs: &[Array<u32>], len: usize, rate: usize, suffix: u8, out_len: usize) -> Result<Vec<Array<u32>>> {
    let zero = Array::<u32>::full_on(ctx, n, 0)?;
    let mut state: State = std::array::from_fn(|_| zero.clone());
    let words = rate / 4;
    let blocks = len / rate + 1;
    for block in 0..blocks {
        // The last block holds the rest of the message, the suffix and the final bit.
        let last = block == blocks - 1;
        let rem = len - block * rate;
        let mut pad = vec![0u32; words];
        if last {
            pad[rem / 4] ^= u32::from(suffix) << (8 * (rem % 4));
            pad[words - 1] ^= 0x8000_0000;
        }
        for (w, pad) in pad.iter().enumerate() {
            let from_message = if last { 4 * w + 4 <= rem } else { true };
            let mut word = if from_message {
                Some(inputs[(block * rate) / 4 + w].clone())
            } else if last && 4 * w < rem {
                // A partial word: the message bytes only, then the suffix.
                Some(&inputs[(block * rate) / 4 + w] & (u32::MAX >> (32 - 8 * (rem - 4 * w) as u32)))
            } else {
                None
            };
            if last && *pad != 0 {
                word = Some(match word {
                    Some(m) => m ^ *pad,
                    None => Array::full_on(ctx, n, *pad)?,
                });
            }
            if let Some(word) = word {
                state[w] = &state[w] ^ word;
            }
        }
        state = keccak_f(&state);
    }
    let mut out = Vec::new();
    loop {
        out.extend(state[..words].iter().cloned());
        if out.len() * 4 >= out_len {
            out.truncate(out_len.div_ceil(4));
            return Ok(out);
        }
        state = keccak_f(&state);
    }
}

// =================================================================================================
// Checks
// =================================================================================================

/// Deterministic pseudo-random bytes.
pub fn bytes(seed: u64, n: usize) -> Vec<u8> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 24) as u8
        })
        .collect()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// `messages` (all of one length) as the words `sponge` takes.
pub fn message_words(ctx: &Context, messages: &[Vec<u8>], len: usize) -> Result<Vec<Array<u32>>> {
    let n = messages.len();
    (0..len.div_ceil(4))
        .map(|w| {
            let column: Vec<u32> = messages
                .iter()
                .map(|m| (0..4).map(|k| u32::from(m.get(4 * w + k).copied().unwrap_or(0)) << (8 * k)).fold(0, |acc, v| acc | v))
                .collect();
            Array::from_slice_on(ctx, &column, n)
        })
        .collect()
}

/// The bytes of instance `i` of the words `sponge` returned.
pub fn instance_bytes(words: &[Vec<u32>], i: usize, out_len: usize) -> Vec<u8> {
    let mut out: Vec<u8> = words.iter().flat_map(|w| w[i].to_le_bytes()).collect();
    out.truncate(out_len);
    out
}

pub fn read_all(words: &[Array<u32>]) -> Result<Vec<Vec<u32>>> {
    let refs: Vec<&Array<u32>> = words.iter().collect();
    eval_all(&refs)?;
    words.iter().map(Array::to_vec).collect()
}

fn check(what: &str, ok: bool) {
    println!("  {:<58} {}", what, if ok { "ok" } else { "MISMATCH" });
    assert!(ok, "{what} does not match");
}

fn verify(ctx: &Context) -> Result<()> {
    println!("Known answers and the plain-Rust reference");

    // The permutation of the zero state: lane 0 is 0xF1258F7940E1DDE7.
    let zero = Array::<u32>::full_on(ctx, 4, 0)?;
    let permuted = keccak_f(&std::array::from_fn(|_| zero.clone()));
    let lane0 = permuted[0].to_vec()?[0] as u64 | u64::from(permuted[1].to_vec()?[0]) << 32;
    check("Keccak-f[1600] of the zero state, lane 0", lane0 == 0xF125_8F79_40E1_DDE7);
    let mut host = [0u64; 25];
    keccak_f_host(&mut host);
    check("reference permutation of the zero state, lane 0", host[0] == 0xF125_8F79_40E1_DDE7);

    // SHA3-256 and SHAKE128 of the empty message, and "abc" for the reference.
    let sha3 = sponge(ctx, 4, &[], 0, 136, 0x06, 32)?;
    let sha3 = instance_bytes(&read_all(&sha3)?, 0, 32);
    check("SHA3-256(\"\")", hex(&sha3) == "a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a");
    let shake = sponge(ctx, 4, &[], 0, 168, 0x1F, 16)?;
    let shake = instance_bytes(&read_all(&shake)?, 0, 16);
    check("SHAKE128(\"\"), first 16 bytes", hex(&shake) == "7f9c2ba4e88f827d616045507605853e");
    check(
        "reference SHA3-256(\"abc\")",
        hex(&sponge_host(b"abc", 136, 0x06, 32)) == "3a985da74fe225b2045c172d6bd390bd855f086e3e9d525b46bfe24511431532",
    );

    // Random messages of every length class against the reference.
    let n = 300;
    for (rate, suffix, name, out_lens) in [(136usize, 0x06u8, "SHA3-256", [32usize].as_slice()), (168, 0x1F, "SHAKE128", &[16, 168, 200])] {
        for len in [0usize, 1, 3, 31, 32, 135, 136, 137, 168, 169, 300] {
            let messages: Vec<Vec<u8>> = (0..n).map(|i| bytes(1000 * len as u64 + i as u64 + 7, len)).collect();
            let words = message_words(ctx, &messages, len)?;
            for &out_len in out_lens {
                let got = read_all(&sponge(ctx, n, &words, len, rate, suffix, out_len)?)?;
                let all_match = (0..n).all(|i| instance_bytes(&got, i, out_len) == sponge_host(&messages[i], rate, suffix, out_len));
                if !all_match {
                    check(&format!("{name}, {len}-byte messages, {out_len} output bytes, {n} instances"), false);
                }
            }
        }
        println!("  {name}: 11 message lengths x {} output sizes x {n} random instances match the reference", out_lens.len());
    }
    println!();
    Ok(())
}

// =================================================================================================
// Benchmark
// =================================================================================================

fn state_data(n: usize) -> Vec<Vec<u32>> {
    (0..50)
        .map(|w| {
            let mut s = 0x9E37_79B9u32.wrapping_mul(w as u32 + 1) | 1;
            (0..n)
                .map(|_| {
                    s ^= s << 13;
                    s ^= s >> 17;
                    s ^= s << 5;
                    s
                })
                .collect()
        })
        .collect()
}

fn state_from(ctx: &Context, data: &[Vec<u32>]) -> Result<State> {
    let arrays: Vec<Array<u32>> = data.iter().map(|d| Array::from_slice_on(ctx, d, d.len())).collect::<Result<_>>()?;
    arrays.try_into().map_err(|_| LumaError::InvalidShape("50 words".into()))
}

fn eval_state(state: &State) -> Result<()> {
    let refs: Vec<&Array<u32>> = state.iter().collect();
    eval_all(&refs)
}

/// One permutation per `eval`: `reps` times evaluate `state = keccak_f(state)`.
fn one_per_eval(state: &State, reps: usize) -> Result<State> {
    let mut state = keccak_f(state);
    eval_state(&state)?;
    for _ in 1..reps {
        state = keccak_f(&state);
        eval_state(&state)?;
    }
    Ok(state)
}

/// `reps` permutations in one `iterate_all`.
fn looped(state: &State, reps: usize) -> Result<State> {
    let refs: [&Array<u32>; 50] = std::array::from_fn(|i| &state[i]);
    let out = iterate_all(refs, reps, keccak_f);
    eval_state(&out)?;
    Ok(out)
}

/// The best of `runs` calls of `f`: seconds until `f` returned (the host's share: tracing and
/// launching) and until the GPU finished what it launched.
fn best_of(ctx: &Context, runs: usize, mut f: impl FnMut() -> Result<()>) -> Result<(f64, f64)> {
    let (mut host, mut wall) = (f64::INFINITY, f64::INFINITY);
    for _ in 0..runs {
        let start = Instant::now();
        f()?;
        let returned = start.elapsed().as_secs_f64();
        ctx.sync()?;
        host = host.min(returned);
        wall = wall.min(start.elapsed().as_secs_f64());
    }
    Ok((host, wall))
}

fn row(name: &str, n: usize, reps: usize, (host, wall): (f64, f64), profile: &Profile) {
    let perms = (n * reps) as f64;
    println!(
        "{:<34} {:>10.1} {:>12.1} {:>9.1} {:>14.2} {:>9.2}",
        name,
        wall * 1e3,
        perms / wall / 1e6,
        host * 1e3,
        f64::from(profile.launches()) / reps as f64,
        profile.total().as_secs_f64() * 1e3,
    );
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let value = |flag: &str| args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).and_then(|v| v.parse::<usize>().ok());
    let quick = args.iter().any(|a| a == "--quick");
    let n = value("--n").unwrap_or(if quick { 1 << 16 } else { 1 << 20 });
    let reps = value("--reps").unwrap_or(if quick { 4 } else { 32 });

    let ctx = Context::new()?;
    if !args.iter().any(|a| a == "--no-verify") {
        verify(&ctx)?;
    }

    println!("Keccak-f[1600] on {n} independent states ({} MB of state), lazy array operations only", n * 200 / 1_000_000);
    println!("{}", "=".repeat(90));
    println!("{:<34} {:>10} {:>12} {:>9} {:>14} {:>9}", "case", "wall ms", "M perm/s", "host ms", "launches/perm", "gpu ms");
    println!("{}", "-".repeat(90));
    let data = state_data(n);
    let state = state_from(&ctx, &data)?;
    let mut profiles = Vec::new();

    // Correctness of the timed programs: the first instance against the reference.
    let mut host = [0u64; 25];
    for (i, lane) in host.iter_mut().enumerate() {
        *lane = u64::from(data[2 * i][0]) | u64::from(data[2 * i + 1][0]) << 32;
    }
    for _ in 0..reps {
        keccak_f_host(&mut host);
    }
    let matches_host = |state: &State| -> Result<bool> {
        let mut ok = true;
        for (i, lane) in host.iter().enumerate() {
            let (lo, hi) = (state[2 * i].to_vec()?[0], state[2 * i + 1].to_vec()?[0]);
            ok &= u64::from(lo) | u64::from(hi) << 32 == *lane;
        }
        Ok(ok)
    };

    type Run = fn(&State, usize) -> Result<State>;
    let cases: [(&str, Run); 2] = [("one permutation per eval", one_per_eval), ("iterate_all, one traced step", looped)];
    for (name, run) in cases {
        let out = run(&state, reps)?;
        assert!(matches_host(&out)?, "{name}: result differs from the reference");
        let seconds = best_of(&ctx, 5, || run(&state, reps).map(drop))?;
        let (_, profile) = ctx.profile(|| run(&state, reps).and_then(|_| ctx.sync()))?;
        row(name, n, reps, seconds, &profile);
        profiles.push((name, profile));
    }
    println!("{}", "=".repeat(90));
    for (name, profile) in &profiles {
        println!("\n{name}: {} launches, {} splits", profile.launches(), profile.splits.len());
        print!("{}", Profile { splits: Vec::new(), ..profile.clone() });
        let mut reasons: Vec<(String, usize)> = Vec::new();
        for split in &profile.splits {
            let text = split.to_string();
            let reason = text.split("): ").nth(1).unwrap_or("").trim_start_matches(|c: char| c.is_ascii_digit() || c == ' ');
            match reasons.iter_mut().find(|(r, _)| r == reason) {
                Some((_, count)) => *count += 1,
                None => reasons.push((reason.to_owned(), 1)),
            }
        }
        for (reason, count) in reasons {
            println!("  {count:>5} splits: {reason}");
        }
    }
    Ok(())
}
