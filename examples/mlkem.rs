//! ML-KEM (FIPS 203) KeyGen, Encaps and Decaps for ML-KEM-512, -768 and -1024 over a batch of
//! independent instances, written only with lazy array operations: no custom kernel, no WGSL.
//!
//! `cargo run --release --example mlkem -- [--n <batch sizes, comma separated>] [--set 512|768|1024]
//! [--op keygen|encaps|decaps] [--no-verify] [--no-cpu]`
//!
//! The results are checked byte for byte against the RustCrypto `ml-kem` crate on random instances
//! (a tampered ciphertext per instance included: implicit rejection) and against NIST ACVP vectors.
//! Then, per parameter set, operation and batch size, the table gives operations per second with
//! the inputs and outputs on the device (time from the call to `sync`) and with the upload of the
//! inputs and readback of the outputs, the GPU time of the kernels, the host time spent building and
//! recording the operation, launches per batch, the first call (kernel compilation: run with
//! `LUMA_CACHE=off` for a cold cache) and the same operation on all CPU threads with the `ml-kem`
//! crate. `MLKEM_PROFILE=1` lists the kernels of every row, `MLKEM_TRACE=1` splits the wall time into
//! graph building, evaluation and waiting.
//!
//! How it is written (`examples/mlkem/`):
//!
//! * Keccak-f[1600] is `keccak_f_loop`: one round traced, 24 of them run by `iterate_all` as one
//!   kernel with the 1600 state bits of a batch of sponges in registers; the round constants come from
//!   the Keccak LFSR carried as a 51st state word. Hash inputs are views of the byte-string arrays.
//! * A batch of byte strings (keys, ciphertexts, seeds) is a `[words, n]` array of `u32` words:
//!   the bytes of an instance are contiguous, as in memory, so uploads and readbacks need no reshuffle.
//! * A batch of polynomials is a `[256, rows]` array, one row per (instance, polynomial) pair. The
//!   NTT is two kernels of register butterflies without recomputation (four layers over sixteen
//!   slices of the polynomial, three over eight slices of every block of sixteen), each followed by
//!   a join of its outputs; base multiplication works on the pair of neighbours `kappa ^ 1`.
//! * SampleNTT is a fixed number of SHAKE128 blocks per matrix entry with the rejection done by
//!   a scan over the groups of eight candidates and one scatter (`put`); when some polynomial gets
//!   fewer than 256 accepted candidates (about 10^-30 per polynomial with 4 blocks) the operation is
//!   repeated with more blocks, so the result is right for every input.
//! * ByteEncode / ByteDecode, Compress / Decompress and the noise sampler are gathers (`take`) with
//!   shifts; Montgomery reduction with R = 2^16 does the products modulo q = 3329 in `u32`.
//!
//! Not implemented: the input checks of FIPS 203 (the modulus check of an encapsulation key, the
//! hash check of a decapsulation key); keys are trusted, and the code is not constant-time.

#[allow(dead_code)]
#[path = "keccak.rs"]
mod keccak;
#[path = "mlkem/codec.rs"]
mod codec;
#[path = "mlkem/field.rs"]
mod field;
#[path = "mlkem/hash.rs"]
mod hash;
#[path = "mlkem/host.rs"]
pub mod host;
#[path = "mlkem/kem.rs"]
pub mod kem;
#[path = "mlkem/params.rs"]
pub mod params;
#[path = "mlkem/sample.rs"]
mod sample;

use luma::*;

pub use kem::{Checked, Kem, Words};
use params::Params;

/// A `[len / 4, n]` array of the byte strings (all of length `len`, a multiple of 4): the bytes of
/// every string are contiguous, as in the host's memory.
pub fn upload(ctx: &Context, items: &[Vec<u8>]) -> Result<Words> {
    let (n, len) = (items.len(), items[0].len());
    Array::from_slice_on(ctx, &natural(items), [len / 4, n])
}

/// The byte strings of a `[words, n]` array.
pub fn download(words: &Words) -> Result<Vec<Vec<u8>>> {
    let [w, n, ..] = words.shape().dims();
    let data = words.to_vec()?;
    Ok((0..n).map(|i| data[i * w..(i + 1) * w].iter().flat_map(|x| x.to_le_bytes()).collect()).collect())
}

/// Runs `op` with the squeezed blocks of SampleNTT doubled until every polynomial got its 256
/// coefficients (with 4 blocks that is every polynomial but one in 10^30).
pub fn complete<T>(kem: &mut Kem, op: impl Fn(&Kem) -> Checked<T>, evaluate: impl Fn(&T) -> Result<()>) -> Result<T> {
    loop {
        let result = op(kem);
        evaluate(&result.value)?;
        if result.complete.item()? >= 256 {
            return Ok(result.value);
        }
        kem.blocks *= 2;
    }
}

fn check(what: &str, ok: bool) {
    println!("  {:<62} {}", what, if ok { "ok" } else { "MISMATCH" });
    assert!(ok, "{what} does not match");
}

fn random_items(seed: u64, n: usize, len: usize) -> Vec<Vec<u8>> {
    (0..n).map(|i| keccak::bytes(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(i as u64 * 7919 + 1), len)).collect()
}

/// Checks the three operations of `p` on `n` random instances against the RustCrypto implementation.
pub fn verify(ctx: &Context, p: Params, n: usize) -> Result<()> {
    let mut kem = Kem::new(ctx, p, n)?;
    let (d, z, m) = (random_items(1, n, 32), random_items(2, n, 32), random_items(3, n, 32));
    let host_keys: Vec<host::KeyPair> = (0..n).map(|i| host::keygen(p, d[i].as_slice().try_into().unwrap(), z[i].as_slice().try_into().unwrap())).collect();

    let (d_w, z_w, m_w) = (upload(ctx, &d)?, upload(ctx, &z)?, upload(ctx, &m)?);
    let (ek, dk) = complete(&mut kem, |kem| kem.keygen(&d_w, &z_w), |(ek, dk)| eval_all(&[ek, dk]))?;
    let (ek_b, dk_b) = (download(&ek)?, download(&dk)?);
    check(&format!("{} KeyGen: encapsulation keys of {n} instances", p.name), (0..n).all(|i| ek_b[i] == host_keys[i].ek));
    check(&format!("{} KeyGen: decapsulation keys of {n} instances", p.name), (0..n).all(|i| dk_b[i] == host_keys[i].dk));

    let (ct, key) = complete(&mut kem, |kem| kem.encaps(&ek, &m_w), |(ct, key)| eval_all(&[ct, key]))?;
    let (ct_b, key_b) = (download(&ct)?, download(&key)?);
    let host_enc: Vec<host::Encapsulation> = (0..n).map(|i| host::encaps(p, &host_keys[i].ek, m[i].as_slice().try_into().unwrap())).collect();
    check(&format!("{} Encaps: ciphertexts", p.name), (0..n).all(|i| ct_b[i] == host_enc[i].ct));
    check(&format!("{} Encaps: shared secrets", p.name), (0..n).all(|i| key_b[i] == host_enc[i].key));

    let key = complete(&mut kem, |kem| kem.decaps(&dk, &ct), |key| key.eval())?;
    let key_b = download(&key)?;
    check(&format!("{} Decaps: shared secrets recovered", p.name), (0..n).all(|i| key_b[i] == host_enc[i].key));

    // Implicit rejection: a flipped bit in the ciphertext gives the pseudo-random key of J(z, c).
    let tampered: Vec<Vec<u8>> = ct_b
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let mut c = c.clone();
            let at = (i * 37 + 5) % c.len();
            c[at] ^= 1 << (i % 8);
            c
        })
        .collect();
    let tampered_w = upload(ctx, &tampered)?;
    let key = complete(&mut kem, |kem| kem.decaps(&dk, &tampered_w), |key| key.eval())?;
    let key_b = download(&key)?;
    let host_rejected: Vec<[u8; 32]> = (0..n).map(|i| host::decaps(p, &host_keys[i].dk, &tampered[i])).collect();
    check(&format!("{} Decaps: implicit rejection of tampered ciphertexts", p.name), (0..n).all(|i| key_b[i] == host_rejected[i]));
    check(&format!("{} Decaps: rejected keys differ from the real ones", p.name), (0..n).all(|i| key_b[i] != host_enc[i].key));
    Ok(())
}

/// NIST ACVP-Server test vectors: KeyGen (`d`, `z` to `ek`, `dk`), Encaps (`ek`, `m` to `c`, `k`) and
/// Decaps (`dk`, `c` to `k`, a valid ciphertext and a modified one), a few per parameter set.
pub const ACVP: &str = include_str!("../tests/data/ml_kem_acvp.txt");

fn unhex(text: &str) -> Vec<u8> {
    (0..text.len() / 2).map(|i| u8::from_str_radix(&text[2 * i..2 * i + 2], 16).expect("hex digits")).collect()
}

/// Checks the operations against the ACVP vectors of `ACVP`.
pub fn verify_acvp(ctx: &Context) -> Result<()> {
    for p in params::ALL {
        // `kind` lines of this parameter set, fields decoded.
        let cases = |kind: &str| -> Vec<Vec<Vec<u8>>> {
            ACVP.lines()
                .filter(|line| !line.starts_with('#'))
                .map(|line| line.split_whitespace().collect::<Vec<_>>())
                .filter(|f| f.len() > 2 && f[0] == kind && f[1] == p.k.to_string())
                .map(|f| f[2..].iter().map(|x| unhex(x)).collect())
                .collect()
        };
        let column = |cases: &[Vec<Vec<u8>>], i: usize| -> Vec<Vec<u8>> { cases.iter().map(|c| c[i].clone()).collect() };
        let kem_for = |n: usize| Kem::new(ctx, p, n);

        let keygen = cases("keygen");
        let mut kem = kem_for(keygen.len())?;
        let (d, z) = (upload(ctx, &column(&keygen, 0))?, upload(ctx, &column(&keygen, 1))?);
        let (ek, dk) = complete(&mut kem, |kem| kem.keygen(&d, &z), |(ek, dk)| eval_all(&[ek, dk]))?;
        check(&format!("{} ACVP KeyGen ({} vectors)", p.name, keygen.len()), download(&ek)? == column(&keygen, 2) && download(&dk)? == column(&keygen, 3));

        let encaps = cases("encaps");
        let mut kem = kem_for(encaps.len())?;
        let (ek, m) = (upload(ctx, &column(&encaps, 0))?, upload(ctx, &column(&encaps, 1))?);
        let (ct, key) = complete(&mut kem, |kem| kem.encaps(&ek, &m), |(ct, key)| eval_all(&[ct, key]))?;
        check(&format!("{} ACVP Encaps ({} vectors)", p.name, encaps.len()), download(&ct)? == column(&encaps, 2) && download(&key)? == column(&encaps, 3));

        let decaps = cases("decaps");
        let mut kem = kem_for(decaps.len())?;
        let (dk, ct) = (upload(ctx, &column(&decaps, 0))?, upload(ctx, &column(&decaps, 1))?);
        let key = complete(&mut kem, |kem| kem.decaps(&dk, &ct), |key| key.eval())?;
        check(&format!("{} ACVP Decaps, valid and modified ciphertexts ({} vectors)", p.name, decaps.len()), download(&key)? == column(&decaps, 2));
    }
    Ok(())
}

// =================================================================================================
// Benchmark
// =================================================================================================

use std::time::{Duration, Instant};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Op {
    KeyGen,
    Encaps,
    Decaps,
}

impl Op {
    const ALL: [Op; 3] = [Op::KeyGen, Op::Encaps, Op::Decaps];
    fn name(self) -> &'static str {
        match self {
            Op::KeyGen => "KeyGen",
            Op::Encaps => "Encaps",
            Op::Decaps => "Decaps",
        }
    }
}

/// Natural layout (instance after instance) of `items` as `u32` words.
fn natural(items: &[Vec<u8>]) -> Vec<u32> {
    items.iter().flat_map(|item| item.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))).collect()
}

/// A `[words, n]` array from host data in natural layout.
fn from_natural(ctx: &Context, data: &[u32], n: usize) -> Result<Words> {
    Array::from_slice_on(ctx, data, [data.len() / n, n])
}

/// The natural-layout words of a `[words, n]` array, read back.
fn to_natural(words: &Words) -> Result<Vec<u32>> {
    words.to_vec()
}

fn best_of(runs: usize, mut f: impl FnMut() -> Result<Duration>) -> Result<Duration> {
    let mut best = Duration::MAX;
    for _ in 0..runs {
        best = best.min(f()?);
    }
    Ok(best)
}

struct Row {
    p: Params,
    op: Op,
    n: usize,
    /// Best wall time of the operation with its inputs and outputs on the device.
    resident: Duration,
    /// The host time of the same run until the call returned (graph building and recording).
    host: Duration,
    /// Best wall time including upload of the inputs and readback of the outputs.
    transfer: Duration,
    launches: u32,
    gpu: Duration,
    first_call: Duration,
    cpu: Option<f64>,
    splits: Vec<String>,
}

fn ops_per_second(n: usize, t: Duration) -> f64 {
    n as f64 / t.as_secs_f64()
}

struct Case<'a> {
    ctx: &'a Context,
    p: Params,
    n: usize,
    cpu: bool,
}

impl Case<'_> {
    fn run(&self, ops: &[Op]) -> Result<Vec<Row>> {
        let (ctx, p, n) = (self.ctx, self.p, self.n);
        let mut kem = Kem::new(ctx, p, n)?;
        let (d, z, m) = (random_items(11, n, 32), random_items(12, n, 32), random_items(13, n, 32));
        let (d_w, z_w, m_w) = (upload(ctx, &d)?, upload(ctx, &z)?, upload(ctx, &m)?);
        let (d_n, z_n, m_n) = (natural(&d), natural(&z), natural(&m));

        // Device-resident inputs of Encaps and Decaps, from the device's own KeyGen and Encaps.
        let (ek, dk) = complete(&mut kem, |kem| kem.keygen(&d_w, &z_w), |(ek, dk)| eval_all(&[ek, dk]))?;
        let (ct, _) = complete(&mut kem, |kem| kem.encaps(&ek, &m_w), |(ct, key)| eval_all(&[ct, key]))?;
        ctx.sync()?;
        let (ek_n, dk_n, ct_n) = (to_natural(&ek)?, to_natural(&dk)?, to_natural(&ct)?);

        // The CPU baseline works on the same bytes.
        let cpu_rate = |op: Op| -> Option<f64> {
            if !self.cpu {
                return None;
            }
            let threads = std::thread::available_parallelism().map_or(1, |t| t.get());
            let (ek_b, dk_b, ct_b) = (download(&ek).ok()?, download(&dk).ok()?, download(&ct).ok()?);
            let mut best = f64::INFINITY;
            for _ in 0..2 {
                let start = Instant::now();
                match op {
                    Op::KeyGen => drop(host::parallel(n, threads, |i| host::keygen(p, d[i].as_slice().try_into().unwrap(), z[i].as_slice().try_into().unwrap()))),
                    Op::Encaps => drop(host::parallel(n, threads, |i| host::encaps(p, &ek_b[i], m[i].as_slice().try_into().unwrap()))),
                    Op::Decaps => drop(host::parallel(n, threads, |i| host::decaps(p, &dk_b[i], &ct_b[i]))),
                }
                best = best.min(start.elapsed().as_secs_f64());
            }
            Some(n as f64 / best)
        };

        let mut rows = Vec::new();
        for &op in ops {
            // One run: build the operation, evaluate its outputs, wait for the device.
            let outputs = |kem: &Kem| -> Result<(Vec<Words>, Array<u32>)> {
                Ok(match op {
                    Op::KeyGen => {
                        let c = kem.keygen(&d_w, &z_w);
                        (vec![c.value.0, c.value.1], c.complete)
                    }
                    Op::Encaps => {
                        let c = kem.encaps(&ek, &m_w);
                        (vec![c.value.0, c.value.1], c.complete)
                    }
                    Op::Decaps => {
                        let c = kem.decaps(&dk, &ct);
                        (vec![c.value], c.complete)
                    }
                })
            };
            let resident = |kem: &Kem| -> Result<(Duration, Duration)> {
                let start = Instant::now();
                let (out, complete) = outputs(kem)?;
                let built = start.elapsed();
                eval_all(&out.iter().collect::<Vec<_>>())?;
                let host = start.elapsed();
                ctx.sync()?;
                let total = start.elapsed();
                if std::env::var_os("MLKEM_TRACE").is_some() {
                    println!("    build {:.2} ms, eval {:.2} ms, sync {:.2} ms", built.as_secs_f64() * 1e3, (host - built).as_secs_f64() * 1e3, (total - host).as_secs_f64() * 1e3);
                }
                assert!(complete.item()? >= 256, "SampleNTT needed more blocks");
                Ok((total, host))
            };
            // The first call in a context of its own: pipelines are compiled (or read from the disk
            // cache; `LUMA_CACHE=off` for a cold start).
            let first = {
                let fresh = Context::new()?;
                let fresh_kem = Kem::new(&fresh, p, n)?;
                let (d_w, z_w, m_w) = (from_natural(&fresh, &d_n, n)?, from_natural(&fresh, &z_n, n)?, from_natural(&fresh, &m_n, n)?);
                let (ek, dk, ct) = (from_natural(&fresh, &ek_n, n)?, from_natural(&fresh, &dk_n, n)?, from_natural(&fresh, &ct_n, n)?);
                fresh.sync()?;
                let start = Instant::now();
                let outs: Vec<Words> = match op {
                    Op::KeyGen => {
                        let c = fresh_kem.keygen(&d_w, &z_w);
                        vec![c.value.0, c.value.1]
                    }
                    Op::Encaps => {
                        let c = fresh_kem.encaps(&ek, &m_w);
                        vec![c.value.0, c.value.1]
                    }
                    Op::Decaps => vec![fresh_kem.decaps(&dk, &ct).value],
                };
                eval_all(&outs.iter().collect::<Vec<_>>())?;
                fresh.sync()?;
                start.elapsed()
            };
            let mut host = Duration::MAX;
            let resident_time = best_of(5, || {
                let (total, h) = resident(&kem)?;
                host = host.min(h);
                Ok(total)
            })?;
            let (_, profile) = ctx.profile(|| -> Result<()> {
                let (out, _) = outputs(&kem)?;
                eval_all(&out.iter().collect::<Vec<_>>())?;
                ctx.sync()
            })?;
            let transfer = best_of(3, || {
                let start = Instant::now();
                let outs: Vec<Words> = match op {
                    Op::KeyGen => {
                        let (d_w, z_w) = (from_natural(ctx, &d_n, n)?, from_natural(ctx, &z_n, n)?);
                        let c = kem.keygen(&d_w, &z_w);
                        vec![c.value.0, c.value.1]
                    }
                    Op::Encaps => {
                        let (ek, m_w) = (from_natural(ctx, &ek_n, n)?, from_natural(ctx, &m_n, n)?);
                        let c = kem.encaps(&ek, &m_w);
                        vec![c.value.0, c.value.1]
                    }
                    Op::Decaps => {
                        let (dk, ct) = (from_natural(ctx, &dk_n, n)?, from_natural(ctx, &ct_n, n)?);
                        vec![kem.decaps(&dk, &ct).value]
                    }
                };
                for out in &outs {
                    std::hint::black_box(to_natural(out)?);
                }
                Ok(start.elapsed())
            })?;
            if std::env::var_os("MLKEM_PROFILE").is_some() {
                println!("{} {} {}: {} launches, {:.2} ms of kernels", p.name, op.name(), n, profile.launches(), profile.total().as_secs_f64() * 1e3);
                for k in profile.kernels.iter().take(24) {
                    let label: String = k.label.chars().take(150).collect();
                    println!("  {:>5}x {:>9.1} us  {}", k.launches, k.total.as_secs_f64() * 1e6, label);
                }
                for s in &profile.splits {
                    println!("  {s}");
                }
            }
            rows.push(Row {
                p,
                op,
                n,
                resident: resident_time,
                host,
                transfer,
                launches: profile.launches(),
                gpu: profile.total(),
                first_call: first,
                cpu: cpu_rate(op),
                splits: profile.splits.iter().map(ToString::to_string).collect(),
            });
        }
        Ok(rows)
    }
}

fn print_row(r: &Row) {
    println!(
        "{:<12} {:<7} {:>6} {:>11.3} {:>9.2} {:>9.2} {:>12.3} {:>9.2} {:>8} {:>8.2} {:>9.0} {:>11}",
        r.p.name,
        r.op.name(),
        r.n,
        ops_per_second(r.n, r.resident) / 1e6,
        r.resident.as_secs_f64() * 1e3,
        r.gpu.as_secs_f64() * 1e3,
        ops_per_second(r.n, r.transfer) / 1e6,
        r.host.as_secs_f64() * 1e3,
        r.launches,
        r.first_call.as_secs_f64() * 1e3,
        r.splits.len(),
        r.cpu.map_or("-".to_owned(), |c| format!("{:.3}", c / 1e6)),
    );
}

fn print_header() {
    println!(
        "{:<12} {:<7} {:>6} {:>11} {:>9} {:>9} {:>12} {:>9} {:>8} {:>8} {:>9} {:>11}",
        "set", "op", "batch", "M ops/s", "ms", "gpu ms", "M ops/s +xfer", "host ms", "launches", "1st ms", "splits", "CPU M ops/s"
    );
    println!("{}", "-".repeat(124));
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let flag = |name: &str| args.iter().any(|a| a == name);
    let value = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let ctx = Context::new()?;
    if !flag("--no-verify") {
        println!("Correctness against the RustCrypto ml-kem crate and the NIST ACVP vectors");
        for p in params::ALL {
            verify(&ctx, p, 200)?;
        }
        verify_acvp(&ctx)?;
        println!();
    }
    let sizes: Vec<usize> = value("--n").map_or(vec![32768, 65536], |v| v.split(',').filter_map(|x| x.parse().ok()).collect());
    let sets: Vec<Params> = params::ALL.into_iter().filter(|p| value("--set").is_none_or(|s| p.name.ends_with(&s))).collect();
    let ops: Vec<Op> = Op::ALL.into_iter().filter(|o| value("--op").is_none_or(|s| o.name().eq_ignore_ascii_case(&s))).collect();
    print_header();
    let mut all = Vec::new();
    for &n in &sizes {
        for &p in &sets {
            let case = Case { ctx: &ctx, p, n, cpu: !flag("--no-cpu") };
            for row in case.run(&ops)? {
                print_row(&row);
                all.push(row);
            }
        }
    }
    Ok(())
}
