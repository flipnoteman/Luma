//! Integer-heavy workloads written only with lazy `u32` array operations: batched SHA3-256 (the
//! 24 Keccak rounds unrolled in the graph, and as one `iterate_all` register loop) and ML-KEM-768
//! (KeyGen, Encaps, Decaps). The implementations are the repository's `examples/keccak.rs` and
//! `examples/mlkem/`, included as modules, so the benchmark follows them. ML-KEM is verified
//! against the NIST ACVP vectors of `tests/data` and by a KeyGen -> Encaps -> Decaps round trip;
//! the hashes against the host sponge.

#[allow(dead_code, unused, clippy::all)]
#[path = "../../../../../examples/keccak.rs"]
mod keccak;
#[allow(dead_code, unused, clippy::all)]
#[path = "../../../../../examples/mlkem/codec.rs"]
mod codec;
#[allow(dead_code, unused, clippy::all)]
#[path = "../../../../../examples/mlkem/field.rs"]
mod field;
#[allow(dead_code, unused, clippy::all)]
#[path = "../../../../../examples/mlkem/hash.rs"]
mod hash;
#[allow(dead_code, unused, clippy::all)]
#[path = "../../../../../examples/mlkem/kem.rs"]
mod kem;
#[allow(dead_code, unused, clippy::all)]
#[path = "../../../../../examples/mlkem/params.rs"]
mod params;
#[allow(dead_code, unused, clippy::all)]
#[path = "../../../../../examples/mlkem/sample.rs"]
mod sample;

use luma::*;

use super::pick;
use crate::harness::*;
use kem::{Checked, Kem, Words};

const ACVP: &str = include_str!("../../../../../tests/data/ml_kem_acvp.txt");

pub fn run(h: &mut Harness) {
    sha3(h);
    mlkem(h);
}

fn sha3(h: &mut Harness) {
    h.guarded("keccak", |h| {
        let ctx = h.ctx.clone();
        let len = 64;
        let sizes: Vec<usize> = if h.ctx.is_cpu() { vec![1 << 12, pick(h, 1 << 12, 1 << 14)] } else { vec![1 << 12, pick(h, 1 << 14, 1 << 16)] };
        let mut sizes = sizes;
        sizes.dedup();
        for n in sizes {
            let messages: Vec<Vec<u8>> = (0..n).map(|i| keccak::bytes(0x5eed + i as u64, len)).collect();
            let words = keccak::message_words(&ctx, &messages, len).unwrap();
            eval_all(&words.iter().collect::<Vec<_>>()).unwrap();
            let verify = |out: Vec<Array<u32>>, what: &str| {
                let out = keccak::read_all(&out).unwrap();
                for i in (0..n).step_by((n / 64).max(1)) {
                    let want = keccak::sponge_host(&messages[i], 136, 0x06, 32);
                    assert_eq!(keccak::instance_bytes(&out, i, 32), want, "{what}: SHA3-256 of instance {i}");
                }
            };
            verify(keccak::sponge(&ctx, n, &words, len, 136, 0x06, 32).unwrap(), "unrolled");
            verify(hash::sha3_256(&ctx, n, &words, len).unwrap(), "loop");
            let size = format!("{n} x {len}B");
            h.case("keccak/sha3_unrolled", &size, Work::Items(n as f64, "Mhash/s"), &mut || {
                let out = keccak::sponge(&ctx, n, &words, len, 136, 0x06, 32).unwrap();
                eval_all(&out.iter().collect::<Vec<_>>()).unwrap();
            });
            h.case("keccak/sha3_loop", &size, Work::Items(n as f64, "Mhash/s"), &mut || {
                let out = hash::sha3_256(&ctx, n, &words, len).unwrap();
                eval_all(&out.iter().collect::<Vec<_>>()).unwrap();
            });
        }
    });
}

/// Natural layout (instance after instance) of `items` as `u32` words.
fn natural(items: &[Vec<u8>]) -> Vec<u32> {
    items.iter().flat_map(|item| item.chunks(4).map(|c| u32::from_le_bytes(c.try_into().expect("strings of whole words")))).collect()
}

/// A `[len / 4, n]` array of byte strings of one length.
fn upload(ctx: &Context, items: &[Vec<u8>]) -> Words {
    let (n, len) = (items.len(), items[0].len());
    Array::from_slice_on(ctx, &natural(items), [len / 4, n]).unwrap()
}

fn download(words: &Words) -> Vec<Vec<u8>> {
    let [w, n, ..] = words.shape().dims();
    let data = words.to_vec().unwrap();
    (0..n).map(|i| data[i * w..(i + 1) * w].iter().flat_map(|x| x.to_le_bytes()).collect()).collect()
}

/// Runs `op` with the SampleNTT blocks doubled until every polynomial got its 256 coefficients.
fn complete<T>(kem: &mut Kem, op: impl Fn(&Kem) -> Checked<T>, evaluate: impl Fn(&T) -> Result<()>) -> T {
    loop {
        let result = op(kem);
        evaluate(&result.value).unwrap();
        if result.complete.item().unwrap() >= 256 {
            return result.value;
        }
        kem.blocks *= 2;
    }
}

fn unhex(text: &str) -> Vec<u8> {
    (0..text.len() / 2).map(|i| u8::from_str_radix(&text[2 * i..2 * i + 2], 16).expect("hex digits")).collect()
}

/// The three operations of `p` against the NIST ACVP vectors of `tests/data`.
fn verify_acvp(ctx: &Context, p: params::Params) {
    let cases = |kind: &str| -> Vec<Vec<Vec<u8>>> {
        ACVP.lines()
            .filter(|line| !line.starts_with('#'))
            .map(|line| line.split_whitespace().collect::<Vec<_>>())
            .filter(|f| f.len() > 2 && f[0] == kind && f[1] == p.k.to_string())
            .map(|f| f[2..].iter().map(|x| unhex(x)).collect())
            .collect()
    };
    let column = |cases: &[Vec<Vec<u8>>], i: usize| -> Vec<Vec<u8>> { cases.iter().map(|c| c[i].clone()).collect() };

    let keygen = cases("keygen");
    let mut kem = Kem::new(ctx, p, keygen.len()).unwrap();
    let (d, z) = (upload(ctx, &column(&keygen, 0)), upload(ctx, &column(&keygen, 1)));
    let (ek, dk) = complete(&mut kem, |kem| kem.keygen(&d, &z), |(ek, dk)| eval_all(&[ek, dk]));
    assert!(download(&ek) == column(&keygen, 2) && download(&dk) == column(&keygen, 3), "{} ACVP KeyGen", p.name);

    let encaps = cases("encaps");
    let mut kem = Kem::new(ctx, p, encaps.len()).unwrap();
    let (ek, m) = (upload(ctx, &column(&encaps, 0)), upload(ctx, &column(&encaps, 1)));
    let (ct, key) = complete(&mut kem, |kem| kem.encaps(&ek, &m), |(ct, key)| eval_all(&[ct, key]));
    assert!(download(&ct) == column(&encaps, 2) && download(&key) == column(&encaps, 3), "{} ACVP Encaps", p.name);

    let decaps = cases("decaps");
    let mut kem = Kem::new(ctx, p, decaps.len()).unwrap();
    let (dk, ct) = (upload(ctx, &column(&decaps, 0)), upload(ctx, &column(&decaps, 1)));
    let key = complete(&mut kem, |kem| kem.decaps(&dk, &ct), |key| key.eval());
    assert!(download(&key) == column(&decaps, 2), "{} ACVP Decaps", p.name);
}

fn mlkem(h: &mut Harness) {
    h.guarded("mlkem", |h| {
        let ctx = h.ctx.clone();
        let p = params::ML_KEM_768;
        verify_acvp(&ctx, p);

        let n = if h.ctx.is_cpu() { pick(h, 16, 64) } else { pick(h, 128, 512) };
        let random = |seed: u64| -> Vec<Vec<u8>> { (0..n).map(|i| keccak::bytes(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(i as u64 * 7919 + 1), 32)).collect() };
        let (d, z, m) = (upload(&ctx, &random(1)), upload(&ctx, &random(2)), upload(&ctx, &random(3)));
        let mut kem = Kem::new(&ctx, p, n).unwrap();
        // The round trip: Decaps(Encaps(ek)) recovers the key Encaps returned.
        let (ek, dk) = complete(&mut kem, |kem| kem.keygen(&d, &z), |(ek, dk)| eval_all(&[ek, dk]));
        let (ct, key) = complete(&mut kem, |kem| kem.encaps(&ek, &m), |(ct, key)| eval_all(&[ct, key]));
        let again = complete(&mut kem, |kem| kem.decaps(&dk, &ct), |k| k.eval());
        assert!(download(&again) == download(&key), "ML-KEM-768 round trip: decapsulated keys differ from the encapsulated ones");
        let size = format!("{} batch {n}", p.name);
        // `Items` reports millions per second: scale the count to read thousands of operations per second.
        let items = Work::Items(n as f64 * 1e3, "kop/s");
        h.case("mlkem/keygen", &size, items, &mut || {
            complete(&mut kem, |kem| kem.keygen(&d, &z), |(ek, dk)| eval_all(&[ek, dk]));
        });
        h.case("mlkem/encaps", &size, items, &mut || {
            complete(&mut kem, |kem| kem.encaps(&ek, &m), |(ct, key)| eval_all(&[ct, key]));
        });
        h.case("mlkem/decaps", &size, items, &mut || {
            complete(&mut kem, |kem| kem.decaps(&dk, &ct), |k| k.eval());
        });
    });
}
