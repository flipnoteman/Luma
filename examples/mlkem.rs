//! ML-KEM (FIPS 203) in lazy array operations.

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
mod host;
#[path = "mlkem/kem.rs"]
mod kem;
#[path = "mlkem/params.rs"]
mod params;
#[path = "mlkem/sample.rs"]
mod sample;

use luma::*;

use kem::{Checked, Kem, Words};
use params::Params;

/// A `[n, len / 4]` array of the byte strings (all of length `len`, a multiple of 4).
pub fn upload(ctx: &Context, items: &[Vec<u8>]) -> Result<Words> {
    let (n, len) = (items.len(), items[0].len());
    let mut data = vec![0u32; n * len / 4];
    for (i, item) in items.iter().enumerate() {
        for (w, chunk) in item.chunks_exact(4).enumerate() {
            data[i + n * w] = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
    }
    Array::from_slice_on(ctx, &data, [n, len / 4])
}

/// The byte strings of a `[n, words]` array.
pub fn download(words: &Words) -> Result<Vec<Vec<u8>>> {
    let [n, w, ..] = words.shape().dims();
    let data = words.to_vec()?;
    Ok((0..n).map(|i| (0..w).flat_map(|x| data[i + n * x].to_le_bytes()).collect()).collect())
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
    let mut kem = Kem::new(ctx, p, n, 2)?;
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

fn main() -> Result<()> {
    let ctx = Context::new()?;
    for p in params::ALL {
        verify(&ctx, p, 200)?;
    }
    Ok(())
}
