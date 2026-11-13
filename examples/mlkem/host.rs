//! The RustCrypto `ml-kem` crate behind a byte-slice interface: the reference the lazy
//! implementation is checked against, and the CPU baseline of the benchmark.

use ml_kem::kem::{Decapsulate, KeyExport};
#[allow(deprecated)]
use ml_kem::{DecapsulationKey, ExpandedKeyEncoding, MlKem1024, MlKem512, MlKem768, Seed, B32};

use super::params::Params;

/// A key pair's encodings as FIPS 203 defines them.
pub struct KeyPair {
    pub ek: Vec<u8>,
    pub dk: Vec<u8>,
}

/// A ciphertext and the shared secret it carries.
pub struct Encapsulation {
    pub ct: Vec<u8>,
    pub key: [u8; 32],
}

macro_rules! per_set {
    ($p:expr, $kem:ident => $body:expr) => {
        match $p.k {
            2 => {
                type $kem = MlKem512;
                $body
            }
            3 => {
                type $kem = MlKem768;
                $body
            }
            4 => {
                type $kem = MlKem1024;
                $body
            }
            k => panic!("no ML-KEM with k = {k}"),
        }
    };
}

/// `ML-KEM.KeyGen_internal(d, z)`.
#[allow(deprecated)]
pub fn keygen(p: Params, d: &[u8; 32], z: &[u8; 32]) -> KeyPair {
    per_set!(p, Kem => {
        let mut seed = Seed::default();
        seed[..32].copy_from_slice(d);
        seed[32..].copy_from_slice(z);
        let dk = DecapsulationKey::<Kem>::from_seed(seed);
        KeyPair { ek: dk.encapsulation_key().to_bytes().to_vec(), dk: dk.to_expanded_bytes().to_vec() }
    })
}

/// `ML-KEM.Encaps_internal(ek, m)`.
pub fn encaps(p: Params, ek: &[u8], m: &[u8; 32]) -> Encapsulation {
    per_set!(p, Kem => {
        let ek = ml_kem::EncapsulationKey::<Kem>::new(ek.try_into().expect("encapsulation key size")).expect("valid encapsulation key");
        let mut message = B32::default();
        message.copy_from_slice(m);
        let (ct, key) = ek.encapsulate_deterministic(&message);
        Encapsulation { ct: ct.to_vec(), key: key.into() }
    })
}

/// `ML-KEM.Decaps(dk, c)` (implicit rejection included).
#[allow(deprecated)]
pub fn decaps(p: Params, dk: &[u8], ct: &[u8]) -> [u8; 32] {
    per_set!(p, Kem => {
        let dk = DecapsulationKey::<Kem>::from_expanded_bytes(dk.try_into().expect("decapsulation key size")).expect("valid decapsulation key");
        dk.decapsulate_slice(ct).expect("ciphertext size").into()
    })
}

/// `f(i)` for `i` in `0..n` on `threads` threads, in order.
pub fn parallel<T: Send>(n: usize, threads: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let per = n.div_ceil(threads.max(1)).max(1);
    let mut out: Vec<Option<T>> = (0..n).map(|_| None).collect();
    std::thread::scope(|scope| {
        for (t, chunk) in out.chunks_mut(per).enumerate() {
            let f = &f;
            scope.spawn(move || {
                for (j, slot) in chunk.iter_mut().enumerate() {
                    *slot = Some(f(t * per + j));
                }
            });
        }
    });
    out.into_iter().map(|v| v.expect("every index computed")).collect()
}
