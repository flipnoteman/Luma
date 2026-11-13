//! The three parameter sets of FIPS 203 (Table 2) and the sizes derived from them.

/// One ML-KEM parameter set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Params {
    pub name: &'static str,
    /// Module rank: polynomials per vector.
    pub k: usize,
    pub eta1: u32,
    pub eta2: u32,
    /// Bits per coefficient of the compressed `u` and `v` of a ciphertext.
    pub du: u32,
    pub dv: u32,
}

pub const ML_KEM_512: Params = Params { name: "ML-KEM-512", k: 2, eta1: 3, eta2: 2, du: 10, dv: 4 };
pub const ML_KEM_768: Params = Params { name: "ML-KEM-768", k: 3, eta1: 2, eta2: 2, du: 10, dv: 4 };
pub const ML_KEM_1024: Params = Params { name: "ML-KEM-1024", k: 4, eta1: 2, eta2: 2, du: 11, dv: 5 };
pub const ALL: [Params; 3] = [ML_KEM_512, ML_KEM_768, ML_KEM_1024];

impl Params {
    /// Bytes of an encapsulation key: `384 k + 32`.
    pub const fn ek_bytes(&self) -> usize {
        384 * self.k + 32
    }
    /// Bytes of a decapsulation key: `768 k + 96`.
    pub const fn dk_bytes(&self) -> usize {
        768 * self.k + 96
    }
    /// Bytes of a ciphertext: `32 (du k + dv)`.
    pub const fn ct_bytes(&self) -> usize {
        32 * (self.du as usize * self.k + self.dv as usize)
    }
}
