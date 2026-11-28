//! Reference verification of matrix-unit kernel variants (W3-R7 §5): the first use of a variant of a [`GemmSpec`] kernel on a device
//! runs it on a synthetic problem that exercises the edges and compares it with an `f64` reference over the products of the
//! operands as the unit reads them (rounded as the unit rounds). The verdict is kept in memory and in the device's [`DiskCache`], keyed
//! by the variant and the revision of the backend's kernel generator, so each (device, variant, revision) is verified once; a variant
//! with a `Bad` verdict is never selected. Xla#51016 is the reason: a tile or orientation the hardware does not support can run
//! silently wrong.
//!
//! The comparison is a bound, never bit equality: a tensor core accumulates in hardware order and width, not as an `fma` chain.

use std::collections::HashMap;
use std::mem::MaybeUninit;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use super::{Backend, BindingInfo, Buffer, BindingKind, BufferUsage, CompiledKernel, Dispatch, GemmLanes, GemmOut, GemmSpec, GemmUnit, RawArg};

/// Magic of a stored verdict.
const UNIT_MAGIC: [u8; 8] = *b"LUMAUNV1";

/// Slice of the K range of the synthetic product (two `bk` of 32): the split-K path is part of every verification.
const KC: usize = 64;

/// Relative error bound of one output against the sum of the magnitudes of its products, for a unit that multiplies exactly and
/// accumulates in `f32`. The spec's `4 * 2^-24` (W3-R7 §5) is met by the typical output, but an output that sums a `1e8` product
/// among 64 others reaches `4.3 * 2^-24` on the exact `f32` chain (measured on an RTX 3090): the bound is `16 * 2^-24`, still
/// six orders of magnitude under what a wrong fragment layout gives (errors of the size of the sum itself).
const ACC_BOUND: f64 = 16.0 / 16_777_216.0;

/// Relative bound for TF32, which rounds each operand to 10 mantissa bits (the kernel's rounding is within one unit in the last place
/// of the RNE rounding): two operands at `2^-10` each, with room.
const TF32_BOUND: f64 = 2.5e-3;

/// The verdict on one variant.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UnitVerdict {
    /// Hash of the variant: its spec (tile, orientations, unit, lanes, output) and the kernel generator's revision.
    pub variant: u64,
    /// Whether the kernel matched the reference.
    pub ok: bool,
    /// The largest error of an output relative to the sum of the magnitudes of its products.
    pub max_rel_err: f32,
}

/// The spec with what the verdict does not depend on cleared: a half-work kernel and a hooked kernel run the plain product's
/// code between their loads and their stores, so one verdict covers them.
fn base(spec: &GemmSpec) -> GemmSpec {
    GemmSpec { half: false, hook: None, ..*spec }
}

/// The key of a variant: its spec and the `revision` of the kernel generator (see [`Backend::kernel_revision`]).
pub fn variant_key(spec: &GemmSpec, revision: u64) -> u64 {
    super::cache::source_revision(&[&format!("{:?}", base(spec))]) ^ revision.rotate_left(29)
}

/// A half type's conversions: from `f32` to bits (round to nearest even) and back.
type Codec = (fn(f32) -> u16, fn(u16) -> f32);

/// The operands as a unit reads them.
#[derive(Clone, Copy)]
struct Format {
    /// Rounds one operand as the unit does (the reference multiplies the rounded values).
    round: fn(f32) -> f32,
    /// The operand's type in the buffers: 4 bytes (`f32`) or 2 (a half type, `to_bits` / `from_bits` of its encoding).
    half: Option<Codec>,
    /// A subnormal of the operand type (the problem sprinkles it in).
    tiny: f32,
    /// The bound of an output's error relative to the sum of the magnitudes of its products.
    bound: f64,
    /// Magnitude of the largest operands of the synthetic problem.
    big: f32,
}

fn exact(x: f32) -> f32 {
    x
}

/// How the unit `unit` reads its operands, `None` for a unit this verifier does not know.
fn format(unit: GemmUnit) -> Option<Format> {
    match unit {
        GemmUnit::Fma => Some(Format { round: exact, half: None, tiny: 1.0e-40, bound: ACC_BOUND, big: 1.0e4 }),
        // The reference keeps the operands unrounded: the bound covers the rounding.
        GemmUnit::Tf32 => Some(Format { round: exact, half: None, tiny: 1.0e-40, bound: TF32_BOUND, big: 1.0e4 }),
        // The products of two halves are exact in `f32`: the same bound as the exact chain. A `big` of 100 (f16) keeps the products and
        // their sums finite in `f32` (and mostly under the `f16` range of the output).
        GemmUnit::F16 | GemmUnit::CoopF16 => Some(Format { round: |x| f16_to_f32(f32_to_f16(x)), half: Some((f32_to_f16, f16_to_f32)), tiny: 1.0e-6, bound: ACC_BOUND, big: 100.0 }),
        GemmUnit::Bf16 => Some(Format { round: |x| bf16_to_f32(f32_to_bf16(x)), half: Some((f32_to_bf16, bf16_to_f32)), tiny: 1.0e-40, bound: ACC_BOUND, big: 3.0e4 }),
        #[allow(unreachable_patterns, reason = "GemmUnit is non_exhaustive: units of other crates' backends")]
        _ => None,
    }
}


/// `x` as `bf16` bits, to nearest even, a NaN as the canonical quiet NaN.
pub fn f32_to_bf16(x: f32) -> u16 {
    if x.is_nan() {
        return ((x.to_bits() >> 16) as u16 & 0x8000) | 0x7FC0;
    }
    let bits = x.to_bits();
    (bits.wrapping_add(0x7FFF + ((bits >> 16) & 1)) >> 16) as u16
}

/// The value of `bf16` bits.
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// `x` as `f16` bits, to nearest even (overflow gives an infinity), a NaN as the canonical quiet NaN.
pub fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let man = bits & 0x7F_FFFF;
    if exp == 0xFF {
        return sign | 0x7C00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 112;
    if e >= 31 {
        return sign | 0x7C00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let (mut h, rem, half) = (m >> shift, m & ((1 << shift) - 1), 1u32 << (shift - 1));
        if rem > half || (rem == half && h & 1 == 1) {
            h += 1;
        }
        return sign | h as u16;
    }
    let mut h = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1FFF;
    if rem > 0x1000 || (rem == 0x1000 && h & 1 == 1) {
        h += 1;
    }
    sign | h as u16
}

/// The value of `f16` bits.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let exp = i32::from((bits >> 10) & 0x1F);
    let man = f32::from(bits & 0x3FF);
    match exp {
        0 => sign * man * 2f32.powi(-24),
        31 => if man == 0.0 { sign * f32::INFINITY } else { f32::NAN },
        _ => sign * (1.0 + man / 1024.0) * 2f32.powi(exp - 15),
    }
}

/// A row-major matrix of `rows x cols` in a buffer of `ld` columns from element `off`, NaN everywhere the matrix is not.
struct Stored {
    data: Vec<f32>,
    ld: usize,
    off: usize,
}

impl Stored {
    fn at(&self, r: usize, c: usize) -> f32 {
        self.data[self.off + r * self.ld + c]
    }
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

/// The synthetic problem of a variant: `(m, n, k) = (rows + 3, cols + 5, 8 bk + 5)` (the 16-byte lanes need multiples of 4: `+4`, `+8`,
/// `+4`), `bk = 32`, the K range cut into slices of 64, seeded operands with `+-0`, subnormals and large magnitudes, one `NaN` in the
/// left operand and one infinity in the right.
pub struct Problem {
    spec: GemmSpec,
    format: Format,
    dims: (usize, usize, usize),
    a: Stored,
    b: Stored,
}

impl Problem {
    /// The problem of `spec`, `None` if the verifier does not know its unit.
    pub fn new(spec: &GemmSpec) -> Option<Problem> {
        let spec = base(spec);
        let format = format(spec.unit)?;
        let aligned = spec.lanes == GemmLanes::Vec4;
        // The 16-byte lanes need multiples of 4 elements (`f32`) or 8 (halves).
        let word = if format.half.is_some() { 8 } else { 4 };
        let (m, n, k) = if aligned { (spec.rows as usize + word, spec.cols as usize + 8, 8 * 32 + word) } else { (spec.rows as usize + 3, spec.cols as usize + 5, 8 * 32 + 5) };
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ variant_key(&spec, 0));
        let up = |x: usize| x.next_multiple_of(word);
        // The scalar lanes take any leading dimension and offset; the 16-byte lanes need multiples of 4.
        let layout = |cols: usize, odd: usize| if spec.lanes == GemmLanes::Scalar { (cols + odd, odd + 2) } else { (up(cols) + word, 8) };
        let (a_rows, a_cols) = if spec.lhs_transposed { (k, m) } else { (m, k) };
        let (b_rows, b_cols) = if spec.rhs_transposed { (n, k) } else { (k, n) };
        let mut fill = |rows: usize, cols: usize, odd: usize| {
            let (ld, off) = layout(cols, odd);
            let mut data = vec![f32::NAN; off + rows * ld + 8];
            for r in 0..rows {
                for c in 0..cols {
                    let at = r * cols + c;
                    data[off + r * ld + c] = match at {
                        _ if at % 17 == 3 => 0.0,
                        _ if at % 19 == 5 => -0.0,
                        _ if at % 29 == 7 => format.tiny,
                        _ if at % 31 == 11 => rng.next() * format.big,
                        _ => rng.next(),
                    };
                }
            }
            Stored { data, ld, off }
        };
        let mut a = fill(a_rows, a_cols, 3);
        let mut b = fill(b_rows, b_cols, 1);
        // One NaN at logical (1, 7) of the left operand, one infinity at logical (9, 2) of the right.
        let (ar, ac) = if spec.lhs_transposed { (7, 1) } else { (1, 7) };
        a.data[a.off + ar * a.ld + ac] = f32::NAN;
        let (br, bc) = if spec.rhs_transposed { (2, 9) } else { (9, 2) };
        b.data[b.off + br * b.ld + bc] = f32::INFINITY;
        Some(Problem { spec, format, dims: (m, n, k), a, b })
    }

    fn lhs(&self, i: usize, q: usize) -> f32 {
        if self.spec.lhs_transposed { self.a.at(q, i) } else { self.a.at(i, q) }
    }

    fn rhs(&self, q: usize, j: usize) -> f32 {
        if self.spec.rhs_transposed { self.b.at(j, q) } else { self.b.at(q, j) }
    }

    /// The number of K slices of the launch.
    fn slices(&self) -> usize {
        if self.spec.out == GemmOut::F32 { self.dims.2.div_ceil(KC) } else { 1 }
    }

    /// Elements of K per slice.
    fn kc(&self) -> usize {
        if self.spec.out == GemmOut::F32 { KC } else { self.dims.2 }
    }

    /// Compares `got`, the slices of the product one after the other (`slices * m * n` values), with the reference. Returns whether it
    /// is within the bound and the largest relative error. A non-finite reference output only has its class compared (`NaN` where the
    /// reference has `NaN`, non-finite where it is infinite).
    pub fn judge(&self, got: &[f32]) -> (bool, f32) {
        let (m, n, k) = self.dims;
        if got.len() != self.slices() * m * n {
            return (false, f32::INFINITY);
        }
        let (mut ok, mut worst) = (true, 0f64);
        for i in 0..m {
            for j in 0..n {
                let (mut want, mut mag) = (0f64, 0f64);
                for q in 0..k {
                    let product = f64::from((self.format.round)(self.lhs(i, q))) * f64::from((self.format.round)(self.rhs(q, j)));
                    want += product;
                    mag += product.abs();
                }
                let value: f64 = (0..self.slices()).map(|z| f64::from(got[z * m * n + i * n + j])).sum();
                // A half output adds its own rounding (`2^-11` of the value for `f16`, `2^-8` for `bf16`, an absolute `2^-25` where `f16`
                // is subnormal) and overflows to an infinity.
                let (out_eps, out_abs, overflow) = match self.spec.out {
                    GemmOut::F16 => (4.8828125e-4, 3.0e-8, 65520.0),
                    GemmOut::Bf16 => (3.90625e-3, 0.0, f64::INFINITY),
                    _ => (0.0, 0.0, f64::INFINITY),
                };
                if !want.is_finite() {
                    ok &= value.is_nan() == want.is_nan() && !value.is_finite();
                } else if !value.is_finite() {
                    // Only an output past the range of its type may be an infinity.
                    ok &= want.abs() >= overflow * 0.999 && value.is_infinite() && value.signum() == want.signum();
                } else {
                    let error = (value - want).abs();
                    let relative = error / mag.max(1e-30);
                    worst = worst.max(relative);
                    ok &= error <= self.format.bound * mag + out_eps * want.abs() * 1.01 + out_abs + 1e-30;
                }
            }
        }
        (ok, worst as f32)
    }

    /// Runs the kernel of the spec on `backend` and returns the slices of the product, or why it could not.
    pub fn run(&self, backend: &dyn Backend) -> Result<Vec<f32>, String> {
        let (m, n, k) = self.dims;
        let spec = &self.spec;
        let raw = backend.compile_gemm(spec, None).ok_or("the backend has no such kernel")?;
        let start = Instant::now();
        while !raw.is_ready() {
            if start.elapsed() > Duration::from_secs(60) {
                return Err("the kernel never finished compiling".into());
            }
            std::thread::sleep(Duration::from_micros(200));
        }
        let binding = |name: &str, kind| BindingInfo { name: name.into(), kind, element_stride: None, min_size: None };
        let bindings = vec![
            binding("c", BindingKind::StorageReadWrite),
            binding("a", BindingKind::StorageRead),
            binding("b", BindingKind::StorageRead),
            BindingInfo { min_size: Some(32), ..binding("u", BindingKind::Uniform) },
        ];
        let kernel = CompiledKernel::new("unit-verify".into(), raw, bindings, 4, None, [1, 1, 1]);
        let half = self.format.half;
        let bytes = |data: &[f32]| -> Vec<u8> {
            match half {
                Some((encode, _)) => data.iter().flat_map(|&x| encode(x).to_le_bytes()).collect(),
                None => data.iter().flat_map(|x| x.to_le_bytes()).collect(),
            }
        };
        let max = u64::MAX;
        // Mappable storage where the device shares memory with the host, else a device buffer written by an upload.
        let fill = |data: &[f32]| -> Result<Buffer, String> {
            let data = bytes(data);
            if backend.info().zero_copy {
                return backend.alloc_filled(&data, max).map_err(|e| e.to_string());
            }
            let buffer = backend.create_buffer(data.len().next_multiple_of(4) as u64, BufferUsage::STORAGE | BufferUsage::COPY_DST, false).ok_or("no memory for an operand")?;
            backend.record_upload(&buffer, &data, max).map_err(|e| e.to_string())?;
            Ok(buffer)
        };
        let a_buf = fill(&self.a.data)?;
        let b_buf = fill(&self.b.data)?;
        let slices = self.slices();
        let out_bytes = match spec.out {
            GemmOut::F32 => 4,
            _ => 2,
        };
        let c_buf = backend.create_buffer((slices * m * n * out_bytes) as u64, BufferUsage::STORAGE | BufferUsage::COPY_SRC, false).ok_or("no memory for the result")?;
        let mut uniform = Vec::new();
        for word in [m, n, k, self.kc(), self.a.ld, self.b.ld, self.a.off, self.b.off] {
            uniform.extend_from_slice(&(word as u32).to_le_bytes());
        }
        let grid = [n.div_ceil(spec.cols as usize) as u32, m.div_ceil(spec.rows as usize) as u32, slices as u32];
        let args = [RawArg::write(&c_buf, c_buf.size()), RawArg::read(&a_buf, a_buf.size()), RawArg::read(&b_buf, b_buf.size()), RawArg::Uniform(&uniform)];
        backend.record_launch(&kernel, Dispatch::Direct(grid), [grid[0] * spec.cols, grid[1] * spec.rows, grid[2]], &args).map_err(|e| e.to_string())?;
        backend.sync(Duration::ZERO).map_err(|e| e.to_string())?;
        let mut out = vec![MaybeUninit::<u8>::uninit(); slices * m * n * out_bytes];
        backend.read_into(&c_buf, &mut out, false, Duration::ZERO).map_err(|e| e.to_string())?;
        // SAFETY: `read_into` filled every byte of `out`.
        let out: Vec<u8> = out.into_iter().map(|byte| unsafe { byte.assume_init() }).collect();
        Ok(match spec.out {
            GemmOut::F32 => out.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect(),
            GemmOut::F16 => out.as_chunks::<2>().0.iter().map(|c| f16_to_f32(u16::from_le_bytes(*c))).collect(),
            _ => out.as_chunks::<2>().0.iter().map(|c| bf16_to_f32(u16::from_le_bytes(*c))).collect(),
        })
    }
}

/// Alters the output of a verification run (test hook: a kernel with a flipped fragment index).
pub type Tamper = Arc<dyn Fn(&GemmSpec, &mut [f32]) + Send + Sync>;

/// The verdicts of a context: in memory, backed by the device's disk cache.
#[derive(Default)]
pub struct UnitVerifier {
    cells: Mutex<HashMap<u64, Arc<OnceLock<UnitVerdict>>>>,
    tamper: Mutex<Option<Tamper>>,
}

impl std::fmt::Debug for UnitVerifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UnitVerifier").finish_non_exhaustive()
    }
}

fn stem(disk: &super::DiskCache, variant: u64) -> String {
    format!("unit-{}-{variant:016x}", disk.key())
}

fn load(disk: &super::DiskCache, variant: u64) -> Option<UnitVerdict> {
    let bytes = disk.load_blob(&stem(disk, variant), UNIT_MAGIC)?;
    let [ok, a, b, c, d] = bytes[..] else { return None };
    (ok <= 1).then(|| UnitVerdict { variant, ok: ok == 1, max_rel_err: f32::from_le_bytes([a, b, c, d]) })
}

fn store(disk: &super::DiskCache, verdict: &UnitVerdict) {
    let mut payload = vec![u8::from(verdict.ok)];
    payload.extend_from_slice(&verdict.max_rel_err.to_le_bytes());
    disk.store_blob(&stem(disk, verdict.variant), UNIT_MAGIC, &payload);
}

impl UnitVerifier {
    pub fn new() -> UnitVerifier {
        UnitVerifier::default()
    }

    /// Installs `tamper`, applied to the output of every verification run from now on (tests only).
    #[doc(hidden)]
    pub fn set_tamper(&self, tamper: Option<Tamper>) {
        *self.tamper.lock().unwrap_or_else(|e| e.into_inner()) = tamper;
    }

    fn cell(&self, variant: u64) -> Arc<OnceLock<UnitVerdict>> {
        Arc::clone(self.cells.lock().unwrap_or_else(|e| e.into_inner()).entry(variant).or_default())
    }

    /// The verdict on `spec`'s variant if it is known (in memory or on disk), without running anything.
    pub fn known(&self, backend: &dyn Backend, spec: &GemmSpec) -> Option<UnitVerdict> {
        let variant = variant_key(spec, backend.kernel_revision());
        let cell = self.cell(variant);
        if let Some(verdict) = cell.get() {
            return Some(*verdict);
        }
        let verdict = load(backend.disk_cache(), variant)?;
        Some(*cell.get_or_init(|| verdict))
    }

    /// The verdict on `spec`'s variant, running the verification if there is none yet (a concurrent caller of the same variant waits for
    /// it). With the time it took when this call ran it, `None` when the verdict was known. A variant that cannot be verified (its
    /// kernel does not compile or launch, or its unit is unknown) is `Bad` for the life of this verifier and is not stored.
    pub fn verdict(&self, backend: &dyn Backend, spec: &GemmSpec) -> (UnitVerdict, Option<Duration>) {
        let variant = variant_key(spec, backend.kernel_revision());
        let cell = self.cell(variant);
        let mut waited = None;
        let verdict = *cell.get_or_init(|| {
            let disk = backend.disk_cache();
            if let Some(verdict) = load(disk, variant) {
                return verdict;
            }
            let start = Instant::now();
            let verdict = self.run(backend, spec, variant);
            waited = Some(start.elapsed());
            if let Some((verdict, stable)) = verdict {
                if stable {
                    store(disk, &verdict);
                }
                if !verdict.ok {
                    log::warn!("luma: the matrix kernel {spec:?} failed its verification (largest relative error {}) and is disabled on this device", verdict.max_rel_err);
                }
                return verdict;
            }
            log::warn!("luma: the matrix kernel {spec:?} could not be verified and is disabled");
            UnitVerdict { variant, ok: false, max_rel_err: f32::INFINITY }
        });
        (verdict, waited)
    }

    /// Runs the verification; `None` if it could not run, else the verdict and whether it is to be stored. The kernel runs twice on the
    /// problem and the two outputs must be bit-identical: a race inside a workgroup (shared memory read before the barrier that
    /// completes it) can stay within the error bound while the output changes from launch to launch.
    fn run(&self, backend: &dyn Backend, spec: &GemmSpec, variant: u64) -> Option<(UnitVerdict, bool)> {
        let problem = Problem::new(spec)?;
        // A tampered run is a test's: its verdict is not kept.
        let tamper = self.tamper.lock().unwrap_or_else(|e| e.into_inner()).clone();
        let mut runs = Vec::with_capacity(2);
        for _ in 0..2 {
            let mut got = match problem.run(backend) {
                Ok(got) => got,
                Err(error) => {
                    log::debug!("luma: verifying {spec:?}: {error}");
                    return None;
                }
            };
            if let Some(tamper) = &tamper {
                tamper(spec, &mut got);
            }
            runs.push(got);
        }
        let (ok, max_rel_err) = problem.judge(&runs[0]);
        let reproducible = bit_identical(&runs[0], &runs[1]);
        if ok && !reproducible {
            log::warn!("luma: the matrix kernel {spec:?} gave different results in two identical launches");
        }
        let max_rel_err = if reproducible { max_rel_err } else { f32::INFINITY };
        Some((UnitVerdict { variant, ok: ok && reproducible, max_rel_err }, tamper.is_none()))
    }
}

/// Whether two outputs are the same bits (every NaN equal to every NaN: a payload is not a race).
fn bit_identical(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What a correct kernel returns, computed on the host: the slices of the product in `f32` from `f64` sums.
    fn host_kernel(problem: &Problem) -> Vec<f32> {
        let (m, n, k) = problem.dims;
        let mut out = vec![0f32; problem.slices() * m * n];
        for z in 0..problem.slices() {
            for i in 0..m {
                for j in 0..n {
                    let sum: f64 = (z * KC..k.min((z + 1) * KC)).map(|q| f64::from((problem.format.round)(problem.lhs(i, q))) * f64::from((problem.format.round)(problem.rhs(q, j)))).sum();
                    out[z * m * n + i * n + j] = sum as f32;
                }
            }
        }
        out
    }

    fn specs() -> Vec<GemmSpec> {
        let mut specs = Vec::new();
        for unit in [GemmUnit::Fma, GemmUnit::Tf32, GemmUnit::F16, GemmUnit::Bf16, GemmUnit::CoopF16] {
            for lanes in [GemmLanes::Vec4, GemmLanes::Vec4Tails, GemmLanes::Scalar] {
                for (lhs_t, rhs_t) in [(false, false), (true, false), (false, true), (true, true)] {
                    specs.push(GemmSpec::new(64, 64, lhs_t, rhs_t, unit).with_lanes(lanes));
                }
            }
        }
        specs
    }

    #[test]
    fn a_correct_kernel_passes_in_every_variant() {
        for spec in specs() {
            let problem = Problem::new(&spec).expect("known unit");
            let (ok, err) = problem.judge(&host_kernel(&problem));
            assert!(ok, "{spec:?}: {err}");
            assert!(err < 1e-6, "{spec:?}: the host kernel is far inside the bound ({err})");
        }
    }

    #[test]
    fn the_half_conversions_round_to_nearest_even() {
        // Every f16 survives the round trip (a NaN becomes the canonical one).
        for bits in 0..=u16::MAX {
            let value = f16_to_f32(bits);
            if value.is_nan() {
                assert!(f16_to_f32(f32_to_f16(value)).is_nan());
            } else {
                assert_eq!(f32_to_f16(value), bits, "{bits:#06x} = {value}");
            }
        }
        // Ties go to even, overflow to infinity, tiny values to zero, subnormals are kept.
        assert_eq!(f32_to_f16(1.0 + 2f32.powi(-11)), 0x3C00, "tie to even (down)");
        assert_eq!(f32_to_f16(1.0 + 3.0 * 2f32.powi(-11)), 0x3C02, "tie to even (up)");
        assert_eq!(f32_to_f16(65519.0), 0x7BFF);
        assert_eq!(f32_to_f16(65520.0), 0x7C00);
        assert_eq!(f32_to_f16(-1.0e9), 0xFC00);
        assert_eq!(f32_to_f16(2f32.powi(-25)), 0x0000, "half of the smallest subnormal ties to even");
        assert_eq!(f32_to_f16(2f32.powi(-24)), 0x0001);
        assert_eq!(f32_to_f16(f32::NAN) & 0x7FFF, 0x7E00);
        // bf16: the high half of the bits, rounded.
        assert_eq!(f32_to_bf16(1.0), 0x3F80);
        assert_eq!(f32_to_bf16(f32::from_bits(0x3F80_8000)), 0x3F80, "tie to even");
        assert_eq!(f32_to_bf16(f32::from_bits(0x3F81_8000)), 0x3F82);
        assert_eq!(f32_to_bf16(f32::INFINITY), 0x7F80);
        assert_eq!(f32_to_bf16(f32::NAN) & 0x7FFF, 0x7FC0);
        assert_eq!(bf16_to_f32(0x3F80), 1.0);
    }

    #[test]
    fn the_problem_has_the_edges_the_variant_needs() {
        let aligned = Problem::new(&GemmSpec::new(64, 64, false, false, GemmUnit::Tf32)).unwrap();
        assert_eq!(aligned.dims, (68, 72, 260));
        assert!(aligned.dims.0.is_multiple_of(4) && aligned.dims.1.is_multiple_of(4) && aligned.dims.2.is_multiple_of(4) && aligned.a.ld.is_multiple_of(4) && aligned.a.off.is_multiple_of(4));
        let ragged = Problem::new(&GemmSpec::new(128, 64, true, true, GemmUnit::Fma).with_lanes(GemmLanes::Scalar)).unwrap();
        assert_eq!(ragged.dims, (131, 69, 261));
        assert!(!ragged.a.ld.is_multiple_of(4) || !ragged.a.off.is_multiple_of(4), "the scalar lanes see an unaligned operand");
        assert!(aligned.slices() > 2, "split-K is covered");
        let half = Problem::new(&GemmSpec::new(64, 64, false, true, GemmUnit::F16)).unwrap();
        assert!(half.dims.0.is_multiple_of(8) && half.dims.1.is_multiple_of(8) && half.dims.2.is_multiple_of(8) && half.a.ld.is_multiple_of(8) && half.a.off.is_multiple_of(8), "8-element lanes for halves");
    }

    #[test]
    fn a_wrong_kernel_fails() {
        for spec in specs() {
            let problem = Problem::new(&spec).unwrap();
            let right = host_kernel(&problem);
            // One output wrong by far more than the bound.
            let mut wrong = right.clone();
            let at = 5 * problem.dims.1 + 7;
            wrong[at] += 1.0e12;
            assert!(!problem.judge(&wrong).0, "{spec:?}: one wrong output");
            // A swapped pair of fragment columns.
            let mut swapped = right.clone();
            let n = problem.dims.1;
            for z in 0..problem.slices() {
                for i in 0..problem.dims.0 {
                    swapped.swap(z * problem.dims.0 * n + i * n + 8, z * problem.dims.0 * n + i * n + 9);
                }
            }
            assert!(!problem.judge(&swapped).0, "{spec:?}: swapped columns");
            // A truncated result.
            assert!(!problem.judge(&right[1..]).0, "{spec:?}: wrong length");
        }
    }

    #[test]
    fn reproducibility_is_bit_equality() {
        let problem = Problem::new(&GemmSpec::new(64, 64, false, false, GemmUnit::Fma)).unwrap();
        let right = host_kernel(&problem);
        assert!(bit_identical(&right, &right.clone()), "NaNs of different payloads are the same result");
        let mut racy = right.clone();
        let at = right.iter().position(|x| x.is_finite() && *x != 0.0).unwrap();
        racy[at] = f32::from_bits(racy[at].to_bits() ^ 1);
        assert!(problem.judge(&racy).0, "one ulp stays inside the error bound: only the second launch can tell");
        assert!(!bit_identical(&right, &racy));
        assert!(!bit_identical(&right, &right[1..]));
        assert!(bit_identical(&[f32::NAN], &[-f32::NAN]));
        assert!(!bit_identical(&[0.0], &[-0.0]));
    }

    #[test]
    fn non_finite_references_compare_by_class() {
        let spec = GemmSpec::new(64, 64, false, false, GemmUnit::Fma);
        let problem = Problem::new(&spec).unwrap();
        let right = host_kernel(&problem);
        let (m, n, _) = problem.dims;
        // Row 1 holds the NaN (every output of it is NaN), column 2 the infinity.
        assert!((0..n).all(|j| right[n + j].is_nan() || (0..problem.slices()).any(|z| right[z * m * n + n + j].is_nan())));
        let mut fixed = right.clone();
        // A NaN where the reference is finite fails.
        fixed[0] = f32::NAN;
        assert!(!problem.judge(&fixed).0);
        // A finite value where the reference is NaN fails.
        let mut finite = right.clone();
        for z in 0..problem.slices() {
            finite[z * m * n + n + 4] = 1.0;
        }
        assert!(!problem.judge(&finite).0);
    }

    #[test]
    fn variants_differ_by_what_changes_the_kernel() {
        let spec = GemmSpec::new(64, 64, false, false, GemmUnit::Tf32);
        let key = |spec: &GemmSpec| variant_key(spec, 7);
        assert_eq!(key(&spec), key(&spec.with_half().with_hook(super::super::HookId(5))), "half and hooks share their plain product's verdict");
        for other in [
            GemmSpec::new(64, 128, false, false, GemmUnit::Tf32),
            GemmSpec::new(64, 64, true, false, GemmUnit::Tf32),
            GemmSpec::new(64, 64, false, true, GemmUnit::Tf32),
            GemmSpec::new(64, 64, false, false, GemmUnit::Fma),
            spec.with_lanes(GemmLanes::Scalar),
        ] {
            assert_ne!(key(&spec), key(&other), "{other:?}");
        }
        assert_ne!(variant_key(&spec, 7), variant_key(&spec, 8), "the generator's revision is part of the key");
    }

    #[test]
    fn verdicts_round_trip_through_the_disk_cache() {
        let dir = std::env::temp_dir().join(format!("luma-unit-verify-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let disk = super::super::DiskCache::open(Some(dir.clone()), "dev".into());
        assert_eq!(load(&disk, 42), None);
        let good = UnitVerdict { variant: 42, ok: true, max_rel_err: 1.5e-8 };
        let bad = UnitVerdict { variant: 43, ok: false, max_rel_err: 0.25 };
        store(&disk, &good);
        store(&disk, &bad);
        let reopened = super::super::DiskCache::open(Some(dir.clone()), "dev".into());
        assert_eq!(load(&reopened, 42), Some(good));
        assert_eq!(load(&reopened, 43), Some(bad));
        assert_eq!(load(&super::super::DiskCache::open(Some(dir.clone()), "other-device".into()), 42), None, "per device");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
