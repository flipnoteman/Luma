//! The PTX matrix products on shapes, leading dimensions and offsets that are not multiples of anything: every lane against a host
//! reference (bit for bit for `f32`), the lanes against each other (bit for bit for TF32), and NaNs in every element the product must
//! not read.

use std::collections::HashMap;
use std::mem::MaybeUninit;
use std::time::{Duration, Instant};

use luma_core::{Backend, BindingInfo, BindingKind, BufferUsage, CompiledKernel, Dispatch, GemmLanes, GemmSpec, RawArg};
use luma_cuda::CudaBackend;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

/// A row-major matrix of `rows x cols` in a buffer of `ld` columns from element `off`, NaN everywhere the matrix is not.
struct Stored {
    data: Vec<f32>,
    ld: usize,
    off: usize,
}

fn stored(rng: &mut Rng, rows: usize, cols: usize, ld: usize, off: usize) -> Stored {
    let mut data = vec![f32::NAN; off + rows * ld + 3];
    for r in 0..rows {
        for c in 0..cols {
            data[off + r * ld + c] = rng.next();
        }
    }
    Stored { data, ld, off }
}

impl Stored {
    fn at(&self, r: usize, c: usize) -> f32 {
        self.data[self.off + r * self.ld + c]
    }
}

fn bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn kernel(backend: &CudaBackend, spec: &GemmSpec) -> CompiledKernel {
    let raw = backend.compile_gemm(spec, None).expect("the backend plans the product");
    let start = Instant::now();
    while !raw.is_ready() {
        assert!(start.elapsed() < Duration::from_secs(60), "the kernel never finished compiling");
        std::thread::sleep(Duration::from_millis(1));
    }
    let binding = |name: &str, kind| BindingInfo { name: name.into(), kind, element_stride: None, min_size: None };
    let bindings = vec![
        binding("c", BindingKind::StorageReadWrite),
        binding("a", BindingKind::StorageRead),
        binding("b", BindingKind::StorageRead),
        BindingInfo { min_size: Some(32), ..binding("u", BindingKind::Uniform) },
    ];
    CompiledKernel::new("gemm".into(), raw, bindings, 4, None, [1, 1, 1])
}

/// The slices of the product `A B` (`m x k` by `k x n`, the operands as they are stored when `lhs_t` / `rhs_t`) of `kc` elements of K.
fn launch(backend: &CudaBackend, cache: &mut HashMap<GemmSpec, CompiledKernel>, spec: &GemmSpec, (m, n, k): (usize, usize, usize), kc: usize, a: &Stored, b: &Stored) -> Vec<f32> {
    let slices = k.div_ceil(kc).max(1);
    let max = u64::MAX;
    let a_buf = backend.alloc_filled(&bytes(&a.data), max).expect("a");
    let b_buf = backend.alloc_filled(&bytes(&b.data), max).expect("b");
    let c_buf = backend.create_buffer((slices * m * n * 4) as u64, BufferUsage::STORAGE, false).expect("c");
    let mut uniform = Vec::new();
    for word in [m, n, k, kc, a.ld, b.ld, a.off, b.off] {
        uniform.extend_from_slice(&(word as u32).to_le_bytes());
    }
    let grid = [n.div_ceil(spec.cols as usize) as u32, m.div_ceil(spec.rows as usize) as u32, slices as u32];
    let args = [RawArg::write(&c_buf, c_buf.size()), RawArg::read(&a_buf, a_buf.size()), RawArg::read(&b_buf, b_buf.size()), RawArg::Uniform(&uniform)];
    let compiled = cache.entry(*spec).or_insert_with(|| kernel(backend, spec));
    backend.record_launch(compiled, Dispatch::Direct(grid), [grid[0] * spec.cols, grid[1] * spec.rows, grid[2]], &args).expect("the launch is recorded");
    backend.sync(Duration::ZERO).expect("the launch ran");
    let mut out = vec![MaybeUninit::<u8>::uninit(); slices * m * n * 4];
    backend.read_into(&c_buf, &mut out, false, Duration::ZERO).expect("the result");
    out.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(c.map(|byte| unsafe { byte.assume_init() }))).collect()
}

/// `f32` reference: the chain of fused multiply-adds over `k` in ascending order of each slice.
fn reference(dims: (usize, usize, usize), kc: usize, lhs_t: bool, rhs_t: bool, a: &Stored, b: &Stored) -> Vec<f32> {
    let (m, n, k) = dims;
    let slices = k.div_ceil(kc).max(1);
    let mut out = vec![0f32; slices * m * n];
    for z in 0..slices {
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0f32;
                for q in z * kc..k.min((z + 1) * kc) {
                    let x = if lhs_t { a.at(q, i) } else { a.at(i, q) };
                    let y = if rhs_t { b.at(j, q) } else { b.at(q, j) };
                    acc = x.mul_add(y, acc);
                }
                out[z * m * n + i * n + j] = acc;
            }
        }
    }
    out
}

/// Bound of the TF32 error of one output: 2^-10 relative to the sum of the magnitudes.
fn tf32_close(dims: (usize, usize, usize), lhs_t: bool, rhs_t: bool, a: &Stored, b: &Stored, got: &[f32]) {
    let (m, n, k) = dims;
    for i in 0..m {
        for j in 0..n {
            let (mut exact, mut mag) = (0f64, 0f64);
            for q in 0..k {
                let x = f64::from(if lhs_t { a.at(q, i) } else { a.at(i, q) });
                let y = f64::from(if rhs_t { b.at(j, q) } else { b.at(q, j) });
                exact += x * y;
                mag += (x * y).abs();
            }
            let value = f64::from(got[i * n + j]);
            assert!((value - exact).abs() <= 2e-3 * mag + 1e-6, "({i}, {j}) of {m}x{n}x{k}: {value} against {exact}");
        }
    }
}

fn up(x: usize) -> usize {
    x.next_multiple_of(4)
}

#[test]
fn every_lane_matches_the_reference_on_ragged_shapes_leading_dimensions_and_offsets() {
    let backend = match CudaBackend::new() {
        Ok(backend) => backend,
        Err(error) => {
            println!("SKIPPED: no CUDA device ({error})");
            return;
        }
    };
    if !backend.gemm_ragged() {
        println!("SKIPPED: the device has no ragged matrix products");
        return;
    }
    let mut cache = HashMap::new();
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let shapes = [(1, 1, 1), (3, 5, 7), (17, 33, 31), (100, 31, 129), (65, 64, 33), (31, 130, 17), (129, 70, 200)];
    for exact in [true, false] {
        let tiles: Vec<(u32, u32)> = backend.gemm_tiles(exact).to_vec();
        for &(m, n, k) in &shapes {
            for (lhs_t, rhs_t) in [(false, false), (true, false), (false, true), (true, true)] {
                // The stored shapes: `A` is `m x k` (or `k x m`), `B` is `k x n` (or `n x k`).
                let (a_rows, a_cols) = if lhs_t { (k, m) } else { (m, k) };
                let (b_rows, b_cols) = if rhs_t { (n, k) } else { (k, n) };
                // Scalar takes any leading dimension and offset; the 16-byte lanes need multiples of 4.
                let odd_a = stored(&mut rng, a_rows, a_cols, a_cols + 3, 5);
                let odd_b = stored(&mut rng, b_rows, b_cols, b_cols + 1, 3);
                let even_a = stored(&mut rng, a_rows, a_cols, up(a_cols) + 4, 8);
                let even_b = stored(&mut rng, b_rows, b_cols, up(b_cols), 4);
                for (index, &(rows, cols)) in tiles.iter().enumerate() {
                    // Every tile at the small shapes, a few at the big ones.
                    if m * n * k > 50_000 && index % 2 == 1 {
                        continue;
                    }
                    let spec = GemmSpec::new(rows, cols, lhs_t, rhs_t, exact);
                    for kc in [k, 32usize.min(k)] {
                        let dims = (m, n, k);
                        let scalar = launch(&backend, &mut cache, &spec.with_lanes(GemmLanes::Scalar), dims, kc, &odd_a, &odd_b);
                        let tails = launch(&backend, &mut cache, &spec.with_lanes(GemmLanes::Vec4Tails), dims, kc, &even_a, &even_b);
                        let scalar_even = launch(&backend, &mut cache, &spec.with_lanes(GemmLanes::Scalar), dims, kc, &even_a, &even_b);
                        let label = format!("{rows}x{cols} {m}x{n}x{k} kc={kc} lhs_t={lhs_t} rhs_t={rhs_t} exact={exact}");
                        if exact {
                            let want_odd = reference(dims, kc, lhs_t, rhs_t, &odd_a, &odd_b);
                            let want_even = reference(dims, kc, lhs_t, rhs_t, &even_a, &even_b);
                            // The operands differ in their padding only: the values are the draws in order, so compare each lane to its own reference.
                            assert!(scalar.iter().zip(&want_odd).all(|(x, y)| x.to_bits() == y.to_bits()), "scalar, odd ld: {label}");
                            assert!(tails.iter().zip(&want_even).all(|(x, y)| x.to_bits() == y.to_bits()), "tails: {label}");
                            assert!(scalar_even.iter().zip(&want_even).all(|(x, y)| x.to_bits() == y.to_bits()), "scalar, even ld: {label}");
                        } else {
                            assert!(tails.iter().zip(&scalar_even).all(|(x, y)| x.to_bits() == y.to_bits()), "tails and scalar differ: {label}");
                            if kc == k {
                                tf32_close(dims, lhs_t, rhs_t, &odd_a, &odd_b, &scalar);
                                tf32_close(dims, lhs_t, rhs_t, &even_a, &even_b, &tails);
                            }
                        }
                        // The aligned lane where it applies: multiples of 4 everywhere.
                        if m % 4 == 0 && n % 4 == 0 && k % 4 == 0 {
                            let vec4 = launch(&backend, &mut cache, &spec, dims, kc, &even_a, &even_b);
                            assert!(vec4.iter().zip(&scalar_even).all(|(x, y)| x.to_bits() == y.to_bits()), "vec4 and scalar differ: {label}");
                        }
                    }
                }
            }
        }
    }
}
