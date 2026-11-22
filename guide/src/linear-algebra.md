# Dense linear algebra

Factorisations and solvers for dense `f32` matrices, on every backend: Cholesky, triangular and
symmetric positive definite solves, LU with partial pivoting, general solves, inverse and
determinant, Householder QR and least squares. They are ordinary lazy operations: recorded, not
waited for; the factors are arrays that stay on the device.

## Layout

A matrix is `[cols, rows]` like the operands of `matmul` (row `i` is contiguous, element `(i, j)`
at `j + cols * i`), and a batch of matrices is a third axis, `[cols, rows, batch]`. A right-hand
side of `k` columns for an `n`-row system is `[k, n]`; a plain `[n]` array is one vector and the
solution has the shape of the right-hand side.

```rust
use luma::*;

// [[4, 2], [2, 5]] x = [2, 3]  =>  x = [0.25, 0.5]
let a = Array::from_slice(&[4.0f32, 2.0, 2.0, 5.0], [2, 2])?;
let b = Array::from_slice(&[2.0f32, 3.0], 2)?;

let l = a.cholesky();                      // A = L Lᵀ, L lower triangular
assert_eq!(l.to_vec()?, vec![2.0, 0.0, 1.0, 2.0]);
let x = a.solve_spd(&b);                   // Cholesky + two triangular solves
let y = a.solve(&b);                       // LU with partial pivoting
for (u, v) in x.to_vec()?.iter().zip(y.to_vec()?) {
    assert!((u - v).abs() < 1e-5);
}
assert!((a.det().item()? - 16.0).abs() < 1e-4);

let f = a.lu();                            // P A = L U, packed, with the permutation
let back = f.solve(&b);                    // reuse the factors for more right-hand sides
assert!((back.to_vec()?[0] - 0.25).abs() < 1e-5);
# Ok::<(), luma::LumaError>(())
```

| Method | Result |
| --- | --- |
| `cholesky()`, `cholesky_upper()` | the factor `L` (`A = L Lᵀ`) or `U` (`A = Uᵀ U`) of an SPD matrix; reads one triangle |
| `solve_spd(&b)`, `cholesky_solve(&b)` | `A X = B` for SPD `A` (from `A`, or from its factor) |
| `solve_triangular(&b, Triangle, transpose, unit_diag)` | `op(A) X = B` for a triangular `A` |
| `lu()` | `Lu { lu, perm }`: `P A = L U`, with `lower()`, `upper()`, `solve`, `inverse`, `det` |
| `solve(&b)` | square: LU; tall: least squares; wide: minimum norm |
| `inverse()`, `det()`, `slogdet()` | inverse; determinant; sign and log of the absolute determinant (no overflow) |
| `qr()` | `Qr { qr, tau, t }` with `r()`, `q_thin()`, `q()`, `apply_q`, `apply_qt`, `lstsq` |
| `lstsq(&b)` | least squares by QR |

## What runs where

* Small matrices and every matrix of a batch are factored by one workgroup each, in one launch:
  a batch of 4096 matrices of 16 x 16 costs one launch, where a library without batched LAPACK
  loops over calls.
* A big matrix is factored recursively; the leaves are small kernels and every other flop is a
  full-`f32` `matmul_f32` of the halves (never TF32: a factorisation's accuracy depends on it).
* A triangular solve with up to 32 right-hand sides (`solve_triangular`, and the two solves inside
  `solve` and `solve_spd`) of a triangle of more than 64 rows is one launch on CUDA devices: one
  workgroup per block of 64 rows takes its turn from an atomic ticket, waits for the solved blocks
  before it (the words they publish are the complement of the bits, 0 meaning "not there yet": no flags
  or fences), and solves its own block. The factorisations' solves take the diagonal blocks as explicit
  inverses (computed while the workgroup waits; the blocks are well-conditioned triangles of a
  factorisation, a Cholesky factor or a unit `L` with bounded multipliers); `solve_triangular` of an
  arbitrary triangle substitutes. Many right-hand sides are blocks of 64 rows multiplied by their inverses
  inside the recursion of halves with `matmul_f32` (every backend).
* Numerical failure never poisons or raises: a matrix that is not positive definite leaves `NaN`
  on the diagonal of its Cholesky factor, a singular matrix a zero on the diagonal of `U` (the
  solve then gives `Inf`/`NaN`, the determinant 0). Shape errors poison the result as usual.

## SVD, pseudo-inverse, rank and the symmetric eigenproblem

```rust
use luma::*;

let a = Array::from_slice(&[3.0f32, 0.0, 0.0, -2.0], [2, 2])?;
let svd = a.svd();                         // A = U diag(s) Vᵀ, s descending
let s = svd.s.to_vec()?;
assert!((s[0] - 3.0).abs() < 1e-5 && (s[1] - 2.0).abs() < 1e-5);
assert_eq!(a.rank().to_vec()?, vec![2]);   // singular values above max(m, n) eps s_max

let sym = Array::from_slice(&[2.0f32, 1.0, 1.0, 2.0], [2, 2])?;
let e = sym.eigh();                        // reads the lower triangle; values ascending
let l = e.values.to_vec()?;
assert!((l[0] - 1.0).abs() < 1e-5 && (l[1] - 3.0).abs() < 1e-5);
# Ok::<(), luma::LumaError>(())
```

| Method | Result |
| --- | --- |
| `svd()` | `Svd { u, s, vt }`, thin: `u` `[k, m]`, `s` `[k, 1]`, `vt` `[n, k]`, `k = min(m, n)`, any shape |
| `singular_values()` | the values alone (no `V`, no vectors written) |
| `pinv()`, `pinv_tol(tol)` | Moore-Penrose pseudo-inverse `[m, n]`; values `<= tol s_max` count as zero (default `max(m, n) eps`) |
| `rank()`, `rank_tol(tol)` | numerical rank as a `u32` `[1, 1]` |
| `cond()` | `s_max / s_min` |
| `eigh()`, `eigvalsh()` | `Eigh { values, vectors }` of a symmetric matrix (lower triangle read), ascending; the values alone |

The method is one-sided Jacobi: columns of `A` are rotated in pairs (the pairs of a round-robin
round are disjoint, so a round is parallel) until they are orthogonal; the singular values are
their norms. A matrix with its `V` that fits workgroup memory (up to 64 x 64 on devices with 32 KiB
of it, 32 x 32 with 16 KiB) and every matrix of a batch is decomposed by one workgroup, in one
launch, stopping inside the kernel when no rotation is left. A bigger matrix lives in global
memory and one launch per round-robin round rotates `n / 2` pairs at once; the sweeps continue
until one rotates nothing (the host looks every few sweeps, so a big `svd()` waits for the device,
unlike the other operations). `eigh` is the SVD of `A + ||A||_F I` read from the lower triangle:
positive semi-definite, so `U = V` holds the eigenvectors and `lambda = sigma - ||A||_F`.

Accuracy in `f32`: `||A - U diag(s) Vᵀ||_F / ||A||_F` and the singular values are within a few
`eps` of `s_max` (absolute), the orthogonality of `U` and `V` degrades as about `n sqrt(n) eps`;
the eigenvalues of `eigh` carry an absolute error of a few `eps ||A||_F`. A zero singular value
gets a left vector completed to an orthonormal set. `NaN` input gives `NaN` values and never
hangs (the sweeps are bounded).
