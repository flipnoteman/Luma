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
* Numerical failure never poisons or raises: a matrix that is not positive definite leaves `NaN`
  on the diagonal of its Cholesky factor, a singular matrix a zero on the diagonal of `U` (the
  solve then gives `Inf`/`NaN`, the determinant 0). Shape errors poison the result as usual.
