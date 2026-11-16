# Writing a PDE solver in Luma array ops (no custom kernels)

Notes from porting **bowshock** (2-D axisymmetric compressible flow: Euler, then RANS with k-omega SST) to the GPU.
Today bowshock's GPU path is about 3 000 lines of hand-written WGSL in custom kernels, a third copy of math that
already exists as equations in doc comments and as f64 Rust on the CPU. The goal here is the opposite: write the
math once as array expressions that read like the equations, and let Luma fuse them.

This file lists what Luma needs for that, ordered by what blocks first. Every item says why, with the bowshock
code that needs it. Status checked against `ff1ff99` (0.2.4).

---

## The shape of the problem

A finite-volume solver on a structured `ni x nj` grid, one iteration:

1. **Primitives** per cell: `rho, u, v, p` from the conserved state `q = (rho, rho u, rho v, E)`.
2. **Ghost cells**: two layers around the domain, plus the cells of a solid body block. Each boundary has its own
   rule: free stream, extrapolation, mirror across a wall (slip: reflect velocity about the wall normal; no-slip:
   reverse it), characteristic far field (nonlinear function of the interior state).
3. **Face states**: MUSCL reconstruction with a limiter, from 4 cells around each face.
4. **Face fluxes**: approximate Riemann solver (HLLC, blended with Rusanov near shocks) on every face.
   i-faces are `(ni + 1) x nj`, j-faces are `ni x (nj + 1)`.
5. **Residual** per cell: `-(F[i+1] w[i+1] - F[i] w[i] + G[j+1] w[j+1] - G[j] w[j]) / V + source`.
6. **Time step** per cell (local CFL), **update**, and a **residual norm** for the stop rule.
7. Implicit variant: two batched **block-tridiagonal solves** (4 x 4 blocks) along every grid line.
8. RANS adds: cell gradients (Green-Gauss over the 4 faces), viscous face fluxes, two more transported scalars
   (k, omega) with the same face/cell structure, and a wall-distance field.

Everything except step 7 is stencils and pointwise math. Pointwise math is branchy (Riemann solver cases,
limiters, positivity fallbacks, turbulence-model `min`/`max`/`tanh` blends) and long: one HLLC flux is ~150
operations per component.

---

## Tier 1: needed to write the math at all

### 1. Slices and non-circular shifts

**Why.** Face arrays have one more entry than cell arrays along their axis. A residual is a difference of two
slices of a face array; a face state reads the two cells either side of it. `shift` today is circular
(`[1,2,3,4].shift(1) == [4,1,2,3]`), so every stencil reads the opposite boundary, and there is no way to change
the extent of an axis.

**Proposed.**

```rust
a.slice(axis, range)             // view, lazy, fuses as an offset read; range may be strided (step) later
a.shift_fill(offsets, value)     // non-wrapping, out-of-range reads give `value`
a.shift_clamp(offsets)           // out-of-range reads give the edge element
a.shift_mirror(offsets)          // out-of-range reads mirror about the edge
```

`slice` returning a lazy view (no copy) matters: `f.slice(0, 1..) - f.slice(0, ..ni)` must be one kernel.

**bowshock use.** Every flux and residual. Without it nothing else in this list can be used.

### 2. Region writes and padding (boundary conditions as expressions)

**Why.** Boundary conditions are "the ghost row equals some function of the first interior row(s)". With a padded
array (`(ni + 4) x (nj + 4)`), each boundary is a region assignment. The rules differ per edge and some are
nonlinear (characteristic far field), so a fixed `pad(mode)` is not enough on its own.

**Proposed.**

```rust
let w = interior.pad([(2, 2), (2, 2)], PadMode::Edge);   // allocate with ghost layers
w.slice(1, ..2).assign(&freestream_ghost(&w.slice(1, 2..4)));   // lazy region write, fuses with its producer
w.slice(0, ..2).assign(&mirror_wall(&w.slice(0, 2..4).flip(0), &nx, &nr));
```

Needs: `pad`, `flip(axis)` (mirror order for wall ghosts), and `assign` into a slice that is lazy and in place
(compound assignment already fuses in 0.2; extend it to slices).

### 3. Elementwise functions that are missing

`tanh` (SST blending functions F1, F2), `log` (wall functions, Reynolds-number correlations), `sign` (limiters),
`atan2`, `hypot` (face normals, wall distances), `floor`/`ceil`, integer `%`, `copysign`. Present today: `sqrt`,
`exp`, `abs`, `powf`, `min`/`max`/`minimum`/`maximum`/`clamp`, comparisons, `select`.

### 4. Lazy 0-d scalars that broadcast

**Why.** Values computed on the GPU that feed the next expression: the residual norm (stop rule), a CFL that ramps
with the iteration count, the free-stream state when the Mach number changes, a global minimum time step.
Each of these is a host readback per step today, or a kernel rebuild.

**Proposed.** A reduction returns `Array<f32>` of shape `[1]` (lazy), and binary ops broadcast a shape-`[1]`
operand. A scalar uniform that can change between launches without recompiling (`Scalar::new(0.5)`,
`s.set(0.7)`) covers host-side parameters.

### 5. Records of arrays (or vec4 elements)

**Why.** The state is 4 components per cell, the gradient is 6, the turbulence state is 2. Writing every
operation on parallel scalar arrays quadruples the code and loses the reading of the equations
(`F = 0.5 (F_L + F_R) - 0.5 lam (q_R - q_L)` should be one line).

**Proposed, either:**

- `Array<[f32; N]>` with component access (`q.c(0)`), component-wise arithmetic, `dot`, and scalar
  broadcast across components; or
- a derive for a struct of arrays (`#[derive(luma::Fields)] struct Cons { rho, mu, mv, e }`) that gives
  the same operators and fuses all fields into one kernel (multi-output fusion exists already).

### 6. Gather / scatter

`a.take(&idx)` and `a.put(&idx, &values)` with `Array<u32>` index arrays. bowshock needs it for the mirrored ghost
cells behind a base wall (a fixed index map), the cells across the solid body block, and per-line lookups.

---

## Tier 2: needed for it to be fast

Without these, the code from tier 1 is correct but slower than the custom kernels it replaces.

### 7. Fusion that scales to large expression graphs

- **Node limit**: one HLLC face flux is ~600 nodes for 4 components. The default limit (128,
  `LUMA_FUSION_MAX_NODES`) splits it into ~5 kernels with materialised intermediates. Raise the default, allow a
  per-expression override, and split on estimated register pressure, not node count.
- **Common subexpressions**: an intermediate read several times (the sound speed, a limited slope used in two
  face states) must be computed once per thread.
- **A profile note when an expression was split**, with where and why.

### 8. Materialise instead of recompute for shifted reads

Today a small lazy expression read through a shift is recomputed at the shifted element. For stencils that read
the same expensive intermediate at several offsets, that multiplies the work: a cell residual reads the flux of 2
faces per direction, each flux reads a reconstruction over 4 cells. Rule of thumb: if a node is read at more than
2 offsets, or costs more than ~N operations, evaluate it into a buffer (or workgroup memory, as the staged shifts
already do) and read that.

An explicit `a.cache()` (materialise here, do not wait) as a user hint would cover the remaining cases.

### 9. Record an iteration once, replay it

One solver iteration is 5-15 kernels. bowshock runs 50 iterations between host syncs. Building and recording the
same launch sequence every step costs host time and showed up as a regression in 0.2.4 for custom kernels
(+2 to +7 % per iteration vs 0.1.3 on the same kernels). A recorded plan replayed k times, with only scalar
uniforms changing, removes that (the equivalent of CUDA graphs). `iterate*` could use it internally.

---

## Tier 3: the solver loop

### 10. Multi-array `iterate_until` with a user stop rule

`iterate_until(max_steps, tol, step)` today: one `Array<f32>`, stop when `max|next - prev| <= tol`.
A solver needs:

- several arrays of state (`q`, `k`, `omega`), as `iterate_all` has;
- a **user reduction and predicate**, evaluated on the GPU: "density residual norm 6 orders below its peak",
  "the monitored drag coefficient varied less than 1e-4 over the last 2 000 steps";
- the **step index** inside the step closure (first-order start-up for the first N steps, CFL ramp
  `cfl0 * growth^k`);
- **per-step scalar history** returned at the end (the residual of every step, written to CSV for convergence plots);
- the reason it stopped (which predicate fired, or max steps).

```rust
let run = luma::iterate_until_all(
    (q, k, om),
    Until::new(max_steps)
        .record("residual", |s| s.q_res.norm())          // per-step scalar, kept as history
        .stop_when(|h| h["residual"].peak() / h["residual"].last() >= 1e6),
    |state, step| solver_step(state, step),                   // `step` is a lazy scalar
)?;
```

---

## Tier 4: library primitives (not user math)

Some algorithms are not pointwise or stencils. They should be Luma primitives, so users never write kernels for
them.

### 11. Batched tridiagonal and block-tridiagonal solves along an axis

`luma::solve_tridiagonal(axis, lo, diag, up, rhs)` and the block version with `B x B` blocks (bowshock: 4 x 4
for the flow, 2 x 2 or diagonal for turbulence), one independent system per grid line. Cyclic reduction or PCR
in workgroup memory. This is the single biggest missing piece for implicit PDE solvers (line relaxation, ADI,
implicit diffusion). Optional: a segment start per line (lines that begin past a solid region).

### 12. Precision: f64, or a double-float type

bowshock's GPU path matches its f64 CPU reference to 1e-9 only because the residual is evaluated in double-float
(pairs of f32 with error-free addition and multiplication). That needs operation order that no compiler step
reorders (bowshock reads an opaque zero from a uniform to stop the driver turning `(a + b) - a` into `b`).

- `f64` arrays where the device supports `SHADER_F64` (slow on consumer GPUs, still invaluable for validation);
- a library `df32` element type with the error-free transformations built in and guarded against
  reassociation, usable through the same array ops;
- deterministic-order (or compensated) reductions as an option, for residual norms and regression tests.

### 13. Scans along an axis

Prefix sums and associative scans with a user operator (cumulative quantities along a wall, integrals along grid
lines).

---

## Smaller things

- `flip(axis)`, `transpose` (lazy), `concat` along an axis.
- Masked reductions (`sum_where(mask)`); multiplying by a mask works but costs a pass.
- Reductions along one axis (`sum(axis)`), e.g. integrate pressure along a wall line.
- `Array<u32>` index generators per axis (`index(axis)` with the array's shape), for position-dependent masks.
- A debug mode that checks every evaluated array for NaN/Inf and names the expression that produced it.

---

## What the Euler residual would look like

Target reading for the proposed API (Rusanov flux for brevity; HLLC is the same with more `select`):

```rust
// q: Cons (record of 4 arrays), ni x nj; w: primitives with 2 ghost layers, (ni+4) x (nj+4)
let w = prim(&q).pad([(2, 2), (2, 2)], PadMode::Edge);
apply_bcs(&w, &cfg);                                            // region assigns, tier 1 item 2

// i-faces: (ni + 1) x nj
let (m2, m1, p0, p1) = (w.slice(0, 0..ni+1), w.slice(0, 1..ni+2), w.slice(0, 2..ni+3), w.slice(0, 3..ni+4));
let (wl, wr) = muscl(&m2, &m1, &p0, &p1, limiter);              // pointwise, uses select / sign
let lam = spectral(&wl, &face.nx, &face.nr).max(&spectral(&wr, &face.nx, &face.nr));
let fi  = 0.5 * (flux(&wl, &face.nx, &face.nr) + flux(&wr, &face.nx, &face.nr))
        - 0.5 * &lam * (cons(&wr) - cons(&wl));
let fi  = fi.cache();                                           // tier 2 item 8: read twice below

// j-faces likewise -> gj, ni x (nj + 1)

let res = -( fi.slice(0, 1..) * &wi.slice(0, 1..) - fi.slice(0, ..ni) * &wi.slice(0, ..ni)
           + gj.slice(1, 1..) * &wj.slice(1, 1..) - gj.slice(1, ..nj) * &wj.slice(1, ..nj) ) / &vol
          + source(&w, &area, &vol);
```

Every line is an equation from the paper. Features used: slices (1), padding and region writes (2), records (5),
`select`/`sign` inside `muscl` (3), large fusion with CSE (7), `cache` (8).

---

## Suggested order

| Step | Items | Unlocks |
|---|---|---|
| 1 | 1 slices / non-circular shift, 3 missing unary ops | stencils on bounded domains |
| 2 | 2 padding / region writes, 5 records of arrays | boundary conditions and readable equations |
| 3 | 7 fusion scaling, 8 materialise-vs-recompute | performance on par with hand-written kernels |
| 4 | 4 lazy scalars, 10 multi-array `iterate_until` | the whole solver loop on the GPU, no per-step host sync |
| 5 | 11 batched tridiagonal solve | implicit solvers |
| 6 | 12 df32 / f64, 9 recorded replay | validated precision, low launch overhead |

With steps 1-4, bowshock's explicit Euler solver and its RANS terms could drop their custom kernels. Steps 5-6
replace the last ones (implicit line solves, extended precision).

## How to check each feature

bowshock has a validated CPU reference for every piece above. A useful acceptance test for each Luma feature is
"the array-op version of bowshock routine X matches the f64 CPU routine to f32 round-off on a fixed state":
gradients, face states, fluxes, residual, time step, turbulence source terms. Those comparisons already exist for
the custom kernels (`bowshock-gpu-cases/tests/gpu_vs_cpu.rs`) and can be pointed at array-op versions.

---

# Findings from the bowshock-lazy prototype (explicit Euler in array ops)

Repo: `bowshock/bowshock-lazy` (branch `lazy-proto`). `cargo run --release --example luma_issues` in that crate
reproduces bugs 1-3.

## Bugs

1. **Df32 kernel with only compare/abs/min/max ops cannot bind its uniform.** `a.abs()`, `a.minimum(&b)`,
   `a.lt(&b)` on `Array<Df32>` fail with `create_bind_group: bindings (3) != layout (2)`. `calls_library`
   (`src/jit/program.rs` ~471-481) adds the guard uniform, but `df_abs/df_lt/df_min/df_max` (`src/builtin.rs`
   ~384-389) never read it, so naga drops the binding. Works with a scalar operand or any arithmetic op in the
   same kernel.
2. **`(&x - x.to_f32().to_df32()).to_f32()` is `[0, 0]` when fused** (want `x.lo`). The Df32 -> f32 cast
   (`src/jit/codegen.rs` ~210, `x.x + x.y`) is not guarded against reassociation like the library arithmetic.
   Correct when `x.to_f32()` is `cache()`d first.
3. **Input-buffer count wrong after evaluations:** an expression over 14 distinct buffers reports
   "21 input buffers > 16" and becomes 76 launches instead of 12, only after three arrays were `cache()`d in a row.
4. **Parallel tests sharing `default_context()` give wrong results** (state errors 1e-3..1e-1 in 3 of 4 tests under
   the default parallel `cargo test`); one `Context::new()` per test fixes it. Not reduced to a minimal repro yet.

## API gaps hit

- `Array` is not `Clone` (an identity `slice` was used as a second handle).
- `Fields`, `iterate_all`, `iterate_until_all` and the history are `f32` only: a Df32 state has to be carried as a
  hi/lo pair of f32 arrays, and a Df32 residual recorded as two records.
- No `Operand<f64> for f64` (native f64 port not attempted).
- `concat` evaluates all parts and selects between them, so ghost expressions must be cached first.
- Input-buffer limit 16 per kernel drives most of the staging work (the natural code made 166 launches per
  residual evaluation; hand staging with `cache()` got it to 18).

## Measured (RTX 3090, cone M=2, minmod, us/iteration, SSP-RK3)

| grid | CPU f64 (32 threads) | custom F32 | custom Extended | lazy f32 | lazy df32 |
|---|---|---|---|---|---|
| 100x50 | 613 | 29.0 | 74.9 | 97.2 | 243.5 |
| 200x100 | 1285 | 33.5 | 81.1 | 109.2 | 278.4 |
| 400x200 | 3199 | 61.7 | 258.2 | 158.6 | 618.8 |

Lazy is 2.4-3.4x the hand-written kernels. Main cause: 68-82 launches per RK3 step vs ~7, about ten of the 18 per
residual being tiny ghost-layer kernels at the ~3 us launch floor. Accuracy is excellent: lazy f32 matches the custom
F32 kernels (~1e-6), lazy Df32 matches the f64 CPU to ~3e-14 (vs ~5e-8 for the custom Extended kernels, which keep
f32 geometry). Compile: lazy df32 0.75-1.4 s vs 3.5-5.3 s for the custom Extended kernels.

## What would close the speed gap

- Raise or remove the 16-input-buffer limit per kernel (bind more buffers, or pack small arrays), so whole stages
  fuse without hand-placed `cache()`.
- Fuse many small region writes (`assign` of ghost layers on all four edges) into one kernel.
- Let the stage update of RK3 fuse with the residual that feeds it.
- Generic (Df32 / f64) `Fields` and `iterate*`.

## Found 2026-11-12 (Luma e4f7d22): parallel threads on `default_context()` are ~10x slower than private contexts

bowshock-lazy `tests/accuracy.rs` (4 tests, each builds lazy solvers in 2 precisions x 4 stagings and runs 20 steps):

| setup | wall |
|---|---|
| one `Context::new()` per test, parallel (default test harness) | 14 s |
| one `Context::new()` per test, `--test-threads=1` | 26 s |
| shared `default_context()`, `--test-threads=1` | 21 s |
| shared `default_context()`, parallel | **151 s** |

Results are correct in every setup. Sharing saves ~4 s of compiles when serial, but parallel threads on one context
slow each other down ~10x (lock contention or cross-thread flushing/serialisation, e.g. the F8 "evaluate every
held lazy array before tracing a loop"). To reproduce: in bowshock-lazy, replace `luma::Context::new()` by
`luma::default_context()` in `tests/accuracy.rs::both` and run `cargo test --release --test accuracy`.

---

# Status and next priorities (2026-11-12, Luma e4f7d22, bowshock-lazy 9086bda)

The four bugs and the API gaps listed under "Findings from the bowshock-lazy prototype" above are **fixed**
(Df32 compare kernels, guarded casts, input-buffer count, threads on `default_context()`, `Array: Clone`, generic
`Fields` / `iterate*`, lazy `concat`, device input-buffer limit). The prototype dropped all its workarounds.

## Where it stands (RTX 3090, cone M=2, minmod, us/iteration, `iterate_all` replay)

| grid | custom F32 | custom Extended | lazy f32 `Fluxes` | lazy df32 `Fluxes` | lazy f32 `Fusion` | lazy df32 `Fusion` |
|---|---|---|---|---|---|---|
| 100x50 | 28.2 | 91.0 | 34.6 | 191.7 | 155 | 782 |
| 200x100 | 35.3 | 85.2 | 42.5 | 236.2 | 211 | 1032 |

Kernels per SSP-RK3 step (200x100): `Fluxes` 14 (0 cuts), `Split` 20, `Lines` 26 (8 cuts), `Fusion` 67 (19 cuts).
`Fluxes` = hand-placed `cache()` of the padded state and the face fluxes; `Fusion` = no `cache()` at all, the code as
one would write it. Accuracy: lazy f32 ~1e-6, lazy Df32 ~1e-14 against the f64 CPU solver, all stagings.

## Next, in order

1. **Ghost regions recomputed by every reader (the main gap).** A region `assign` into a lazy padded array becomes
   `select(region, rhs, old)`, so every kernel that reads the padded array computes the ghost expressions at every
   cell it reads, and large ghost expressions push the readers over the fusion limits. This is why `Fusion` is 3-5x
   slower than the hand-staged `Fluxes`. Expected: the ghost region costs about its own size (evaluate it once, in
   workgroup memory or a small buffer, as the cost model now does for shifted reads; or compile the select so a cell
   computes only its own branch). Repro: bowshock-lazy `cargo run --release --example luma_issues`, case 1:
   0 / 10 / 40 ghost ops -> 2 launches, 100 ghost ops -> 4 launches, 2 cuts (`1537 inlined operations > 1536`).
   Done when `Staging::Fusion` is within ~1.2x of `Fluxes` on `cargo run --release --example bench -- --quick`.
2. **Df32 compile time.** First call (build + compile + trace) 3.3-3.4 s for `Fluxes`, 6-7 s for `Lines`, 10.6-12.5 s
   for `Fusion` (custom Extended kernels: ~4.8 s). Dominates test runs and the start of every sweep. Faster codegen
   for Df32 and/or a persistent on-disk pipeline cache.
3. **Parallel threads on one context ~10x slower** (section above: 151 s vs 14 s).
4. **Expression build cost on the host:** 0.7-1.25 us per operation (`luma_issues` case 2: 100 ops 71 us, 400 ops
   502 us, 900 ops 1074 us). The per-step `eval` loop form costs 2.0-2.6 ms per iteration for ~0.1 ms of GPU work
   (`luma::jit::create` leads callgrind). It got slower since the previous run (1.7 ms), possibly the
   simplify-on-creation step (not confirmed). `iterate*` replay avoids it, which is why it matters only for user loops.
5. **A scalar reduction is 3 launches** (partial sums, second pass, `sqrt(sum/N)` epilogue): 3 of the 14 kernels of
   a `Fluxes` RK3 step. Fuse the epilogue into the second pass. Repro: `bench -- --profile-only --staging Fluxes`.
6. **`solve_block_tridiagonal` for few, long lines.** It runs one invocation per line (Thomas): 100-400 threads on
   bowshock grids. bowshock's custom implicit kernels use cyclic reduction within a workgroup per line. A
   cyclic-reduction / PCR path for few lines decides whether the implicit scheme and RANS can move to array ops.
   Not yet measured in the prototype.

## Not worth effort now: cost-model calibration

`LUMA_CALIBRATE=off` changed nothing measurable (`Fluxes`, us/iteration, on -> off: f32 37.7 -> 37.3 / 48.4 -> 41.8,
df32 217 -> 192 / 277 -> 250 at 100x50 / 200x100, inside run-to-run noise; accuracy tests 3.8 s both ways). The
hand-placed `cache()` leaves the cost model few decisions. Calibration might matter for `Fusion`, but its bottleneck is
item 1. Keep it if cheap; consider off by default or a per-device on-disk cache of the result.

---

# Re-check (2026-11-12, Luma 837fa77, bowshock-lazy 266ad66): what changed, what is left

All 7 bowshock-lazy tests pass; accuracy unchanged (lazy f32 ~1e-6, lazy Df32 ~1e-14 vs the f64 CPU solver).

## Numbers (RTX 3090, cone M=2, minmod, us/iteration, `iterate_all` replay unless noted)

| grid | custom F32 | custom Extended | lazy f32 `Fluxes` (e4f7d22 -> now) | lazy df32 `Fluxes` (e4f7d22 -> now) |
|---|---|---|---|---|
| 100x50 | 28.7 | 75.1 | 34.6 -> **25.0** | 192 -> **166** |
| 200x100 | 44.0 | 82.1 | 42.5 -> **33.5** | 236 -> **203** |

By staging at 200x100 (replay; kernels / fusion cuts per SSP-RK3 step):

| staging | f32 us/iter | df32 us/iter | f32 launches / cuts | df32 launches / cuts |
|---|---|---|---|---|
| `Fusion` (no `cache()`) | 211 -> **34** | 1032 -> **431** | 67/19 -> **8/1** | 67/19 -> **40/5** |
| `Lines` | 106 -> 79 | 718 -> 615 | 26/8 -> 25/1 | 26/8 -> 33/7 |
| `Fluxes` | 42 -> 39 | 246 -> 212 | 14/0 -> 12/0 | 14/0 -> 14/1 |
| `Split` | 62 -> 54 | 236 -> 203 | 20/0 -> 18/0 | 20/0 -> 20/1 |

Lazy f32 now beats the hand-written F32 kernels, and for f32 the code as written (`Fusion`) is as fast as the
hand-staged `Fluxes`. Df32 is still ~2.2-2.5x the hand-written Extended kernels, and Df32 `Fusion` is ~2x `Fluxes`.

## Status of the priority list above

1. **Ghost regions recomputed by every reader: mostly fixed.** `luma_issues` case 1: with the padded record dropped
   before evaluation (as in the solver) it is 1 kernel for 0-100 ghost ops. Still open: with the padded record still
   alive, 1 / 1 / 6 / 8 launches at 0 / 10 / 40 / 100 ghost ops, 2 cuts at 100 (`1537 inlined operations > 1536`,
   at the `assign`).
2. **Df32 compile time: fixed for warm runs.** First call (construction + compile + trace + 50 iterations) is now
   0.02-0.24 s (was 3.3-12.5 s), from the on-disk pipeline cache. Cold compile (empty cache) not re-measured.
3. **Parallel threads on one context: not re-measured** (bowshock-lazy now has a `LAZY_SHARED_CONTEXT` test option).
4. **Expression build cost: better, still the eval-loop bottleneck.** 0.47 / 0.70 / 0.77 us per op at 100 / 400 / 900
   ops (was 0.71 / 1.26 / 1.19). `eval` loop with `Fluxes`: 1.25-1.28 ms/iter (was 2.0-2.6 ms), still ~35-40x the
   replay form (`Fusion` eval 1.8-3.2 ms).
5. **Scalar reduction launches: not checked separately** (f32 `Fluxes` RK3 step 14 -> 12 launches).
6. **`solve_block_tridiagonal` for few, long lines: not exercised yet.**

## Next for Luma, in order

1. **Df32 under `Fusion`:** 40 launches and 5 cuts per RK3 step (f32: 8 and 1), 431 us vs 212 us for `Fluxes`.
   One residual evaluation is 4 launches / 3 cuts in Df32 vs 1 / 0 in f32. Repro: `bench -- --profile-only --staging
   Fusion` in bowshock-lazy (the profile lists the split notes).
2. **Ghost region with the padded record still alive** (item 1 above, `kept` column of `luma_issues` case 1).
3. **Expression build cost** (item 4), for user-written loops that do not use `iterate*`.
4. **Block-tridiagonal for few long lines** (next bowshock step: the implicit scheme on `solve_block_tridiagonal`).

---

# Re-check (2026-11-13, Luma 8620ec8, bowshock 6dfc33a)

Tests pass (7), accuracy unchanged. `bench --quick` (RTX 3090, us/iteration, replay; 837fa77 -> 8620ec8):

| grid | custom F32 | custom Ext. | lazy f32 `Fluxes` | lazy df32 `Fluxes` |
|---|---|---|---|---|
| 100x50 | 28.9 | 75.8 | 25.0 -> 24.9 | 166 -> 164 |
| 200x100 | 33.3 | 81.8 | 33.5 -> 33.4 | 203 -> 202 |

| 200x100 | f32 us/iter | df32 us/iter | df32 launches / cuts per RK3 step |
|---|---|---|---|
| `Fusion` | 34 -> 33 | **431 -> 247** | 40/5 -> **33/6** |
| `Lines` | 79 -> 77 | **615 -> 216** | 33/7 -> **20/7** |
| `Fluxes` | 39 -> 39 | 212 -> 213 | 14/1 -> 14/1 |
| `Split` | 54 -> 55 | 203 -> 202 | 20/1 -> 20/1 |

Df32 code-as-written is now ~1.15x of the hand-staged `Fluxes` (was ~2x). Df32 is still ~2.2-2.5x the hand-written
Extended kernels at the best staging: that ratio did not move, it is the floor of the Df32 arithmetic as generated
(per-RK3-step GPU time 224 us for `Fluxes` vs the custom Extended kernels' ~80 us/iteration).

Item status:
- Ghost region, padded record kept alive: better (40 ghost ops 6 -> 2 launches; 100 ops 8 -> 4 launches, 2 cuts at
  `1537 inlined operations > 1536`). Dropped case: 1 kernel throughout.
- Expression build cost: unchanged (0.49 / 0.71 / 0.79 us per op at 100 / 400 / 900 ops); `eval` loop 1.26-1.31 ms/iter
  with `Fluxes`.
- Warm first call: 0.02-0.18 s, except df32 `Fusion` at 100x50 1.23 s (cold compile for that shape).

Next for Luma, in order:
1. **Df32 kernel throughput.** The remaining gap to hand-written Extended is ~2.5x at equal launch count. Compare
   the generated WGSL of one Df32 residual kernel with bowshock-gpu's `shaders_ext.rs` (hand-written double-float:
   error-free transforms with FMA where available, `two_sum` only where needed, hi-only arithmetic for limiter and
   wave-speed decisions). Likely wins: use FMA-based `two_prod`, avoid renormalising after every op, keep comparisons
   and `select` conditions on the hi word.
2. Df32 `Fusion` cuts (6 per step) and the kept-alive ghost region at 100 ops.
3. Expression build cost (user-written eval loops).
4. Block-tridiagonal for few long lines (next bowshock step).

---

# Re-check (2026-11-13, Luma cd17279 "faster Df32 library", bowshock 6dfc33a)

## Perf: Df32 got much faster

`bench --quick`, RTX 3090, us/iteration, replay (8620ec8 -> cd17279):

| grid | CPU f64 | custom F32 | custom Ext. | lazy f32 `Fluxes` | lazy df32 best staging |
|---|---|---|---|---|---|
| 100x50 | 605 | 28.6 | 75.3 | 24.9 -> 25.0 | 164 -> **104** (`Fluxes`) |
| 200x100 | 1265 | 33-50 | 81.6 | 33.4 -> 33.4 | 202 -> **124** (`Lines`) |

Df32 by staging at 200x100: `Fusion` 247 -> 222, `Lines` 216 -> **124**, `Fluxes` 213 -> 125, `Split` 202 -> 141.
Lazy Df32 is now ~1.4-1.5x the hand-written Extended kernels (was ~2.5x) while matching the f64 CPU to ~1e-14
(Extended: ~5e-8). Df32 first call (cold for some shapes): `Fusion` 1.35-2.36 s, `Lines` 0.76-0.92 s.
Tests pass; accuracy unchanged.

## Bug: hang / heap corruption at process exit (NVIDIA driver compiler thread still running)

1. `cargo run --release --example bench -- --quick` in bowshock-lazy printed all its output, then aborted with
   `free(): chunks in smallbin corrupted` (exit 134).
2. Repeating `target/release/examples/bench --quick --staging Fluxes`: runs 1 and 2 exited cleanly, run 3 printed
   nothing on stderr and **hung for > 12 min at 100% CPU** (a normal run takes ~60 s).

Stacks of the hung run (gdb, `thread apply all bt`):
- main thread: inside `exit()` -> `__run_exit_handlers` -> `libEGL_nvidia.so` exit handler ->
  `pthread_rwlock_wrlock` (waiting);
- one driver thread (name inherited: "bench"): 20 frames deep in `libnvidia-gpucomp.so.615.71.09` (NVIDIA's shader
  compiler), spinning at 100% CPU; no Rust frames on that thread.

So the process reaches `exit()` while the NVIDIA driver is still compiling a pipeline on its own worker thread. The
driver's exit handler then waits on a lock the compiler holds (hang), or frees state the compiler is still using
(heap corruption: the earlier `corrupted size vs. prev_size` on 0.2.4 and this `free(): chunks in smallbin corrupted`
fit). Candidate trigger: the pipeline cache is now written "when a device is released and at exit" (6023a56), or
pipelines created asynchronously are still compiling when `main` returns. bowshock-lazy's bench also has a second
Luma (0.2.4, for the custom kernels) in the same process, so two devices/caches are released at exit.

Suggested: at device release / exit, wait for (or cancel) all pending pipeline creations before
`vkDestroyDevice` / before returning from the release path; and make the at-exit cache write not trigger compiles.
`LUMA_EXIT_WAIT` exists for NVIDIA driver threads; check that it covers compiler threads started by pipeline
creation (not only the workers of a submit).

Repro: in bowshock-lazy, `for k in $(seq 1 5); do target/release/examples/bench --quick --staging Fluxes >/dev/null; echo $?; done`
(hit at run 3 of 3 here; `luma_issues` 8/8 clean).

---

# New findings while porting more of the solver (2026-11-13, Luma cd17279)

Solid body block (base wall, block ghosts) and the line-implicit scheme now run on Luma array ops in bowshock-lazy
(accuracy vs the f64 CPU: explicit with block f32 ~1e-6 / Df32 ~3e-11; implicit f32 ~1e-6 / Df32 ~3e-9, converged
iteration counts identical to the CPU). `solve_block_tridiagonal` traces and replays inside `iterate_until_all` /
`iterate_all` fine, and the automatic cyclic-reduction choice is right (Thomas is 2.5x slower on 144x70 / 288x140).

## 1. No `stack`: `concat` of 16 parts is slow (biggest implicit-scheme cost after the solves)

`solve_block_tridiagonal` wants `[16, ni, nj]` blocks and a `[4, ni, nj]` right-hand side; the solver builds each block
entry as an `[ni, nj]` expression. `concat::<f32, 16>(0, parts)` of `[1, ni, nj]` parts costs 11.7 us per stack at
144x70 (161k elements; a bandwidth-bound copy is ~2 us). Six per implicit step = 70 us = 29% of the step's GPU time, as
much as the four line solves (82 us). Alternative tried: zeros base + 16 region assigns: ~20x slower overall.
Wish: a `stack(axis, [&Array; N])` that fuses its parts' expressions into one kernel writing the stacked layout, or
block solves that take `[Array; 16]` / `[Array; 4]` records directly (the builtin reads 16 buffers).

## 2. `reshape` evaluates its operand

`solve_result.slice(0, k..k+1).reshape([ni, nj])` (array.rs:708) materialises the slice. Workaround in bowshock-lazy:
`take` with precomputed index arrays, which stays lazy. Expected: a reshape of a contiguous slice stays a view.

## 3. No `is_finite` / `is_nan`

The positivity damping of the implicit update needs "all four components finite"; written as `x.abs().lt(1e30)`,
relying on NaN comparing false in the generated WGSL. An `is_finite()` -> `Array<u32>` would make it explicit.

## 4. Region assigns grow the fused expression past the operation limit

Base/wake grids zero the residual, `dt` and the fluxes on a solid block and overwrite block ghost layers with region
assigns (a few hundred cells each). One SSP-RK3 step on 156x76: f32 `Fluxes` 16 launches / ~70 us (2 cuts
`1537 inlined operations > 1536` at the zeroing `assign`s) vs 12 / 51 us on a block-free grid of the same size;
`Fusion` 89 launches / ~350 us vs 26 / 100 us; Df32 `Fusion` 96 launches / 2041 us (6 cuts: op limit at the zero /
`dt` assigns, "recomputing 9 times costs more than storing" at the base-flux assign). Every reader of the assigned array
inlines both branches. Possible: evaluate a region assign as "base expression, then a small kernel writing the region"
when the region is small, instead of inlining a select into every reader.

---

# Viscous terms + SST on Luma array ops (2026-11-13, Luma c709621)

Laminar Navier-Stokes and Menter SST now run in bowshock's array-op solver (matches the f64 CPU: explicit Df32
~1e-11..1e-13, implicit f32 ~1e-6..4e-5 / Df32 ~1e-7; converged G3 body total CD identical to 5 digits, 4000 implicit
SST iterations in 2.2 s f32 / 2.7 s Df32 vs ~30 s CPU). Per implicit SST step (156x122, f32): 65 launches, 522 us, the
six 16-part `concat` stacks take 105 us (the `stack` from 90e68d6 is not used yet in bowshock; will switch).

## 1. `Array<Df32>` has no `tanh`, `exp`, `ln`, `powf`

Only `sqrt` and `abs` (ops.rs). SST blending functions `F1 = tanh(a^4)`, `F2 = tanh(b^2)` are evaluated in f32 and
converted; Sutherland's `T^1.5` is written `T sqrt(T)`. Fine for this solver, but a Df32 `tanh`/`exp`/`ln` (even at
~f32x2 accuracy) would remove the precision switch from the code.

## 2. `slice_mut` on a `slice(..)` of a lazy expression: error text

`l.slice_mut(..).assign(..)` where `l` is itself `expr.slice(..)` fails with "cannot take a mutable view of a read-only
view". Expected either that it works (assign into the region of the underlying expression, as before b59b906) or an
error naming the fix (take `slice_mut` on the owner). Worked around by building the two face sides with separate `pad`s.

---

# Perf work: examples and numbers for a fixing cycle (2026-11-14, Luma c709621, bowshock main d2d50ac)

Status of the items above: `stack` (fixed, used), region-assign fusion (fixed), `reshape` lazy and `is_finite` (fixed in
Luma; bowshock still has the old workarounds, will switch). Still open in Luma: Df32 `tanh`/`exp`/`ln`/`powf`;
`slice_mut` on a `slice(..)` error text. Exit hang = NVIDIA driver, not Luma.

The remaining gap is speed: bowshock's array-op implicit scheme is ~1.7-2x slower per iteration than the old
hand-written WGSL kernels (old: sting 144x70 f32 106 us, Extended 144 us; base 156x76 97 / 143 us; viscous SST G3
156x122 263 / 319 us).

## How to reproduce (bowshock repo, `/home/sandvoxel/Documents/bowshock`, main)

bowshock pins Luma by git rev; to test a local Luma change, add to the bowshock root `Cargo.toml` (do not commit):

```toml
[patch."https://github.com/flipnoteman/Luma.git"]
luma = { path = "/home/sandvoxel/Documents/luma" }
```

```sh
cd bowshock-gpu
cargo run --release --example bench -- --implicit --quick          # implicit table + per-kernel profile (~35 s)
cargo run --release --example bench -- --quick --no-implicit       # explicit SSP-RK3 table + profiles
cargo run --release --example bench_viscous -- --levels 0          # viscous SST implicit, G3 body
cargo run --release --example bench_viscous -- --levels 0 --laminar
cargo run --release --example luma_issues                          # fusion / ghost-region / expression-build cases
cargo test --release -p bowshock-gpu-cases                         # accuracy vs the f64 CPU (must stay passing)
```

Source of the implicit step: `bowshock-gpu/src/implicit.rs` (assembly of 4x4 blocks as `[Array<f32>; 16]`,
`stack` -> `solve_block_tridiagonal(_from)` -> unstack via `take`), viscous/SST: `src/viscous.rs`, `src/turbulence.rs`.

## Example 1: implicit step, inviscid (today)

`bench -- --implicit --quick`, us per iteration (eval-loop `until` / `iterate_all` replay):

| case | grid | CPU f64 | f32 until / replay | df32 until / replay |
|---|---|---|---|---|
| sting | 144x70 | 3151 | 220 / 181 | 263 / 219 |
| sting | 288x140 | 9772 | 454 / 410 | 520 / 469 |
| base (solid block, line starts) | 156x76 | 2691 | 244 / 196 | 320 / 268 |
| base | 312x152 | 10599 | 490 / 442 | 648 / 576 |

Profile of one f32 sting step at 144x70: 27 launches, 191 us GPU time, 0 fusion cuts:

| kernel | launches | total us |
|---|---|---|
| `CyclicTridiagonal { block: 4, shared_rows: 144 }` (i lines, 70 lines of 144) | 2 | 46.0 |
| `CyclicTridiagonal { block: 4, shared_rows: 70 }` (j lines, 144 lines of 70) | 2 | 40.3 |
| `FusedStack` 16 parts (block matrices) | 6 | 18.1 |
| `FusedStack` 3-4 parts (right-hand sides) | 2 | 5.3 |
| everything else (residual, assembly, update; ~15 fused kernels) | ~15 | ~80 |

Targets / questions:
1. **Block line solve: ~21 us per solve of 70 lines x 144 rows of 4x4 blocks** (160 kB of blocks, a few MFLOP). The
   hand-written cyclic-reduction kernel did the same solves in roughly half. Ideas: more than one line per workgroup
   for short lines, `workgroup: 32` vs 64/128, register-blocked 4x4 inverse (closed-form cofactor inverse instead of
   pivoted elimination), fewer workgroup barriers per reduction level, reading `lo/diag/up` as `vec4` rows.
2. **Stacks: block solves taking `[Array<f32>; 16]` / `[Array<f32>; 4]` directly** (the builtin binds 16 + 16 + 16 + 4
   buffers or a fused gather) would remove the 8 stack launches (~23 us) and the unstack `take`s. Alternatively let
   the stack fuse into its producers (the 16 entries are elementwise expressions of the same cells).
3. **Replay anomaly to check:** the df32 sting replay printed 610 us for one step vs 244 us for the eval-loop step
   (other cases: replay ~= eval). Looks like a one-off (first replay compile counted?) but worth a look in
   `profile_replay` (bench.rs).

## Example 2: viscous SST implicit step (G3 body)

`bench_viscous -- --levels 0,1`, us per iteration (replay): f32 387 / 626, df32 519 / 797 (levels 0 / 1, 156x122 /
312x122); laminar f32 304 / 513. Old hand-written: 263 / 377 (F32), 319 / 428 (Extended).

One f32 SST step at level 0: ~65 launches, ~440 us. Six block line solves (~130 us), four scalar
`solve_tridiagonal` launches for k and omega (~24 us), stacks (~25 us), the rest gradient / viscous-flux / turbulence
source kernels. Targets:
1. Same block-solve items as above (twice as many solves per step here).
2. **Batch scalar tridiagonal solves:** k and omega are two independent scalar systems on the same lines; a
   `solve_tridiagonal` that takes a batch (`[2, ni, nj]` along axis 1/2, or `[&Array; N]`) would halve those launches.
3. **Launch count:** 65 launches for one step; check `profile.splits` for cuts that could fuse (e.g. gradients and
   viscous fluxes reading the same padded record).

## Example 3: Df32 cost

Explicit SSP-RK3 (bench `--quick --no-implicit`): Df32 ~4-5x f32 on the same staging; viscous Df32 1.25-1.35x f32
(most of the step is f32 LHS). If Df32 arithmetic gets cheaper (FMA products already in), the explicit path shows it
directly. Missing Df32 transcendental functions (`tanh` at least, for SST blending) are a correctness/convenience item,
not speed.

## Acceptance for any Luma perf change

- `cargo test --release -p bowshock-gpu-cases` passes unchanged (GPU-vs-CPU tolerances: f32 state ~1e-6, Df32
  ~1e-8..1e-14; converged iteration counts equal to the CPU).
- Report the `bench -- --implicit --quick` table and the profile rows above before/after.

---

# bowshock on luma `bowshock-fixes` (ebe212c): results (2026-11-14)

bowshock branch `luma-bf` (local, [patch] to ~/Documents/luma-bf). All GPU-vs-CPU tests pass, tolerances unchanged;
SST k/omega Df32 agreement improved with Df32 `tanh` (5.4e-6 -> 3.6e-6).

| us/iter (replay) | before (c709621) | after (bowshock-fixes) |
|---|---|---|
| implicit sting 144x70 f32 / df32 | 195 / 225 | 145 / 184 |
| implicit sting 288x140 f32 / df32 | 415 / 473 | 326 / 383 |
| implicit base 156x76 f32 / df32 | 201 / 270 | 164 / 231 |
| implicit base 312x152 f32 / df32 | 449 / 578 | 358 / 487 |
| viscous SST G3 156x122 f32 / df32 | 398 / 526 | 319 / 447 |
| viscous SST G3 312x122 f32 / df32 | 637 / 808 | 487 / 645 |

Launches per step: implicit 27 -> 19, viscous SST 65 -> 34. Stacks gone; `SolveParts` per-launch time ~ the old
`CyclicTridiagonal` (18-23 us at 156x76), so the line solve itself did not get faster in this case.

## New: Df32 kernels with Mask logic got slower (regression)

Df32 base-case implicit step, one eval-loop step (profile): 294 us -> 375 us, while f32 and the sting grid got faster.
- flux kernel `Fused[4](Generate{157}, Ge, Le, BitAnd, ...)` 60 -> 96 us
- residual kernel `Fused[4](Generate{156}, Le, ...)` 19 -> 56 us
Code: masks combined with `&` and used in `select` (bowshock-gpu `body.rs::reconstruct`, `implicit.rs::damping`; the
block masks from `Generate` ranges). Before: the same logic as `u32` comparisons multiplied together. Expected: Mask
`&`/`select` no slower than the u32 multiply. Repro: bowshock `luma-bf` branch, `cd bowshock-gpu && cargo run
--release --example bench -- --implicit --quick` (look at the "lazy df32 base" profile).

Not re-tested: `slice_mut` on a read-only slice (row 9; bowshock still uses two `pad`s).

---

# bowshock on luma `Array<bool>` (60e8931): results

bowshock branch `usable-lib` (`Mask` -> `Array<bool>`, no other change). All GPU-vs-CPU tests pass, tolerances unchanged.
The Df32 Mask regression above is fixed: lazy df32 base one eval-loop step 375 -> 283 us (flux kernel 59.3 us, residual
kernel 18.8 us; before the regression 294 us).

## Possible regression of the bool change: Df32 base-case replay +10 %

| us/iter (replay) | bowshock-fixes (bowshock 871640e) | 60e8931 |
|---|---|---|
| implicit sting 144x70 f32 / df32 | 145 / 184 | 149.9 / 185.0 |
| implicit base 156x76 f32 / df32 | 164 / 231 | 169.5 / **254.2** |
| implicit base 312x152 f32 / df32 | 358 / 487 | 354.8 / 478.7 |
| viscous SST G3 156x122 f32 / df32 | 319 / 447 | 320.1 / 450.9 |

Only the level-0 base df32 replay moved by more than 10 % (231 -> 254 us, +10.0 %), while its one-step profile got faster
(284 us replay step). [INFERENCE: run-to-run noise or a per-iteration host cost outside the profiled step.] Repro: bowshock
`usable-lib`, `cd bowshock-gpu && cargo run --release --example bench -- --implicit --quick`.

## Block-tridiagonal parts solve along axis 0 of thousands of lines is 7x slower than along axes 1 / 2 (bowshock 3D, 5x5 blocks)

`solve_block_tridiagonal_parts::<25, 5>` (one invocation per line, Thomas) on evaluated `[187, 70, 24]` f32 arrays (RTX 3090, 80 parts,
best of 5, `LUMA_TRIDIAGONAL` unset), time per solve:

| line axis | lines | n | us | algorithm chosen |
|---|---|---|---|---|
| 0 (`[187, 70, 24]`) | 1680 | 187 | **3300** | Thomas |
| 1 (`[187, 70, 24]`) | 4488 | 70 | 450 | Thomas |
| 2 (`[187, 70, 24]`) | 13090 | 24 | 290 | Thomas |
| 1 (`[70, 187, 24]`, the same lines as axis 0 above, stored with j fastest) | 1680 | 187 | 530 | Thomas |
| 0 forced `LUMA_TRIDIAGONAL=cyclic` | 1680 | 187 | 350 | cyclic |
| 2 forced `cyclic` | 13090 | 24 | 1700 | cyclic |

* Along axis 0 the neighbouring invocations read lines `n` elements apart: the loads of the 80 parts do not coalesce (one sector per
  invocation and part and step). The same lines stored with another axis fastest (axis 1 of the transposed arrays) solve 6x faster. The
  bowshock side works around it (it assembles and solves the i factor on transposed arrays, `gpu/implicit3.rs`), but a Luma-side fix
  would help every user of contiguous lines: stage the line through workgroup memory / a transposed tile when `stride == 1` and the line
  count is in the thousands, or let `cyclic::plan` count `lines * block^2` for `stride == 1` with the line length (here 1680 lines x 25 =
  42000 > `line_cyclic_work` 8192, so Thomas is chosen, and forced cyclic is 9x faster).
* The automatic choice has no per-call override: `LUMA_TRIDIAGONAL` is read when the context is created, so one process cannot take cyclic
  for the i lines (350 us) and Thomas for the j and k lines (450 / 290 us; cyclic is 540 / 1700 there). A `solve_*_with(LineSolver, ..)`
  entry point (or a hint argument) would give 2.2 ms instead of 5.1 ms per 3-direction pass of the ANF level-0 grid with the algorithms
  as measured.
* Stacked `solve_block_tridiagonal` (including the `stack`) is 1.0 - 1.7 ms along every axis on the same data: no help here.

Repro: bowshock `crates/bowshock-validate/examples/lift3d.rs --case anf --level 0 --precision df32 --bench 200` (profile of one step, the
`SolveParts` rows); the standalone probe was a 70-line `examples/solve3d_probe.rs` (random diagonally dominant blocks).
