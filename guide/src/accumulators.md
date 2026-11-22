# Incremental accumulation

Data that arrives in batches over time (a training loop's activations, a sensor, a simulation's
diagnostics, a log of metrics) rarely fits in memory at once, and its size is not known up front.
An *accumulator* folds every batch into a small **state** on the GPU, as the batch arrives, and
turns the state into results (a mean, a variance, a histogram, the `k` largest values, ...) at
any time. The batches are never kept; the states of two accumulators merge, so runs on several
threads, devices or machines combine afterwards; and a state is a few bytes you can write to a
file.

The accumulators are general building blocks: statistics, machine-learning metrics, logging
and signal analysis all use the same few.

| Type | State | Results |
| --- | --- | --- |
| [`Accumulator`] | count, sum, mean, variance, skewness, kurtosis, min, max, their global indices | per position along chosen axes, or of the whole batch; of masked elements; per group label |
| [`CoMoments`] | two streams: both means and variances and the sum of products | covariance, correlation, slope, per position |
| [`Covariance`] | the features of one stream: mean and sums of products | covariance and correlation matrix |
| [`Histogram`], [`Histogram2d`] | counts in fixed bins of one stream or of two jointly | counts, marginals, entropy, mutual information |
| [`TopK`] | the `k` largest or smallest values and where they came from | values and global indices |

Every accumulator has the same surface: `push` folds a batch in (nothing is read back from the
device; the work is recorded like any other luma operation, and a batch that is a lazy
expression such as `(&x - &y).abs()` is computed *inside* the accumulation kernel, never stored),
`merge(&other)` adds another accumulator's state, `reset()` clears it, `to_bytes()` and
`from_bytes()` write and read the state, and result methods return lazy arrays.

## Per-feature statistics

Batches of `rows` samples of `features` features are `[features, rows]` arrays (axis 0 is the
fast one, see the [matrices chapter](data.md)); folding axis 1 keeps one state per feature:

```rust
use luma::*;

let acc = Accumulator::<Df32>::builder(Fold::axis(1)).moments(2).sum().min_max().build()?;
for step in 0..10 {
    // 4 features, 100 rows per batch; feature f is f * 10 + step noise.
    let data: Vec<f32> = (0..400).map(|i| (i % 4) as f32 * 10.0 + ((i / 4 + step) % 7) as f32).collect();
    acc.push(&Array::from_slice(&data, [4, 100])?)?;
}
let mean = acc.mean()?.to_f32().to_vec()?;   // [4]
let var = acc.var()?.to_f32().to_vec()?;     // population variance, [4]
assert_eq!(acc.samples_seen(), 1000);
assert!((mean[2] - 23.0).abs() < 0.1);
assert!(var.iter().all(|v| *v > 3.5 && *v < 4.5));
assert_eq!(Accumulator::<Df32>::to_f64(&acc.count()?)?, vec![1000.0; 4]);
# Ok::<(), luma::LumaError>(())
```

`Fold::all()` folds every element of the batch into one stream; `Fold::axes(&[0, 1])` folds
several axes (an image batch per channel). The results have extent 1 along the folded axes, as
`mean_axis` does.

Results are arrays of the *wide* type of the state, `Df32` or `f64` (see Precision below):
`.to_f32()` rounds them, and [`Accumulator::to_f64`] reads exact values on the host.
A count of `2^24 + 3` samples is exact; a plain `f32` count stops at `2^24`.

The builder chooses what is tracked, so that you pay only for it: `moments(order)` (1 mean, 2
variance, 3 skewness, 4 kurtosis; every order is one more reduction in the pass over the
batch), `sum()`, `min()`, `max()`, `argmin()`, `argmax()`. Indices are *global sample
indices*: the position of the sample in the stream of everything pushed (and, after a merge,
of this accumulator's samples followed by the other's).

## Masks, NaN and groups

`push_where(&batch, &mask)` folds in the elements where the `bool` mask is true.
`NanPolicy::Skip` leaves NaN elements out of every statistic; with the default
`NanPolicy::Propagate` a NaN (or infinity) in a position's stream makes its mean and variance NaN
for good, and its extrema unspecified (backends differ in what `min` of a NaN gives).

`groups(n)` keeps a state per group label and position, and `push_grouped(&batch, &labels)`
folds a batch into all of them in one pass: `labels` is a `u32` array with one label per sample
(per element when the whole batch is folded); labels of `n` or more are skipped. The results have
`n` entries along the first folded axis: per-class means of every feature:

```rust
use luma::*;

let acc = Accumulator::<Df32>::builder(Fold::axis(1)).groups(3).build()?;
// 2 features, 6 samples; classes 0, 1, 2, 0, 1, 2.
let x = Array::from_slice(&[1.0f32, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0, 5.0, 50.0, 6.0, 60.0], [2, 6])?;
let labels = Array::from_slice(&[0u32, 1, 2, 0, 1, 2], 6)?;
acc.push_grouped(&x, &labels)?;
// Shape [2, 3]: feature f of class g at f + 2 g.
assert_eq!(Accumulator::<Df32>::to_f64(&acc.mean()?)?, vec![2.5, 25.0, 3.5, 35.0, 4.5, 45.0]);
# Ok::<(), luma::LumaError>(())
```

## Merging and serializing

States merge exactly as if the other's samples had been pushed after this accumulator's. The
merge is Chan et al.'s pairwise update, so it does not matter how the data was split or in which
order the partial states are combined (to the rounding of the wide type):

```rust
use luma::*;

let (a, b) = (Accumulator::<Df32>::new(Fold::all()), Accumulator::<Df32>::new(Fold::all()));
a.push(&Array::from_slice(&[1.0f32, 2.0, 3.0], 3)?)?;
b.push(&Array::from_slice(&[4.0f32, 5.0], 2)?)?;
// `b` could be on another thread, another device or another machine: its state is a byte string.
let shipped = b.to_bytes()?;
a.merge(&Accumulator::<Df32>::from_bytes(&shipped)?)?;
assert_eq!(Accumulator::<Df32>::to_f64(&a.mean()?)?, vec![3.0]);
assert_eq!(Accumulator::<Df32>::to_f64(&a.var()?)?, vec![2.0]);
# Ok::<(), luma::LumaError>(())
```

An accumulator is `Sync`: `push`, `merge` and the results take `&self`, so threads can share one
(pushes from several threads fold in some order) or each keep their own and merge at the end.
Merging with an accumulator on another device goes through the host. The bytes are the same for
both wide types (every value is stored as an `f64`), so an `f64` state written on one machine
is read by a `Df32` accumulator on another.

## Histograms, joint histograms and information

`Histogram::new(bins, lo, hi)` counts in equal-width bins over `[lo, hi]` (the top edge belongs
to the last bin); values below, above and NaN are counted aside ([`Histogram::outliers`]).
`Histogram::with_fold` keeps one histogram per position. `Histogram2d` counts pairs of two
streams jointly, and derives the marginals and the information measures that need only the
joint counts as lazy `f32` arrays:

```rust
use luma::*;

let joint = Histogram2d::<Df32>::new((2, 2), (0.0, 1.0), (0.0, 1.0))?;
// Two streams that always agree: all the counts are on the diagonal.
let x = Array::from_slice(&[0.1f32, 0.2, 0.8, 0.9], 4)?;
joint.push(&x, &x)?;
assert_eq!(Accumulator::<Df32>::to_f64(&joint.counts()?)?, vec![2.0, 0.0, 0.0, 2.0]);
let mi = joint.mutual_information()?.item()?;
assert!((mi - std::f32::consts::LN_2).abs() < 1e-6); // one bit
# Ok::<(), luma::LumaError>(())
```

## Top-k

`TopK::largest(k)` (and `smallest`) keeps the `k` best values of the whole stream with their
global indices; the pushes of up to 16 batches are sorted together and merged with the state's
`k` entries. Ties go to the earlier element; NaN and masked elements are never selected.

```rust
use luma::*;

let top = TopK::<f32, Df32>::largest(2);
top.push(&Array::from_slice(&[0.5f32, 9.0, 3.0], 3)?)?;
top.push(&Array::from_slice(&[7.0f32, 1.0], 2)?)?;
assert_eq!(top.values()?.to_vec()?, vec![9.0, 7.0]);
assert_eq!(TopK::<f32, Df32>::to_f64(&top.indices()?)?, vec![1.0, 3.0]);
# Ok::<(), luma::LumaError>(())
```

`TopK::largest_along(k, axis)` keeps the best `k` of every line along an axis instead, per
feature for `[features, rows]` batches with `axis` 1: the results have `k` entries along the axis
and the indices are *sample* indices along it. It costs `k` passes of an argmax over the batch
(the elements chosen so far are masked out inside the pass), so `k` is at most 32:

```rust
use luma::*;

// Two features; the best two rows of each.
let top = TopK::<i32, Df32>::largest_along(2, 1)?;
top.push(&Array::from_slice(&[1i32, 50, 9, 20, 4, 30], [2, 3])?)?;
top.push(&Array::from_slice(&[7i32, 10], [2, 1])?)?;
// Shape [2, 2]: rank j of feature f at f + 2 j.
assert_eq!(top.values()?.to_vec()?, vec![9, 50, 7, 30]);
assert_eq!(TopK::<i32, Df32>::to_f64(&top.indices()?)?, vec![1.0, 0.0, 3.0, 2.0]);
# Ok::<(), luma::LumaError>(())
```

## Covariance

[`CoMoments`] folds two streams `x` and `y` of one shape (per position, like `Accumulator`);
[`Covariance`] folds the features of one stream against each other: a batch is a
`[features, samples]` array and the state is the mean and the `features x features` matrix of
sums of products. A push is one full-precision `f32` matrix product of the centered batch with
itself (never TF32), and the state is merged in `Df32` or `f64`:

```rust
use luma::*;

let cov = Covariance::<Df32>::new();
// 2 features: the second is twice the first.
cov.push(&Array::from_slice(&[1.0f32, 2.0, 2.0, 4.0, 3.0, 6.0], [2, 3])?)?;
cov.push(&Array::from_slice(&[4.0f32, 8.0], [2, 1])?)?;
let c = Accumulator::<Df32>::to_f64(&cov.cov()?)?;
assert!((c[0] - 1.25).abs() < 1e-6 && (c[1] - 2.5).abs() < 1e-6 && (c[3] - 5.0).abs() < 1e-6);
let r = Accumulator::<Df32>::to_f64(&cov.corr()?)?;
assert!((r[1] - 1.0).abs() < 1e-6);
# Ok::<(), luma::LumaError>(())
```

## Precision

A pushed batch is reduced in `f32` *relative to a pivot*: the running mean of the state (the
first element of the first batch before there is a mean). The sums of `x - pivot` are small and
their rounding errors small, however far from zero the data is; the sums are then added into
a state kept in `Df32` (about 48 bits of mantissa, runs on every device, the default) or `f64`
(CUDA and the CPU backend; `Accumulator::<f64>`). The count, the merge arithmetic and the sum
therefore do not lose digits over `1e9` samples, and `f64` data with a huge offset keeps its
variance (`1e9 + 1e-2 * noise` has a variance within `1e-3` of the exact one, where a plain sum
of squares has none).

What `f32` rounding remains is that of the *batch's* reduction, a few units in the last place
of the spread of the data. `.precise()` runs the pass in the wide type instead, which makes
the sums as exact as the state (at the cost of the wide arithmetic over the batch). It folds the
whole batch only (`Fold::all()`): `Df32` and `f64` arrays have no reductions along an axis, so per-position
states use the default pass. Integer
batches are exact in counts, sums and extrema while `|x - pivot|` and the batch's partial sums
stay below `2^24` (a `precise` pass is exact for all of `u32` and `i32`).

A state of data whose mean drifts far during the stream keeps its accuracy: the pivot follows the
mean at every merge of the recorded pushes.

## Performance

`push` records one reduction kernel over the batch; the reductions of up to
`flush_every(n)` pushes (default 16) are then merged into the state in one more small kernel,
so a push is about 1.6 launches. The cost is the batch's memory traffic, and the batches never
need to be kept: an `[4096, 256]` batch per push (4 MiB) runs at about 330 GB/s on an RTX 3090
(CUDA), with computing the same statistics once from the concatenated array (which needs all the
batches in memory) at about 790 GB/s; luma-bench's `accum` suite and `tools/afcompare`'s `accum`
group measure both and ArrayFire's equivalent.

- Several statistics of one batch share the pass only if they are reductions over the same
  lines: asking for `min_max()` or higher moments adds passes (kurtosis and extrema together are
  about 3.5 passes' worth).
- A batch that is an expression of other arrays is computed inside the pass (see the example
  above), so it costs no extra reads.
- Batches larger than one device buffer are processed in pieces.
- `flush_every` trades launches for memory: pushes between merges keep their batches alive.
