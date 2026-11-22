//! Every integration test in one binary: the tests run as threads of one process, so the device is opened once.
//! `flush` (wall-clock timing) and `binding_limit` (sets an environment variable) own their processes.
#![recursion_limit = "256"]
// `keccak` and `mlkem` each load the example they test, and the mlkem example loads the keccak one: two copies, as in separate binaries.
#![allow(clippy::duplicate_mod)]

mod common;

mod argmax;
mod array_ops;
mod assign;
mod axis_reduce;
mod backends;
mod batching;
mod bitwise;
mod bmm;
mod bool;
mod bool_packed;
mod bykey;
mod check_finite;
mod chunked;
mod classify;
mod clone;
mod complex;
mod cpu;
mod df32_library;
mod elementwise_math;
mod errors;
mod euler_residual;
mod factorize;
mod fields_generic;
mod fft;
mod filter;
mod fusion;
mod generators;
mod group;
mod integers_i32;
mod iterate_all;
mod iterate_until;
mod iterate_until_all;
mod keccak;
mod kernel;
mod kernel_edges;
mod lazy_scalars;
mod matmul;
mod matmul_tf32;
mod memory_stats;
mod mlkem;
mod multi_reduce;
mod nan_reductions;
mod non_blocking_eval;
mod ops;
mod precision;
mod profile;
mod profile_drain;
mod quantile;
mod records;
mod scan;
mod scatter_add;
mod search;
mod segments;
mod smallint;
mod sort;
mod spill;
mod threads;
mod transfer;
mod tridiagonal;
mod vector_reductions;
mod views;
mod image;
mod small_math;
mod structure;
mod stats;
