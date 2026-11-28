#!/usr/bin/env bash
# W3-R7 part 3 (wgpu cooperative matrix): the harness over every coop variant, the generator's unit tests, then the half products on wgpu.
set -u
rc=0
cargo test --release --locked --no-fail-fast -p luma-hal-wgpu --test unit_verify --lib -- 2>&1 | grep -E -A12 "COOPDBG|PROBE|^test |test result|panicked|^error|SKIPPED|variants verified|failed|did not compile" | head -80 || rc=1
LUMA_BACKEND=wgpu cargo test --release --locked --no-fail-fast --test all -- matmul_half:: 2>&1 | grep -E -A12 "COOPDBG|PROBE|^test |test result|panicked|^error|ran a matrix unit" | head -40 || rc=1
exit $rc
