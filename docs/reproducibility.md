# Reproducibility

## Reference platform

| Item | Value |
|---|---|
| GPU | NVIDIA Jetson AGX Thor, compute capability 11.0 (`sm_110a`) |
| Driver | 595.78 |
| CPU | 14-core Arm, aarch64 |
| Memory | 128 GB LPDDR5X, shared by CPU and GPU |
| OS | Ubuntu 24.04.4 LTS, kernel 6.8.12-1021-tegra |
| Rust | 1.99.0, `nightly-2026-10-11` |
| cuda-oxide | 0.2.1, commit `e027ac3` |

## Build and run

```bash
cd code/gpu-matmul
cargo oxide doctor
cargo oxide build -- --release

./target/release/tiled_matrix_multiplication
./target/release/tiled_matrix_mul_unified_memory
```

`tiled_matrix_multiplication` runs the direct and tiled kernels at
`128 x 64 x 96`, prints one end-to-end time for each, and checks all `12,288`
entries against the CPU. `tiled_matrix_mul_unified_memory` runs the Appendix B
variant at `512 x 512 x 512` and checks four entries.

The first unified launch includes page migration and can take seconds; later
launches do not.

## Manuscript measurement

The `512 x 512 x 512` values in the paper are `1.158 ms` direct and `0.285 ms`
tiled, a `4.06x` speedup. They were measured with CUDA events as the mean of five
launches, with 182 sampled entries verified exactly. The event harness is
described in the manuscript but is not part of the original working tree copied
here. The values below are the manuscript's record; no separate benchmark folder is included in this first import.

## What to expect

The teaching binary uses a smaller, non-square shape, so its printed times are
not the paper's table. Treat all numbers as device- and clock-dependent
observations, not specifications.

## Reporting a different result

Include the GPU, driver, OS, Rust and cuda-oxide versions, the command, and the
full output. Add a new dated file under `benchmarks/`; do not edit an existing
record.

## Known pitfalls

- Over SSH, use a login shell so `cargo` is on `PATH`.
- The first unified-memory launch can be slow because of page migration.
- A host-to-device copy can occasionally take much longer under memory pressure.
- `cargo oxide build` warns when `--arch` is omitted but detects the local GPU.
