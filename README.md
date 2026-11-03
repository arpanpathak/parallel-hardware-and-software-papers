# Parallel Hardware & Software Papers

Manuscript and original code for *An Introduction to Tiled Parallel Programming,
Simplified*.

## Paper

[`manuscript.pdf`](papers/2026-tiled-parallel-programming/manuscript.pdf) is the
author's 30-page paper, included unchanged.

At `512 x 512 x 512` on an NVIDIA Jetson AGX Thor:

| Kernel | Time | Speedup |
|---|---:|---:|
| Direct | 1.158 ms | 1.00x |
| Tiled | 0.285 ms | 4.06x |

The tiled kernel reads 64 MiB instead of 1024 MiB, a 16x reduction in the model.

## Code

[`code/gpu-matmul`](code/gpu-matmul) is the original Thor working tree, unchanged.

```bash
cd code/gpu-matmul
cargo oxide doctor
cargo oxide build -- --release

./target/release/tiled_matrix_multiplication
./target/release/tiled_matrix_mul_unified_memory
```

Requirements: NVIDIA GPU, CUDA, Rust `nightly-2026-10-11`, and `cargo-oxide 0.2.1`.

## License

Code is MIT. Text and figures are CC BY 4.0.

## Citation

See [`CITATION.cff`](CITATION.cff).
