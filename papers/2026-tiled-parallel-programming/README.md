# An Introduction to Tiled Parallel Programming, Simplified

**Arpan Pathak — November 2026 — 30 pages**

This directory holds the author's manuscript exactly as it was written. The PDF
is copied without modification; its SHA-256 is recorded below.

## Files

| File | Purpose |
|---|---|
| [`manuscript.pdf`](manuscript.pdf) | Original 30-page paper, unchanged |

```text
SHA-256  1a8526adf8b81a37ae9bd46aeea2709aac4d5c86f77daf99c78ce6970d816a77
```

## Abstract

The paper derives the tiled matrix-multiplication kernel step by step. It starts
with a `4 x 4` product, introduces threads, blocks, grids, shared memory, and
barriers, proves the tiled kernel correct, and implements it in Rust with
cuda-oxide. On an NVIDIA Jetson AGX Thor at `512 x 512 x 512`, the tiled kernel
takes `0.285 ms` against `1.158 ms` for the direct kernel, a `4.06x` speedup
against a `16x` reduction in modelled global reads. It explains why caches make
the measured speedup smaller than the model, and gives a unified-memory variant
in an appendix.

## Artifact

The runnable code is in [`../../code/gpu-matmul`](../../code/gpu-matmul). The
paper's Appendix A maps to `src/main.rs`; Appendix B maps to
`src/bin/tiled_matrix_mul_unified_memory.rs`. The benchmark mode adds CUDA-event
timing and exact verification so the measurements can be repeated.

## Cite

```bibtex
@misc{pathak2026tiled,
  author       = {Arpan Pathak},
  title        = {An Introduction to Tiled Parallel Programming, Simplified},
  year         = {2026},
  howpublished = {Manuscript and open artifact},
  url          = {https://github.com/arpanpathak/parallel-hardware-and-software-papers}
}
```

## License

The manuscript and teaching material are CC BY 4.0. The code is MIT.
