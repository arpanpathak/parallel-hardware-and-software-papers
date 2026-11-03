<div align="center">

# Parallel Hardware & Software Papers

**Papers with runnable artifacts, not just PDFs.**

[![License: MIT](https://img.shields.io/badge/code-MIT-2ea44f.svg)](LICENSE)
[![Content: CC BY 4.0](https://img.shields.io/badge/content-CC%20BY%204.0-ef9421.svg)](LICENSE-CONTENT)
[![Rust](https://img.shields.io/badge/Rust-nightly-B7410E?logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![CUDA](https://img.shields.io/badge/CUDA-13.x-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)

</div>

---

## What this is

Open companions to papers on parallel hardware and software. The first entry is
**An Introduction to Tiled Parallel Programming, Simplified** — a 30-page
tutorial on GPU matrix multiplication in Rust with
[cuda-oxide](https://github.com/NVlabs/cuda-oxide), measured on an NVIDIA Jetson
AGX Thor.

The manuscript and the Thor source files are included as written. The repository
around them holds the run instructions, licensing, and submission notes.

> At `512 x 512 x 512`, the tiled kernel takes **0.285 ms** against **1.158 ms**
> for the direct kernel: a **4.06x** speedup, against a **16x** reduction in
> modelled global reads. The gap is the lesson.

## Repository map

```text
.
├── papers/
│   └── 2026-tiled-parallel-programming/
│       ├── README.md
│       └── manuscript.pdf       # original 30-page paper, unchanged
├── code/
│   └── gpu-matmul/              # original Thor crate, unchanged
│       └── src/
│           ├── main.rs          # standalone direct kernel
│           └── bin/
│               ├── tiled_matrix_multiplication.rs
│               └── tiled_matrix_mul_unified_memory.rs
└── docs/
    └── reproducibility.md
```

## Build and run

Requirements: an NVIDIA GPU supported by cuda-oxide, CUDA, Rust
`nightly-2026-10-11`, and `cargo-oxide 0.2.1`.

```bash
git clone https://github.com/arpanpathak/parallel-hardware-and-software-papers.git
cd parallel-hardware-and-software-papers/code/gpu-matmul

cargo oxide doctor
cargo oxide build -- --release

./target/release/tiled_matrix_multiplication
./target/release/tiled_matrix_mul_unified_memory
```

`tiled_matrix_multiplication` uses `128 x 64 x 96`, runs the direct and tiled
kernels, prints one end-to-end time for each, and checks all `12,288` entries
against the CPU. `tiled_matrix_mul_unified_memory` is the Appendix B variant at
`512 x 512 x 512`; its first launch includes page migration.

The `512 x 512 x 512` timings in the paper come from the author's CUDA-event
harness, not from the smaller teaching run. See
[`docs/reproducibility.md`](docs/reproducibility.md).

## Manuscript

The original PDF is at
[`papers/2026-tiled-parallel-programming/manuscript.pdf`](papers/2026-tiled-parallel-programming/manuscript.pdf).
It is unchanged:

```text
SHA-256  1a8526adf8b81a37ae9bd46aeea2709aac4d5c86f77daf99c78ce6970d816a77
```

The first pull request adds a separate, short JOSE submission paper and the
EduHPC notes. It does not modify the manuscript or the source.

## Citation

```bibtex
@misc{pathak2026tiled,
  author       = {Arpan Pathak},
  title        = {An Introduction to Tiled Parallel Programming, Simplified},
  year         = {2026},
  howpublished = {Manuscript and open artifact},
  url          = {https://github.com/arpanpathak/parallel-hardware-and-software-papers}
}
```

See [`CITATION.cff`](CITATION.cff).

## Contributing

Corrections and benchmark records from other GPUs are welcome. Read
[`CONTRIBUTING.md`](CONTRIBUTING.md). Participation is governed by
[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).

## License

Code: [MIT](LICENSE). Text and teaching material: [CC BY 4.0](LICENSE-CONTENT).
