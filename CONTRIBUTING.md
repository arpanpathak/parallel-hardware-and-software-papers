# Contributing

Thanks for improving this repository. It is a research artifact and teaching
material, so correctness, clarity, and reproducibility matter as much as the
code running.

## Ways to contribute

- **Report a bug.** Include the command, the full error, and the platform record
  described in [`docs/reproducibility.md`](docs/reproducibility.md).
- **Report a benchmark.** Run the teaching binaries on another GPU and open an
  issue with the output and platform. Different results are welcome.
- **Improve teaching material.** New exercises and adoption notes are valuable.
  Explain the learner and the prerequisite knowledge.
- **Fix code or prose.** For a large change, open an issue first. The manuscript
  and the original source files are preserved as written; do not modify them to
  make a teaching point. Add a new example or document instead.

## Setup

Requirements are an NVIDIA GPU supported by cuda-oxide, CUDA, Rust
`nightly-2026-10-11`, and `cargo-oxide 0.2.1`.

```bash
git clone https://github.com/arpanpathak/parallel-hardware-and-software-papers.git
cd parallel-hardware-and-software-papers


cd code/gpu-matmul
cargo oxide doctor
cargo oxide build -- --release
./target/release/tiled_matrix_multiplication
./target/release/tiled_matrix_mul_unified_memory
```

On a machine without a GPU, only the Python check runs.

## Pull requests

1. **Keep it focused.** One bug, one document, or one coherent feature.
2. **Show evidence.** Include the command and its output. For a performance
   claim, include the platform.
3. **Do not edit preserved artifacts.** The manuscript PDF and the three source
   files have recorded hashes; CI fails if they change. Put experiments in a new
   file.
4. **Update the docs.** If behaviour changes, update the README or the relevant
   page in `docs/`.
5. **Avoid unrelated formatting churn.**

## Style

- Rust: `cargo fmt` and `cargo clippy` where the cuda-oxide toolchain permits.
- Kernels: keep the teaching path obvious. Comments explain *why*.
- Prose: short sentences, active voice, American spelling. Define a GPU term the
  first time it appears.
- Measurements: never edit a historical record. Add a new dated file.

## Commits

Use an imperative subject, optionally with a conventional prefix:

```text
feat: add a 32x32 tile experiment
fix: correct a broken documentation link
docs: clarify the unified-memory caveat
```

Sign off with `git commit -s` to certify the
[Developer Certificate of Origin](https://developercertificate.org/).

## Licensing

By contributing, you agree that code is licensed under the
[MIT License](LICENSE) and prose, figures, and teaching material under
[CC BY 4.0](LICENSE-CONTENT). Do not include third-party material unless its
license is compatible and the source is credited.

## Support and conduct

- Questions: open an issue with the `question` label.
- Participation is governed by [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).

Response is best-effort, typically within one to two weeks.
