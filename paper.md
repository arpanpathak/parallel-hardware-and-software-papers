---
title: "Tiled Parallel Programming, Simplified: A Reproducible GPU Matrix-Multiplication Learning Module"
tags:
  - GPU computing
  - parallel programming
  - CUDA
  - Rust
  - cuda-oxide
  - matrix multiplication
  - shared memory
  - computer science education
authors:
  - name: Arpan Pathak
    affiliation: "1"
affiliations:
  - name: Independent Researcher, Seattle, WA, USA
    index: 1
date: 2 November 2026
bibliography: paper.bib
---

# Summary

Tiling is the first substantial GPU optimization most students meet, and several
ideas arrive at once: a memory hierarchy, cooperative loading, a barrier, a new
indexing scheme, and a model that rarely matches the first measurement. This
module teaches them through one worked example: a hand-checkable `4 x 4` product,
a direct kernel with its `2MNK` reads, and a shared-memory tiled kernel that
stages `16 x 16` tiles, uses two barriers, and reduces modelled reads
sixteenfold. It proves the kernel correct, shows two faulty variants traced to
missing barriers, and implements both in Rust with cuda-oxide, which lowers Rust
to PTX [@cudaoxide]. A unified-memory appendix covers Jetson AGX Thor.

The artifact is a 30-page manuscript and a runnable crate. On Jetson AGX Thor at
`M = N = K = 512`, the measured times are **1.158 ms** for the direct kernel and
**0.285 ms** for the tiled kernel, a **4.06x** speedup against a **16x**
reduction in modelled global reads. The gap is the lesson: caches already answer
many repeated fetches, so tiling buys control over reuse rather than reuse
itself.

# Statement of Need

Parallel and GPU courses need examples that connect four things usually taught
separately: the execution model, the memory hierarchy, synchronization, and
measurement. A vendor GEMM call demonstrates none of them; a production kernel
demonstrates all of them at once. A short, complete, inspectable kernel is the
missing middle.

Comprehensive resources exist. *Programming Massively Parallel Processors* is the
standard textbook [@hwu2022], and NVIDIA's CUDA guide and samples are
authoritative [@nvidia-cuda]. This module offers a self-contained
two-lecture-plus-lab sequence with one running example, exact arithmetic, a
correctness argument, and a measurement reproducible on an affordable edge GPU.

Two choices distinguish it. First, Rust and cuda-oxide are used for both host
and kernels, so students do not switch languages at the GPU boundary. Second, the
model-versus-measurement gap is an explicit objective: students predict the
speedup, measure it, and explain why the hardware disagrees.

The audience is upper-level undergraduate and early graduate students who know
Rust and can multiply matrices by hand, but have no GPU experience. The sequence
supports both self-study and instructor-led adoption.

# Learning Objectives

On completion, a student should be able to:

1. map a row-major matrix to global-memory addresses and to a 2D launch;
2. explain what a block shares, and what it does not;
3. derive the tiled product and count global reads for both kernels;
4. justify both barriers and identify the race each one prevents;
5. measure a kernel with CUDA events and explain a result that differs from the
   model.

The prerequisites are one systems or Rust course and matrix multiplication. No
CUDA C++ experience is assumed.

# Instructional Design

The sequence uses the worked-example effect [@sweller1985]. Every new
abstraction is introduced on the smallest object that can be checked by hand,
then generalized. The `4 x 4` product serves the definition, the storage layout,
the thread mapping, the tile load, the barrier argument, and the correctness
proof before the manuscript states the general case.

Cognitive load is managed by introducing one GPU concept per section and keeping
code and pseudocode adjacent. The barriers are not presented as a recipe. The
text removes each one, shows the wrong value it produces (`c00 = 7` and
`c01 = 15` instead of `11` and `13`), and restores it. Students can reproduce
both failures by commenting out a barrier.

The lab uses a predict-run-check cycle: predict the `4 x 4` result and read
count, run `tiled_matrix_multiplication` and compare all 12,288 entries,
predict the `512 x 512 x 512` speedup, then compare with the measured result and
explain the difference. Students finish by changing the tile edge, predicting,
and measuring again.

Inputs are small integers, so every partial sum is exactly representable in
`f32` and checks use equality. A tolerance would hide indexing errors, which are
among the concepts being taught. The trade-off is that the benchmark does not
exercise floating-point rounding; the artifact documentation states this.

# The Artifact

The repository contains the manuscript and the original crate. The
`tiled_matrix_multiplication` binary runs the direct and tiled kernels at
`128 x 64 x 96` and checks all 12,288 entries against the CPU; the
`tiled_matrix_mul_unified_memory` binary implements the Appendix B variant at
`512 x 512 x 512`. Both exit non-zero on a mismatch. The source is preserved as
written, including `src/main.rs`, the earlier standalone direct kernel.

The toolchain file pins Rust `nightly-2026-10-11`, and `Cargo.lock` records the
tested cuda-oxide commit. The paper's `512 x 512 x 512` measurement uses one
warm-up launch and the mean of five launches bracketed by CUDA events, with 182
sampled entries verified exactly. That event harness is not in the original
working tree, so the repository preserves the teaching code rather than claiming
a new run. The platform record and the method are documented
[@pathak2026artifact].

# Experience of Use and Reproducibility

The module was developed and exercised end to end on an NVIDIA Jetson AGX Thor
developer kit: 14 Arm CPU cores, a Blackwell GPU with 2560 CUDA cores, 128 GB of
shared LPDDR5X, and a quoted DRAM peak of 273 GB/s [@jetsonhacks2025]. The
manuscript reports `1.158 ms` and `0.285 ms`, a `4.06x` speedup. The repository
does not restate these as a fresh benchmark; it keeps the original teaching code
and the manuscript together so the measurement can be inspected and repeated
with the author's event harness.

The module has not yet been evaluated in a formal classroom. It has been used
for self-directed study and to prepare the manuscript, and its correctness has
been tested on hardware. It is ready to pilot, not an evaluated intervention. A
pre/post concept check and a wrong-answer collection plan are the next step.

# Adoption

An instructor can adopt it as one lecture on the `4 x 4` derivation, the
barriers, and the measured result, or as two lectures plus a lab using the
manuscript, all three modes, and the tile-size experiment. A short course can add
register blocking, a vendor-library comparison, and a second GPU.

Code is licensed under MIT and the text and figures under CC BY 4.0.
Contributions of benchmark records, exercises, and adoption reports are welcome;
the contribution guide and issue templates describe the process.

# Limitations and Next Steps

One device supplies the measurements. The unified-memory variant requires shared
physical memory and is not a general replacement for explicit buffers. The kernel
is intentionally simple, with no register blocking, vectorized loads, or vendor
library, and is not a performance ceiling. Next steps are a classroom pilot, a
second GPU generation, and a 4-5 page EduHPC version.

# Acknowledgements

The cuda-oxide project from NVIDIA Labs makes the Rust-to-PTX path possible. The
worked-example structure owes a debt to *Programming Massively Parallel
Processors* and the CUDA education community. Errors that remain are the
author's.

# References
