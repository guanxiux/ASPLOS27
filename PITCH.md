# ActiMM Paper Pitch

## Working title

ActiMM: Active-Only Implicit Execution for Practical Sparse Convolution

## One-paragraph pitch

Existing sparse operator stacks for convolutional workloads often deliver clear speedups
only at very high sparsity. In the more practical middle regime, the overhead of coordinate
management, data packing, materialization, and irregular execution can erase the benefit of
skipping inactive work, causing sparse execution to converge back to dense performance or
even become slower. ActiMM proposes an alternative systems abstraction, active-only implicit
execution, that keeps the execution pattern close to dense implicit GEMM while computing and
writing back only active outputs. The goal is not just to support sparse execution, but to
substantially widen the practical profitability region of sparse convolution on standard
vision workloads such as YOLO.

## Problem statement

The real systems problem is not whether sparse convolution can ever be faster than dense.
It clearly can. The harder and more important question is whether a sparse operator stack
remains profitable across the active-ratio regimes that appear in realistic workloads.

Many prior paths appear to have a narrow sweet zone:

- they can be excellent at very high sparsity;
- they degrade rapidly as the active ratio rises;
- they often become neutral or negative before reaching the moderate-density regimes that matter in practice.

ActiMM targets that gap.

## Core abstraction

ActiMM treats `dense tensor + mask` and `point cloud` as two encodings of the same active-set
object. Its abstraction is execution-centric rather than format-centric:

- represent the input as an active set;
- map computation in a way that resembles dense implicit GEMM;
- avoid gather-then-densify as the main execution path;
- compute only active work;
- write back only active outputs.

## Why this can be a systems paper

This is not only about arithmetic reduction. The contribution lives in the execution model:

- how active work is mapped to compute;
- whether intermediate materialization is required;
- how memory traffic scales with active ratio;
- how writeback is organized;
- how runtime overhead behaves outside the extreme-sparsity regime;
- whether the sparse path stays stable and useful on standard models.

That makes the paper naturally systems-oriented.

## Early evidence we already have

From the existing FluxShard-side profiling and debugging, a few observations already look stable:

- sparse operator efficiency, not just sparsity itself, strongly determines whether sparse inference pays off;
- native DeltaCNN-style execution appears to have a relatively narrow profitable zone on standard workloads;
- our own sparse operator path shows a noticeably wider profitable region;
- these observations are visible on standard YOLO-family workloads rather than only on toy kernels.

This is enough to justify a standalone operator/runtime paper direction.

## Main claims to validate

- ActiMM defines a new sparse execution abstraction rather than only a new kernel implementation.
- The abstraction is representation-agnostic: masked dense tensors and point sets are two views of the same active-set problem.
- ActiMM stays closer to implicit dense execution than gather-then-dense sparse pipelines.
- ActiMM widens the practical sweet zone of sparse execution.
- The effect holds on standard vision models, not just handcrafted microbenchmarks.

## Initial evaluation plan

- Baselines:
  - dense implicit execution;
  - DeltaCNN-style sparse execution;
  - additional sparse baselines if a fair comparison path exists.
- Workloads:
  - standard YOLO-family models first;
  - possibly segmentation and pose variants after the base story is stable.
- Metrics:
  - latency versus active ratio or sparsity ratio;
  - dense crossover point;
  - width of the profitable zone;
  - memory traffic and runtime overhead when possible;
  - robustness across feature shapes and hardware.

## What not to undersell

- This should not be pitched as just a FluxShard subcomponent extracted into a new repo.
- This should not be framed as a narrow fix for DeltaCNN.
- The point is not merely lower gather overhead.
- The point is a different execution model that changes where sparse execution is worthwhile.

## Short abstract sketch

Sparse convolution is attractive in principle, but existing operator stacks often provide
meaningful acceleration only at extreme sparsity. In more practical regimes, overheads from
format conversion, packing, irregular memory access, and sparse execution management can
eliminate the benefit of skipping inactive work. We present ActiMM, a representation-agnostic
execution abstraction for sparse convolutional workloads. ActiMM follows an active-only
implicit execution model that preserves key advantages of dense implicit GEMM while avoiding
the cost structure of gather-then-densify sparse pipelines. Across standard vision workloads,
including YOLO-family models, ActiMM substantially widens the active-ratio regime in which
sparse execution remains beneficial, demonstrating that the central bottleneck is not only
how much work is sparse, but how that sparse work is executed.

