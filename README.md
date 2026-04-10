# Paper Workspace

This directory holds the paper-facing side of the ActiMM project.

## Layout

- [PITCH.md](PITCH.md): current paper pitch, claims, and evaluation plan.
- [main.tex](main.tex): LaTeX entry point for drafts.
- [src/](src): section files and figure sources.

## Intent

The paper should frame ActiMM as a systems abstraction for sparse execution, not as a
video-specific technique and not as a narrow engineering optimization on top of an existing
sparse library.

The main narrative should stay centered on this question:

How do we design a practical sparse execution model for standard convolutional workloads
that remains beneficial across a much wider active-ratio range than prior sparse paths?
