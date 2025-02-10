---
title: Capture Overview
description: How to capture a dataset with the help of SCAAML
---

For most of our research we leverage the [ChipWhisperer&reg; ecosystem](https://www.newae.com/chipwhisperer).
On top of this we add convenient components.

- **Attack point iterators:** When using deep learning it is very convenient to reason about data which is balanced.
  During profiling attacks we have a full control over the algorithm inputs.
  When evaluating deep learning metrics (e.g., accuracy, top-k accuracy, mean rank) it is much easier to reason about the results when the classes are completely balanced.
  SCAAML provides support for generators inspired by [latin squares](https://en.wikipedia.org/wiki/Latin_square).
- **Capture resume:** A long lasting capture might easily crash for many reasons such as a power outage.
  When a crash happens the generated inputs by attack point iterators would not be totally balanced.
  We build our tooling to be resistant to crashes and thus resumable.
  We first save all inputs of the algorithm of interest and then iterate those while automatically saving progress.
- **Capture context-managers:** After a crash of a capture campaign we might need to release resources (e.g., call `scope.dis()`).
  Python context-managers help us to ensure resources are always released even in the case of capture in a script running on remote computer.
- **Oscilloscopes:** In our research we use different oscilloscopes.
  We build an API compatible with `cw.capture_trace`.
- **Saving data:** How to save the captured data.
- **Statistical tools:** Finally we provide basic statistical tools.
