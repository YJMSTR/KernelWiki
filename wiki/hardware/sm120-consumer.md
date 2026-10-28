---
id: hw-sm120-consumer
title: "SM120 / SM121 — Consumer Blackwell"
type: hardware
architectures: [sm120, sm120a, sm121]
tags: [nvfp4, block-scale, gemm]
confidence: source-reported
related: [hw-nvfp4, hw-sm103-b300, kernel-deepgemm, lang-triton]
sources: [pr-DeepGEMM-324, pr-DeepGEMM-447, pr-flashinfer-1608, pr-tilelang-3286, pr-tilelang-3430, pr-triton-11862]
aliases: ["SM120", "SM121", "consumer Blackwell"]
---

# SM120 / SM121 — Consumer Blackwell

SM120 (with the `sm120a`/`sm120f` specialization suffixes) and SM121 are the consumer-family Blackwell targets, distinct from the datacenter SM100/SM103 parts this wiki centers on. They appear in the retained corpus as first-class block-scaled-FP GEMM targets with their own integration and scheduling constraints.

## Ecosystem support retained in the corpus

- **DeepGEMM** added SM120 support with `sm120`, `sm120a`, and `sm121` architecture guards ([pr-DeepGEMM-324](../../sources/prs/DeepGEMM/PR-324.md)) and later integrated the SM120 kernel set into main while preserving `nv_dev` features ([pr-DeepGEMM-447](../../sources/prs/DeepGEMM/PR-447.md)).
- **TileLang** built a block-scaled MMA line for these parts: an SM120 block-scaled register-A GEMM that unrolls K atoms and loads scales once per atom ([pr-tilelang-3286](../../sources/prs/tilelang/PR-3286.md)), and enablement of the SM120 block-scaled MMA path on SM121 ([pr-tilelang-3430](../../sources/prs/tilelang/PR-3430.md)). The surrounding series (MXFP8/MXFP4 and `mxf4nvf4` variants) is represented in the ledger even where individual PRs are still open.
- **Triton** carries an explicit capability boundary: the TMA scatter path is not taken on `sm_12x` ([pr-triton-11862](../../sources/prs/triton/PR-11862.md)). Backend paths tuned for SM100 must be re-checked against such guards rather than assumed.
- **FlashInfer** added initial support for SM103, SM110, SM120, and SM121 in one step ([pr-flashinfer-1608](../../sources/prs/flashinfer/PR-1608.md)).

## Selection guidance

The consumer parts share the block-scaled FP4/FP8 GEMM surface with the datacenter parts, but resource envelopes and backend coverage differ: the retained sources show register-A GEMM formulations on TileLang and an excluded TMA scatter path in Triton for `sm_12x`. Validate any SM100-oriented kernel against the per-target guards before assuming parity, and prefer the MXFP4/NVFP4 block-scaled entry points that the ecosystem actually ships for these parts.

## Sources

- [DeepGEMM PR #324 — add sm120 support](https://github.com/deepseek-ai/DeepGEMM/pull/324)
- [DeepGEMM PR #447 — integrate SM120 kernels into main](https://github.com/deepseek-ai/DeepGEMM/pull/447)
- [FlashInfer PR #1608 — initial SM103/SM110/SM120/SM121 support](https://github.com/flashinfer-ai/flashinfer/pull/1608)
- [TileLang PR #3286 — SM120 block-scaled register-A GEMM](https://github.com/tile-ai/tilelang/pull/3286)
- [TileLang PR #3430 — enable SM120 block-scaled MMA on SM121](https://github.com/tile-ai/tilelang/pull/3430)
- [Triton PR #11862 — avoid TMA scatter path on sm_12x](https://github.com/triton-lang/triton/pull/11862)
