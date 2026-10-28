# Blackwell upstream refresh survey — 2026-10-27

> Research note (not yet ingested). Scope: **Blackwell-related updates only**
> (SM100/B200/GB300, SM103/B300, SM120 consumer), per maintainer request.
> Rubin/SM107 items appear only where they ship inside the same releases and
> affect the shared Blackwell codebase.
>
> Evidence boundaries of this survey: upstream checked live on **2026-10-27
> (UTC)** against GitHub release/tag/PR-list pages and the CUDA toolkit
> release-notes page. This extends the repository's existing cutoffs
> (PR corpus 2026-05-20; docs/versions 2026-08-18) by ~5 months of PR activity
> and ~2.5 months of release activity.
>
> Method caveat: PR lists were read from GitHub web search pages (first page,
> sorted by recent update) and the GitHub REST API where noted. `merged` marks
> items whose merge was explicitly observed; unmarked items were visible as
> recent activity (often still open) and must be re-verified before ingestion.
> OR-queries on the web UI also loosen the `is:merged` filter, so treat
> per-repo "total" counts as approximate.

## 1. Tool / release versions newer than `data/tool-versions.yaml` (2026-08-18)

| Tool | Wiki snapshot | Latest observed (2026-10-27) | Blackwell-relevant delta |
|---|---|---|---|
| Triton | 3.7.1 (2026-06-18) | **3.8.0** (2026-08-28) | two-CTA mode support, MMAv5/scaled-dot additions, generic multi-CTA (layout conversion, reductions, gather/scatter, TMA, multicast), `tma.store_wait(read_only=)`, Gluon TMEM predication + accumulator-release fixes, FpSan/GSan/ConSan sanitizers, `@triton.aggregate` public API, `tl.topk(descending=)`. Initial Rubin module (see §5). |
| CUTLASS | 4.7.0 (2026-08-13) | **4.8.1** (2026-10-21); 4.8.0 (2026-09-22); 4.8.0dev (2026-08-27); 4.7.1 in between | CuTe DSL: automatic CTA-V map inference, async atomic TMA reduce-store, opt-in TMEM ping-pong accumulator planning; new Blackwell CuTe-extension examples (dense/blockscaled GEMM, GQA decode attention, back-to-back GEMM, **CLC-scheduler persistent GEMM**); Operator API nvMatmulHeuristics ranking; `tensormap_replace` device-side TMA descriptor shrink fix (Blackwell); SM103 treated as SM100-family in CMake; eight Blackwell examples build for `sm_103a`. |
| CUDA | 13.3 U1 (2026-06-29) | **13.4 GA + Update 1 + Update 2** (notes page updated 2026-10-12) | **PTX ISA 9.4**; Nsight Compute 2026.3.1.2 / Nsight Systems 2026.3.2; first Linux release where the driver is no longer bundled (R615 branch). cuBLAS 13.4 on Blackwell: FP64 emulation via Ozaki-II up to 175 TFLOPS DGEMM / 295 TFLOPS ZGEMM on B200; experimental MXFP8 scale layouts (VEC32_MN_K4 / VEC128_MN_K4 UE8M0); grouped-GEMM dynamic scheduling up to +20% on Blackwell; U1 fixed grouped-GEMM NVFP4 wrong results on B300; known issue: grouped GEMM incorrect results on Blackwell/Rubin introduced by cuBLAS 13.7.0 (bug 6798616). |
| PyTorch | (pre-2.14) | **v2.14.0 / v2.14.1** | Linux binaries on CUDA 13.2.2 (PR #198644, merged). |
| flash-attention | FA4 paper/blog era | stable **v2.8.3.post2**; **FA4 pre-release line `fa4-v4.0.0.beta36`** (beta32→36) | The CuTe/SM100 FA4 implementation is now a versioned pre-release series — a major version anchor for `wiki/kernels/flash-attention-4.md`. |
| FlashInfer | (pre-0.7) | **v0.7.2.post1** (rc3→rc6 series) | see §2 FlashInfer PRs (cake_* family). |
| TileLang | (pre-0.1.15) | **v0.1.15** | SM120 block-scaled MMA line, see §2. |
| TensorRT-LLM | (v1.x era) | **v1.4.0rc1** (v1.3.0 series ran to rc29) | see §2. |
| CCCL | (3.x era) | **v3.5.0** (+ python-1.2.1 tag) | sm100 tuning for DeviceReduce/DeviceRadixSort, see §2. |
| DeepGEMM | (unversioned sync PRs) | **v2.1.1.post3** + rolling `nv_dev_*` tags | Public-release syncs merged 26/09/30 (#462) and 10/14 (#471); vendored CUTLASS moving to v4.8.0 (PR #464, open). |
| TileKernels | (1 PR at cutoff) | **tile-kernels v2.0.0 / v2.1.0** | Repo went from ~1 PR to 24 open/7 closed; see §2. |

## 2. Blackwell operator/kernel PR activity since 2026-05-20

### CUTLASS
- PDL correctness series (Oct 24–25): #3786 read epilogue fusion scalars after
  the PDL wait; #3787 wait on previous grid in every loading warp; #3788 build
  legacy epilogue thread op after the PDL wait; #3781 count the epilogue-load
  warp in the SM100 mixed TMA/cp.async GEMM scheduler pipeline.
- #3779 build and run eight Blackwell examples for `sm_103a` (Oct 24);
  #3754 treat `103f` as SM100-family in CMake (Oct 19); #3767 keep SM90
  heuristic kernels when SM100 is co-targeted (Oct 21).
- UMMA/cluster/TMEM ordering fixes (Sep 21–22): #3661 guard SM100 UMMA peer
  arrival mask on cluster v-extent; #3647 fix cross-CTA barrier scope + missing
  SFB wait in `2cta_mma_blockscale_m128` example; #3652 wait `Acc2` tcgen05.ld
  before releasing TMEM to the MMA warp; #3653/#3654 example-77 TMEM statistics
  and producer_acquire ordering.
- #3637 enable `CUDA_CTA_RECONFIG_ACTIVATED` for SM110a (Jetson Thor) — merged Oct 15.
- #3793 reject MN-major 32-bit operands the SW128_32B smem atom cannot tile
  (CuTeDSL, Oct 27).

### Triton
- **#10676 "[TRITONGPU] Support two-cta mode" — merged Oct 26**; companion
  #12436 two-CTA MMA for `dot_scaled` (open); #10812 two-CTA warp
  specialization (NVWS, open). Two-CTA/tcgen05 2-SM MMA is the headline
  Blackwell compiler story since 3.8.0.
- Warp specialization & TMEM pipeline fixes (merged Oct 20–27): #12461 wait for
  scale reads before reusing TMEM; #12476 fold only uniform select conditions
  into TMEM stores; #12402 Gluon predicate TMEM copies; #12400 Gluon
  accumulator release / reject partial attention tiles; #12472 partition
  `tt.scan` in automatic WS; #12482 async-MMA consumer discovery in `scf.if`;
  #12498 fold warpid for single-warp partitions.
- #12463 fix Blackwell FP4 lhs shared-memory budgeting (open).
- #12355 "require PTX 8.7+ for Blackwell" — closed without merge (relevant to
  PTX-version claims; verify disposition before citing).

### flash-attention (CuTe / SM100 FA4 line)
- Scheduling/config overhaul (Jul–Oct, mostly open series): typed forward/backward/MLA configs (#2733/#3044/#3054), explicit autotuning + benchmarks (#2735, #2743), **B200** (#2741) and **GB300** (#2738) forward-scheduling tunes, forward smem layout in config (#3056).
- CLC (cluster launch control): #3062 run causal forward on CLC by default (Oct 20); #2976 L2 swizzling for CLC scheduling + preserve STATIC 2CTA (Oct 5); #3034 fix 2CTA CLC tile coords — **merged Oct 24**.
- hd256 backward unification series: #2967 merged Oct 23; #3059, #2968 open.
- Feature work: #2877 symmetric D512 fwd+bwd (Sep 10); #2985 `kv_start` causal document masking without BlockMask (Oct 10); #3060 extend paged-KV support SM90→SM80/SM120 (Oct 20); #3046/#3049 block-sparse fwd/bwd refinements; #3008 wait for O TMEM loads before releasing O (Oct 16).

### DeepGEMM
- Periodic public-release syncs: #462 (26/09/30, merged), #471 (10/14, merged); version tags now at **v2.1.1.post3**.
- #464 update vendored CUTLASS to v4.8.0 (open) — will pull the 4.8 Blackwell stack under DeepGEMM.
- SM100: #480 use allocated TMEM base in SM100 HC prenorm GEMM (open).
- SM120 (consumer Blackwell) line: #447 integrate SM120 kernels into main (merged Sep 23); #403 scale-factor layout fix, #406 heuristic calibration harness, #475 split-K alignment after partition caps.
- MQA logits / sparse: #459 sparse MQA-logits follow-ups (merged Sep 24); #485 (SM90) reduce H32 logits heads per-thread (open); #477 reject `c` operand in SM90 fp8_einsum to prevent silent corruption (open).
- #473 fused FP8×MXFP4 MegaMoE for SM90 (open); #466/#421 `pack_ue8m0_to_int` CUDA-graph capture safety.

### FlashInfer — new `cake_*` Blackwell kernel family
- Merged Oct 26–27: #7550 `cake_mm_fp4` cluster-split per-token quantizer + token-remainder GEMM folds on B200/GB300; #7463 `cake_blackwell_bf16_fp4` relaxed tcgen05 entry cluster arrive; #7515/#7569 `cake_fused_qk_rope_fp8_append` sm_100a/sm_103a grid_order stage regeneration; #7555 `cake_gdn` faster backward prep / dqkg / adjoint-walk tape; #7537 `cake_blackwell_msa` TU dedup; #7528 `cake_batch_deepgemm_fp8` masked BLOCK_N + two-CTA large-N/K builds.
- Open: #7568 `cake_gated_act_mxfp8` register-fitted band-path + packed sigmoid; #7453 `cake_kimi_k3_attn_res` K=3 packed register carry, 384-thread M=192 pair programs (SM100/SM103).
- MLA / sparse attention: #6536 GLM5.3 1CTA token-sparse (DSA) MLA decode with FP8 KV (open); #7410 NVFP4 + per-tensor FP8 sparse MLA caches (open); #7325 selective FP8 paged MQA TopK; #6423 SDSA NVFP4 keys in MQA logits + paged cache (open); #7574 zero-fill CUTLASS MLA V rows past kv_len (open).
- MoE: #7523 Blackwell standard + GenPhase autotuning v2 (open); #7556 bit-exact warp-per-row non-fused finalize for CUTLASS fused MoE (open); #7290 split CuTe DSL MoE with activations on B + dynamic MMA N (open); #6542 expert-parallel tile_N candidates for TRT-LLM-gen autotuner; #6835 fuse BF16 staging + trim SM90 MXFP4 combine.

### TileLang — SM120 (consumer Blackwell) focus
- Block-scaled MMA on SM120: #3430 enable on SM121 — **merged**; #3286 block-scaled register-A GEMM with K-atom unroll + single scale load — **merged Oct 26**; #3081 `mxf4nvf4` 2X/ue8m0 and 4X/ue8m0 (open); #3495 SM120 MXFP8/MXFP4 block-scaled MMA (open); #3099 `kind::mxf8f6f4` full operand family + sub-byte TMA producers (open); #3290 SM120 W4A16 NVFP4 GEMM for decode-sized batches (draft).
- SM100: #2774 SM100 MQA-logits kernels + native lowering; #2360 Flash-Attention 1SM/2SM kernels; #2316 `T.tma_copy` barrier_rank + use_2cta fold; #3334 decode reserved NVFP4 UE4M3 NaN scale — **merged**; #3365 TMA gather/scatter constant-row-index fix — **merged Oct 27**.

### quack (Dao-AILab CuTe DSL)
- **#"[SM103] Use the K=96 FP4 MMA on B300 for nvfp4/mxfp4 blockscaled GEMM"** (open) — B300/SM103-specific MMA shape.
- Blackwell perf: RMSNorm/LayerNorm backward fix for 3–24x collapse on wide rows (bf16 N≥96k, fp32 N≥64k); Softmax backward 1.2–1.5x on 16-bit rows >32k.
- PDL: wait for PDL dependencies before grouped tile decoding on SM90 and SM100.
- Rounding documentation: 8-rbit truncation bias + `sm_103a` 16-rbit `cvt.rs`.
- [Norm] share forward kernels across N with same tile width; [Gemm, SM90] decoupled `seqused_k` for varlen_k.

### TileKernels (deepseek-ai)
- **tile-kernels v2.0.0 and v2.1.0 released** (repo now 24 open / 7 closed PRs vs 1 at the May cutoff).
- New kernels: Group-128 Fused Add RMSNorm FP8 Cast; Fused RMSNorm FP8 Quant; extended CUDA Rotary Embedding dtypes/dimensions.
- Fixes: sinkhorn `step_reduce` shared-memory budgets on SM120; MoE multi-GPU launch-site guards/device resolution; MUSA backend support (TileLang target selection + ue8m0 guard); ROCm HIP compile of mHC pre big-fuse kernel; tilelang import compatibility.

### PyTorch
- #199822 [Inductor] autotune TMA vs pointer loads for pointwise/reduction kernels (open Oct 5) — TMA codegen on Blackwell.
- #201920 [Reland][Inductor] persistent-TMA feasibility check without dynamic-dim specialization (open Oct 24).
- #199022 `F.quantize_tensor` / `F.quantize_tensor_dual` (open); #201851 scaled_mm v1 bias-shape validation (open); #197775 WOQ quantization fusions extensible to third-party backends (open).
- v2.14.x ships CUDA 13.2.2 Linux binaries.

### vLLM (kernel-relevant, merged unless noted)
- #63101 [Perf][MXFP4] activation-quant kernel 3x faster, +5% throughput — merged Oct 26.
- #61685 locality-domain allocation for FlashInfer CuTeDSL NVFP4 — merged Oct 26.
- #51724 [Attention][DSA] enable W4A16 DSA — merged Sep 1.
- #63920 Muse Spark NVFP4 MegaMoE input-prep JIT warmup — merged Oct 27.
- #62026 register OOT block-scaled FP8 linear kernels — merged Oct 26.
- #48167 fix FlashInfer non-causal draft attention (DFlash/DSpark) on Blackwell — merged Jul 15.
- #62191 [Kernel][MoE] FlashInfer CuTe-DSL W4A16 NVFP4 experts — open Oct 17.

### SGLang (kernel-relevant)
- #45787 paged-experts: AWQ / compressed-tensors / **MXFP4 / NVFP4** / W8A8 / block-INT8 — merged Oct 25.
- #46141 MSA-v1 sparse backend + **NVFP4 KV pool** — merged Oct 25.
- #46082 split full-attention decode KV across CTAs at large batch (trtllm_mha) — open.
- #46024 `hc_combine_norm_mxfp8` batch-split rows + zero-padding scales — open.
- #44750 explicit CuTe DSL W4A16 MoE for nemotron-h — open.
- #46809 opt-in FP8 KV dequant for eager prefill — open.
- (kimi-k3 SM107 GEMM/TGV/KDA PRs #46464–#46468 merged Oct 25–26 are Rubin-targeted; listed only because they share the kernel framework.)

### TensorRT-LLM (merged bulk-landing PRs)
- #20942 FP8 block-scale low-M GEMM for decode projections (SM100/103 tactic tables).
- #20755 GLM-5.3-Flash serving speedups on Blackwell.
- #20937 Qwen3.8-Flash-Next NVFP4 modeling: GB300 Pareto optimizations (ISL 8K / OSL 1K).
- #20815 Qwen3.5 GB300 upstream of AgentX side-branch optimizations.
- #21101 re-enabled CuteDSL NVFP4 MoE backend tests (large intermediate_size).
- #21011 Kimi K3 attn_res persistent fused kernel: 64-bit candidate offsets fix.
- #20833 phase-dependent MLA heads for Helix prefill.

### CCCL
- #12052 tune deterministic DeviceReduce for sm100 with 64-bit offsets — **merged Oct 26**; #12053 tune DeviceRadixSort pairs for sm100 (64-bit K/V/offsets) — **merged Oct 22**.
- #12252 mbarrier-based L2-free cluster sync in DeviceBatchedTopK — open.
- #9281 WarpBitonicTopK — open (since Jun).
- Vectorized-load machinery series: #12042 (libcu++), #12049 (fancy iterators over contiguous storage), #12213 (BlockLoad/WarpLoad).
- v3.5.0 released.

## 3. Themes worth a wiki refresh round

1. **CLC / persistent scheduling goes mainstream**: FlashInfer cake_* family,
   FA4's CLC-by-default causal forward (#3062), CUTLASS 4.8 CLC-scheduler
   persistent GEMM example, FA4 L2-swizzled CLC scheduling (#2976). The
   wiki's `cluster-launch-control` coverage can now cite shipping defaults,
   not just examples.
2. **Two-CTA (2-SM) tcgen05 MMA maturing across stacks**: Triton #10676
   merged + `dot_scaled` two-CTA in flight; FA4 STATIC 2CTA preservation;
   FlashInfer two-CTA large-N/K DeepGEMM builds; TileLang `use_2cta` folding.
3. **SM103/B300 is a first-class target now**: FlashInfer sm_103a stages,
   quack K=96 FP4 MMA on B300, TRT-LLM SM103 tactic tables, CUTLASS sm_103a
   example builds, DeepGEMM/vLLM/sglang B300 mentions. The wiki's SM100
   pages should grow an SM103 delta note (576-col TMEM / 328KB SMEM already
   appear in CUTLASS 4.8 notes for the shared codebase).
4. **SM120 consumer-Blackwell kernel line**: TileLang mxf4nvf4/mxf8f6f4 MMA,
   DeepGEMM SM120 integration, FA4 paged-KV on SM120, TileKernels SM120
   smem fits. Mostly absent from the current wiki.
5. **NVFP4/MXFP4 everywhere**: FlashInfer cake_mm_fp4, vLLM MXFP4 quant 3x,
   SGLang NVFP4 KV pool + paged experts, TRT-LLM Qwen NVFP4 GB300 tuning,
   TileLang SM120 nvfp4 GEMM, quack B300 K=96 FP4 MMA, cuBLAS MXFP8 scale
   layouts. The nvfp4 pages could cite many more shipping instances.
6. **Version anchors to update**: Triton 3.8.0, CUTLASS 4.8.1, CUDA 13.4
   (PTX ISA 9.4), PyTorch 2.14.x, FA4 `fa4-v4.0.0.beta36` pre-release line,
   FlashInfer 0.7.2.post1, TileLang 0.1.15, TRT-LLM v1.4.0rc1, CCCL 3.5.0,
   DeepGEMM v2.1.1.post3, TileKernels v2.1.0.

## 4. Suggested next steps (not performed in this survey)

- Run a proper `refresh_candidate_ledger.py` round with `cutoff_date`
  2026-10-27 (or later) so candidates/*.yaml `searched_at` fields and
  sources/prs pages regenerate through the audited pipeline rather than from
  this note.
- Update `data/tool-versions.yaml` + `tool-release-receipt.yaml` from the
  release-list receipt for the new versions above.
- Add an SM103/B300 delta page and consider an SM120 scope decision
  (include as Blackwell-family or defer).
- FA4 page: pin the `fa4-v4.0.0.betaNN` pre-release line as the versioned
  artifact backing the CuTe SM100 implementation.

## 5. Out-of-scope observations (context only)

- Rubin/SM107 initial support landed in Triton 3.8.0 and CUTLASS 4.8.x
  (requires CUDA 13.4 GA / R615 driver). CUTLASS 4.8.0 notes describe 576-col
  TMEM and 328KB SMEM as Rubin capabilities; treat as Rubin-scope, not SM100.
- DeepSeek-V4.x, Kimi K3, GLM-5.3-Flash, Qwen3.5/3.8 model-serving PRs appear
  throughout vLLM/SGLang/TRT-LLM results; only their kernel-level payloads
  were retained above.
