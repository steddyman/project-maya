# Mac Pro 2019 on Linux: Radeon Pro Vega II Duo (gfx906)

This fork ([steddyman/project-maya](https://github.com/steddyman/project-maya)) adds kernels and defaults for Vega 20
(`gfx906`) to the DeepSeek-V4.1-Flash engine (`maya-ds4`). They were written and measured on one machine:

| | |
| --- | --- |
| Computer | Mac Pro 2019 (MacPro7,1) |
| GPU | Radeon Pro Vega II Duo: two Vega 20 (gfx906) dies, 32 GB HBM2 each |
| CPU | Intel Xeon W-3245, 16 cores / 32 threads, AVX-512 (Cascade Lake) |
| Memory | 192 GB DDR4-2933 |
| OS | Ubuntu 26.04 with the t2linux kernel, the kernel's own `amdgpu` driver |
| ROCm | 7.14 for gfx906 from TheRock's nightly wheels |

Other gfx906 cards (Instinct MI50 / MI60, Radeon VII) take the same code paths, but nothing below was measured on
them. [MI50_VALIDATION.md](MI50_VALIDATION.md) is the upstream page for the MI50 source build.

The kernels come from the same work on the Strata engine
([its Mac Pro page](https://github.com/steddyman/Strata/blob/main/docs/MAC_PRO_2019_LINUX.md)), which shares this
engine's DeepSeek code. Only DeepSeek-V4.1-Flash is covered here. GLM-5.3-Flash, GLM-5.3 and MiniMax-M3.1-Flash build
and are unchanged; they were not measured on this machine.

## Before you start: let the GPUs clock up

The amdgpu driver has a power level for each card. Pinned at `low`, the dies run at a fraction of their speed. Check it,
and set it to `auto` (as root) if it says `low`:

```bash
cat /sys/class/drm/card*/device/power_dpm_force_performance_level
```

## Build

A HIP build for gfx906 alone turns the gfx906 kernels on (`STRATA_HIP_GFX906`, set by `cmake/hip_backend.cmake`). A
build for any other card, or for several architectures at once, is the same as upstream's.

```bash
cmake -S . -B build-gfx906 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=$ROCM_PATH/llvm/bin/clang -DCMAKE_CXX_COMPILER=$ROCM_PATH/llvm/bin/clang++ \
  -DCMAKE_HIP_COMPILER=$ROCM_PATH/lib/llvm/bin/clang++ -DCMAKE_HIP_ARCHITECTURES=gfx906 -DCMAKE_HIP_PLATFORM=amd \
  -DBUILD_SHARED_LIBS=OFF -DSTRATA_BUILD_TESTS=OFF -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=ON \
  -DSTRATA_PREFILL_MMQ=ON -DSTRATA_GGML_DIR=/path/to/pinned/llama.cpp -DSTRATA_NATIVE_EXPERTS=ON \
  -DSTRATA_PLE_FIXTURE_DIR=$PWD/bench/micro -DSTRATA_PORTABLE=OFF -DSTRATA_WERROR=OFF
cmake --build build-gfx906 --target maya maya-ds4 --parallel
```

The configure step prints `gfx906-only HIP build - the gfx906 kernels are on` when the switch is set.

## What it does, and how fast

All numbers: `maya-ds4` started directly, both dies (`--gpus 0,1`), no drafter (`--dspark none`), the ISTA-DAS 2.00 bpw
GGUF set (IQ1_M / IQ2_XXS / IQ2_XS experts, Q5_K dense weights), power level `auto`, 14 November 2026. One run each
unless a range is given. "Before" is upstream v1.1.63 built the same way.

| Test | Before | This fork |
| --- | ---: | ---: |
| Decode, 256 tokens after a 48-token prompt | 21.9 tok/s | **29.8, 30.5 tok/s** |
| A 3,000-token prompt | 108 tok/s | **182 tok/s** |
| A 14,000-token prompt | 122 tok/s | **291 tok/s** (with its settle) |
| Decode, 256 tokens after the 14,000-token prompt | 23.0 tok/s | **37.1 tok/s** |

The two 14,000-token "before" figures are this fork's build with every new piece switched off, not the upstream binary.

What changed, in the order it was added:

| Change | Measured | Switch |
| --- | --- | --- |
| A Q5_K dense kernel for gfx906: a group of lanes for each row, every load of a lane issued before its first use, the activations in the GPU's local memory, its own rounding of the activations | decode 21.9 → 26.6 tok/s | `DS4_G906_Q5K=0` turns it off |
| Both dies compute every layer (tensor parallel over uneven whole blocks, upstream's opt-in) on by default | decode 26.6 → 27.5–28.9 tok/s; a 3,000-token prompt 108 → 152 tok/s | `MAYA_DS4_TP_GGUF_UNEVEN=0` |
| The prompt's expert tiles in F16 (`v_dot2_f32_f16`), not F32 | a 14,000-token prompt 236 → 314 tok/s | `DS4_G906_SET16=0` |
| The long prompt's layer skip on a tensor-parallel pair (upstream's opt-in) on by default | the same prompt 221 → 236 tok/s | `DS4_YOCO_TP=0` |
| After a chunked prompt the expert cache moves what the prompt used into VRAM (upstream's opt-in settle), 32 swaps under way | decode after the same prompt 20.7 → 30.4 tok/s; the prompt 44.5 → 48.3 s | `DS4_SETTLE_MS=0`, `DS4_CACHE_INFLIGHT=8` |
| The split-KV decode attention of the Strata engine (`ds4_attn_split.cuh`): the keys in chunks, 4 heads a block sharing the decoded rows | decode after the 14,000-token prompt 32.1 → 37.1 tok/s; after a 48-token prompt 29.2, 29.8 → 29.8, 30.5 | `DS4_ATTN_SPLIT=0` |
| No huge-page request for the experts' copy in RAM | not measured here; in the Strata engine it stopped freezes of 1–6 s | `DS4_ANON_THP=1` asks again |
| The router's picks in one pass; IQ2 signs by `v_perm_b32`; 3 waves a SIMD for the token gate / up kernel | within run-to-run noise | none |

The settle's budget is 0.5 ms for each prompt token, 800 ms at least and 5,000 ms at most; `DS4_SETTLE_MS=<ms>` sets it.

Upstream switches measured here and left off, since none moved decode beyond run-to-run noise (27.5–28.9 tok/s):
`MAYA_DS4_HOST_FREE_TP=1` (27.6, 28.6), `MAYA_DS4_HC_MIX_SPLIT=1` (28.9), `DS4_AR_P2P=1` (28.3). `MAYA_DS4_EXPERT_MMQ=1`
did not change the 14,000-token prompt (222 against 221 tok/s). `MAYA_DS4_ROUTE_FUSE=1` does not start on a wave64 build.

## Where the time goes, and what's left

A kernel trace of decode with both dies on every layer (256 tokens after a 48-token prompt): each die is busy 96–98 %
of the time. Of a die's kernel time, 20 % is `ar_sum_kernel`, the wait for the other die's sums (128 a token), and
13 % is `wait_seq_kernel`, the wait for the experts the Xeon computes. The Q5_K kernel is 14 %, the decode attention
6 %, and the rest is spread over launches of 5–50 µs. (Traced before the split-KV attention was added.)

Not done:

- The hyper-connection kernel of the Strata engine. It needs more of this engine's code changed than the pieces above.
- GLM-5.3-Flash, GLM-5.3 and MiniMax-M3.1-Flash on this machine: no profile, no gfx906 kernels.
- A short prompt (a few hundred tokens) still copies every expert that is not in VRAM to the dies, used or not, so
  it reads far slower per token than a long one.

## Quality check

The engine's logits on a 428-token text (prose and code) against an independent PyTorch forward of the same GGUFs on the
CPU (`tools/dsv41/ref_gguf.py` of the Strata fork; this repository's `tools/ds4/ref_gguf.py` does not read this set's
Q5_K embedding). "Off" is this build with every new piece switched off. Two runs of one configuration differ, because
which experts sit in VRAM differs: the same configuration twice agreed with itself on 95.6 % of the top tokens.

| Path | Configuration | Top token same as the reference | Mean KL |
| --- | --- | ---: | ---: |
| Windows of 16 rows (the decode kernels) | off | 405 of 428 | 0.0107 |
| | this fork's defaults (before the split attention) | 403 of 428 | 0.0107 |
| Windows of 4 rows | split attention off (2 runs) | 408, 402 of 428 | 0.0100, 0.0100 |
| | split attention on | 402 of 428 | 0.0097 |
| Chunked prompt, all rows | off (2 runs) | 401, 409 of 428 | 0.0110, 0.0106 |
| Chunked prompt, last 12 rows | off (2 runs) | 12 of 12 | 0.0071, 0.0094 |
| | both dies, F32 tiles (`DS4_G906_SET16=0`), with and without the skip | 12 of 12 | 0.0076, 0.0085 |
| | both dies, F16 tiles, no skip (2 runs) | 12 of 12 | 0.0120, 0.0145 |
| | this fork's defaults (3 runs) | 12 of 12 | 0.0134, 0.0147, 0.0224 |

With both dies on a layer the chunked path gives logits for a prompt's last 12 rows only, so those rows are all that can
be scored there. The F16 tiles raise the KL on them (about 0.008 to about 0.013–0.022) and leave every top token the
same; `DS4_G906_SET16=0` gives the F32 tiles back at the cost of the prompt speed above. No functional benchmark
(tool calls, retrieval) was run on this fork.
