# Changelog

## 0.7.1 (2026-10-12)

Only `openjev-jevk5` is released at 0.7.1; the other images stay at 0.7.0.

- JevK5 moves to v0.3, the checkpoint alibiserikbay/JevK5 now publishes (since 2026-09-25), and
  its id is `jevk5-0.3`; `jevk5-0.2` is no longer served. The image installs `jevk5` 0.3.3, whose
  prompt is unchanged, and a question with more than 16 options now uses the checkpoint's
  `knockout_temperature`, as JevK5's own runtime does. On JevBench's 231 public items OpenJev
  matches JevK5's published v0.3 run: the same input tokens on all 231 and the same top answer
  on 230 (the other is an exact tie).

## 0.7.0 (2026-10-12)

All five images (`openjev`, `openjev-laya`, `openjev-verdict`, `openjev-clm`, `openjev-jevk5`) are
released at 0.7.0, so every backend answers `/v1/decisions`.


- `POST /v1/decisions` serves OpenAI's
  [Decisions API](https://developers.openai.com/api/docs/guides/decisions), so the OpenAI SDKs'
  `client.decisions.create` works against OpenJev. A decision is translated into a
  `/v1/systemone` request and read the same way, so the same questions get the same answers, on
  every backend and through `OPENJEV_MODEL_ROUTES`. Its errors use OpenAI's shape.
- New models: Cloudflare's [Clef](https://huggingface.co/Cloudflare/clef) (`clef-1.0`, 27B) and
  [Clef-flash](https://huggingface.co/Cloudflare/clef-flash) (`clef-flash-1.0`, 9B), as
  simonlehmann's and alpha-x-ai's NVFP4/FP8 quantizations, with `OPENJEV_BACKEND=clef` or
  `clef-flash` in the OpenJev image. Each reads all of a request's questions in one forward pass:
  vLLM runs the backbone in the API server, and Cloudflare's head runs in its pooler. They take
  images.
- `confidence()` of a one-option distribution is 1.0 (#11). No request reached it.

## 0.6.0 (2026-10-06)

- vLLM is pinned to upstream main at
  [`a3e0243b`](https://github.com/vllm-project/vllm/commit/a3e0243b1c3ea313b902a43b0f6559fb4148ff81)
  (0.30.1rc1.dev707), up from `1b3b88ec` (the #57250 merge). It has every DiffusionGemma change
  merged after #57250:

  | PR | Change |
  |---|---|
  | [#58216](https://github.com/vllm-project/vllm/pull/58216) | Constrained reads: a read can score only its label tokens |
  | [#58226](https://github.com/vllm-project/vllm/pull/58226) | One-pass kernel for the sampler statistics |
  | [#51994](https://github.com/vllm-project/vllm/pull/51994) | The attention mask no longer freezes under CUDA graph replay |
  | [#48521](https://github.com/vllm-project/vllm/pull/48521) | The LM head gets the checkpoint's quantization config |

  Both image patches (512 label ids, bidirectional image attention) are still needed and still
  apply. The `clm` and `jevk5` images stay on `1b3b88ec`.
- New setting `OPENJEV_CONSTRAINED=1` sends #58216's `diffusion_constrained`. It is off by
  default: the logprobs come back normalized over the labels, which changes the entropy that
  triggers re-reads. With it on, reads were about 10% faster one at a time and 178 of 180 answers
  were the same.
- Accuracy against 0.5.0 on 2,501 labeled text questions (choice and yes/no) from the v6
  fast-dev set: 76.7% against 77.0%. 0.5.0 alone was right on 35 and 0.6.0 alone on 28, within
  run-to-run noise. Images and scores were not re-measured.
- `scripts/bench.py` and a README throughput table by state length, up to the 64K window
  ([#5](https://github.com/razorback16/openjev/issues/5)).
