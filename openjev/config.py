"""Runtime settings, all from the environment.

Modified by SAIN Industries 2026-10-14: OPENJEV_BACKEND=ollama settings and its model name
(openjev-gemma4-e4b). See NOTICE.
"""
import os
from dataclasses import dataclass, field


def _env(name, default):
    return os.environ.get(name, default)


def _env_num(name, cast, default=None, minimum=None):
    """A numeric setting, or `default` when unset or empty.

    `int(os.environ[...])` raises a ValueError naming only the offending text,
    at import time, so one mistyped variable fails the whole service with a
    message that does not say which variable was wrong. Naming it costs a line.

    `None` means unset, and is distinct from a valid `0`: a cache limit of 0
    disables MLX's buffer cache, which is a real choice and not the same as
    leaving the limit alone.
    """
    raw = _env(name, "")
    if raw == "":
        return default
    try:
        value = cast(raw)
    except ValueError:
        raise ValueError(f"{name}={raw!r} is not a {cast.__name__}") from None
    if minimum is not None and value < minimum:
        raise ValueError(f"{name}={value} is below the minimum of {minimum}")
    return value


@dataclass(frozen=True)
class Settings:
    upstream: str = field(default_factory=lambda: _env("OPENJEV_UPSTREAM", "http://127.0.0.1:8000"))
    upstream_model: str = field(default_factory=lambda: _env("OPENJEV_UPSTREAM_MODEL", "dgemma"))
    tokenizer: str = field(default_factory=lambda: _env("OPENJEV_TOKENIZER", "nvidia/diffusiongemma-26B-A4B-it-NVFP4"))
    backend: str = field(default_factory=lambda: _env("OPENJEV_BACKEND", "vllm"))
    mlx_model: str = field(default_factory=lambda: _env("OPENJEV_MLX_MODEL", "mlx-community/diffusiongemma-26B-A4B-it-4bit"))
    mlx_max_prompt: int = field(default_factory=lambda: int(_env("OPENJEV_MLX_MAX_PROMPT", "32768")))
    # MLX buffer pool ceiling, in GB. Unset leaves MLX alone, whose own default
    # is the memory limit. 0 DISABLES the cache, which is the worst allocator
    # churn rather than the old behaviour - see mlx_backend.apply_mlx_settings.
    mlx_cache_limit_gb: float | None = field(
        default_factory=lambda: _env_num("OPENJEV_MLX_CACHE_LIMIT_GB", float, minimum=0))
    # Prefill cache size in ENTRIES. See mlx_backend.PROMPT_CACHE_TOKENS for why
    # entries as well as tokens: an entry costs KV cache, not prompt length.
    mlx_prompt_cache: int = field(
        default_factory=lambda: _env_num("OPENJEV_MLX_PROMPT_CACHE", int, default=12, minimum=0))
    canvas: int = field(default_factory=lambda: int(_env("OPENJEV_CANVAS", "64")))
    canvas_step: int = field(default_factory=lambda: int(_env("OPENJEV_CANVAS_STEP", "16")))
    max_inflight: int = field(default_factory=lambda: int(_env("OPENJEV_MAX_INFLIGHT", "64")))
    max_queue: int = field(default_factory=lambda: int(_env("OPENJEV_MAX_QUEUE", "512")))
    # Per-request bounds, so one body cannot fan out into unbounded work or memory.
    max_questions: int = field(default_factory=lambda: int(_env("OPENJEV_MAX_QUESTIONS", "256")))
    max_body_bytes: int = field(default_factory=lambda: int(_env("OPENJEV_MAX_BODY_BYTES", str(64 * 1024 * 1024))))
    # Seconds before a request passed through to another OpenJev container is a 503:
    # a heavy read (samples x groups, a long thought) can legitimately take minutes.
    forward_timeout: float = field(default_factory=lambda: float(_env("OPENJEV_FORWARD_TIMEOUT", "300")))
    # Optional auth. OPENJEV_API_KEY: clients send it as a Bearer token.
    # OPENJEV_ORIGIN_SECRET: a front proxy sends it as X-Origin-Secret.
    api_key: str = field(default_factory=lambda: _env("OPENJEV_API_KEY", ""))
    origin_secret: str = field(default_factory=lambda: _env("OPENJEV_ORIGIN_SECRET", ""))
    auto_threshold: float = field(default_factory=lambda: float(_env("OPENJEV_AUTO_THRESHOLD", "0.1")))
    auto_max: int = field(default_factory=lambda: int(_env("OPENJEV_AUTO_MAX", "4")))
    # vLLM's diffusion_constrained: each read scores only its label tokens, not the
    # whole vocabulary. Off by default: the logprobs come back normalized over the
    # labels, so a slot's entropy, and with it auto_threshold's re-reads, change meaning.
    constrained: bool = field(default_factory=lambda: _env("OPENJEV_CONSTRAINED", "0") == "1")
    max_images: int = field(default_factory=lambda: int(_env("OPENJEV_MAX_IMAGES", "8")))
    max_image_bytes: int = field(default_factory=lambda: int(_env("OPENJEV_MAX_IMAGE_BYTES", str(5 * 1024 * 1024))))
    # Kept small: generation denoises many blocks and must not crowd out System One reads.
    gen_max_inflight: int = field(default_factory=lambda: int(_env("OPENJEV_GEN_MAX_INFLIGHT", "8")))
    gen_max_queue: int = field(default_factory=lambda: int(_env("OPENJEV_GEN_MAX_QUEUE", "32")))
    gen_max_tokens: int = field(default_factory=lambda: int(_env("OPENJEV_GEN_MAX_TOKENS", "8192")))
    # The encoder backends (OPENJEV_BACKEND=laya, verdict, clm, jevk5, clef or clef-flash). OPENJEV_DEVICE empty picks CUDA when present.
    laya_model: str = field(default_factory=lambda: _env("OPENJEV_LAYA_MODEL", "convaiinnovations/laya-typed-decisions"))
    verdict_model: str = field(default_factory=lambda: _env("OPENJEV_VERDICT_MODEL", "heman10x/rlcd-modernbert-151m"))
    device: str = field(default_factory=lambda: _env("OPENJEV_DEVICE", ""))
    encoder_batch: int = field(default_factory=lambda: int(_env("OPENJEV_ENCODER_BATCH", "16")))
    # CLM (OPENJEV_BACKEND=clm): its heads read Qwen3-8B embeddings from the vLLM server at OPENJEV_UPSTREAM.
    # OPENJEV_CLM_HEAD is a Hugging Face repo holding CLM_v0.1-8B.pt, or a local .pt file.
    clm_head: str = field(default_factory=lambda: _env("OPENJEV_CLM_HEAD", "Contrastive-LM/CLM-v0.1-8B"))
    clm_max_tokens: int = field(default_factory=lambda: int(_env("OPENJEV_CLM_MAX_TOKENS", "2048")))
    clm_workers: int = field(default_factory=lambda: int(_env("OPENJEV_CLM_WORKERS", "32")))
    clm_cache: str = field(default_factory=lambda: _env("OPENJEV_CLM_CACHE", "256MiB"))
    clm_embed_cache: int = field(default_factory=lambda: int(_env("OPENJEV_CLM_EMBED_CACHE", "20000")))
    # JevK5 (OPENJEV_BACKEND=jevk5): the weights the vLLM server at OPENJEV_UPSTREAM serves. Their
    # jevk5_config.json holds the calibration temperature.
    jevk5_model: str = field(default_factory=lambda: _env("OPENJEV_MODEL", "alibiserikbay/JevK5"))
    jevk5_workers: int = field(default_factory=lambda: int(_env("OPENJEV_JEVK5_WORKERS", "32")))
    # Clef (OPENJEV_BACKEND=clef or clef-flash) runs vLLM in this process. OPENJEV_CLEF_MODEL is a
    # Hugging Face repo or a local directory; empty is the backend's own checkpoint, at the
    # revision pinned in encoders.CLEF_CHECKPOINTS. GPU_UTIL and MAX_NUM_SEQS mean what they mean
    # to the entrypoint's vLLM.
    clef_model: str = field(default_factory=lambda: _env("OPENJEV_CLEF_MODEL", ""))
    clef_max_tokens: int = field(default_factory=lambda: int(_env("OPENJEV_CLEF_MAX_TOKENS", "16384")))
    clef_gpu_util: float = field(default_factory=lambda: float(_env("OPENJEV_GPU_UTIL", "0.85")))
    clef_max_num_seqs: int = field(default_factory=lambda: int(_env("OPENJEV_MAX_NUM_SEQS", "64")))
    # SAIN: OPENJEV_BACKEND=ollama, an autoregressive Gemma 4 on Ollama (CPU). See ollama_backend.py.
    ollama_url: str = field(default_factory=lambda: _env("OPENJEV_OLLAMA_URL", "http://127.0.0.1:11434"))
    ollama_model: str = field(default_factory=lambda: _env("OPENJEV_OLLAMA_MODEL", "gemma4:e4b"))
    ollama_timeout: float = field(default_factory=lambda: float(_env("OPENJEV_OLLAMA_TIMEOUT", "120")))
    ollama_parallel: int = field(default_factory=lambda: int(_env("OPENJEV_OLLAMA_PARALLEL", "1")))
    ollama_num_ctx: int = field(default_factory=lambda: int(_env("OPENJEV_OLLAMA_NUM_CTX", "4096")))
    ollama_keep_alive: str = field(default_factory=lambda: _env("OPENJEV_OLLAMA_KEEP_ALIVE", "-1"))
    warmup: bool = field(default_factory=lambda: _env("OPENJEV_WARMUP", "1") != "0")
    # Other System One models served by other OpenJev containers: "name=url,name=url".
    # A request for one of them is passed through unchanged, so one origin serves all.
    model_routes: dict = field(default_factory=lambda: parse_routes(_env("OPENJEV_MODEL_ROUTES", "")))

    def __post_init__(self):
        """Refuse values a bad environment would otherwise turn into 500s or hangs:
        canvas_step=0 divides by zero on the first read, and a semaphore built with
        0 never opens, so every request would wait out its timeout instead of a 529."""
        positive = ("canvas", "canvas_step", "max_inflight", "max_questions", "max_body_bytes",
                    "max_image_bytes", "gen_max_inflight", "gen_max_tokens", "mlx_max_prompt",
                    "encoder_batch", "clm_workers", "jevk5_workers", "clef_max_tokens", "clef_max_num_seqs", "forward_timeout")
        for name in positive:
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1, got {getattr(self, name)!r} "
                                 f"(OPENJEV_{name.upper()})")
        for name in ("max_queue", "gen_max_queue", "max_images"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must not be negative, got {getattr(self, name)!r} "
                                 f"(OPENJEV_{name.upper()})")


def parse_routes(text):
    routes = {}
    for part in filter(None, (p.strip() for p in text.split(","))):
        name, sep, url = part.partition("=")
        if not (sep and name.strip() and url.strip()):
            raise ValueError(f"OPENJEV_MODEL_ROUTES: {part!r} is not name=url")
        routes[name.strip()] = url.strip().rstrip("/")
    return routes


MODEL_VERSION = "openjev-0.1"
# openjev-0.1 is the wire name of this model, not the package version (see __init__.py).
MODEL_ALIASES = {"openjev-latest", MODEL_VERSION}
GEN_MODEL = "diffusiongemma-26b"
# accepted so TypeSafe's SDKs work unchanged (their default is jev-latest)
SDK_ALIASES = {"jev-latest", "jev-preview"}
MODELS = [
    {"name": "openjev-latest", "description": "Alias for the newest OpenJev release. Currently openjev-0.1.",
     "release_date": "2026-09-18"},
    {"name": "openjev-0.1", "description": "OpenJev 0.1: DiffusionGemma 26B-A4B (NVFP4) on vLLM's structured reads.",
     "release_date": "2026-09-18"},
    {"name": GEN_MODEL, "description": "DiffusionGemma 26B-A4B (NVFP4) text generation at POST /v1/chat/completions.",
     "release_date": "2026-09-18"},
]

# The encoder backends each serve one model, under its own name. The models are other
# people's work; the descriptions credit them wherever the model list is shown.
ENCODER_MODELS = {
    "laya": {"name": "laya-1.0",
             "description": "Laya by Nandakishor M / Convai Innovations (github.com/NandhaKishorM/laya, Apache-2.0): "
                            "the laya-typed-decisions checkpoint, a ModernBERT-large encoder (421M) fine-tuned on "
                            "the typed-decisions workflows. Text only, 1,024 tokens.",
             "release_date": "2026-09-22"},
    "verdict": {"name": "verdict-1.4",
                "description": "Verdict by Heman10x (github.com/Heman10x-NGU/Verdict-open-jev, Apache-2.0): "
                               "rlcd-modernbert-151m, a ModernBERT-base + GLiClass encoder (151M) calibrated with "
                               "RLCD, with the v1.4 inference engine. Text only, 512 tokens, up to 24 choices.",
                "release_date": "2026-09-22"},
    "clm": {"name": "clm-v0.1",
            "description": "CLM v0.1 by Contrastive-LM (github.com/Contrastive-LM/CLM, Apache-2.0): state and action "
                           "projection heads over Qwen3-8B last-token embeddings, here with Qwen3-8B-FP8 on vLLM. "
                           "Text only, 2,048 tokens.",
            "release_date": "2026-09-24"},
    "jevk5": {"name": "jevk5-0.3",
              "description": "JevK5 v0.3 by Alibi Serikbay (github.com/allebee/jevk5, Apache-2.0): Qwen3.5-4B with a "
                             "LoRA distilled from Qwen3.6-27B and GPT-6 Luna, merged, read as a softmax over the answer letters' "
                             "logits (SemIf's readout), on vLLM. Text only, 16,384 tokens.",
              "release_date": "2026-09-25"},
    "clef": {"name": "clef-1.0",
             "description": "Clef by Cloudflare (huggingface.co/Cloudflare/clef, Apache-2.0): Qwen3.8-27B with a "
                            "joint schema head that scores every option of every question in one forward pass. "
                            "Here simonlehmann's NVFP4/FP8 quantization (simonlehmann/clef-NVFP4) on vLLM. Text and "
                            "images, 16,384 tokens.",
             "release_date": "2026-10-06"},
    "clef-flash": {"name": "clef-flash-1.0",
                   "description": "Clef-flash by Cloudflare (huggingface.co/Cloudflare/clef-flash, Apache-2.0): Clef "
                                  "on Qwen3.5-9B. Here alpha-x-ai's NVFP4/FP8 quantization "
                                  "(alpha-x-ai/clef-flash-NVFP4) on vLLM. Text and images, 16,384 tokens.",
                   "release_date": "2026-10-06"},
}


# SAIN: the CPU backend serves Gemma 4 E4B under its own name, never as DiffusionGemma.
OLLAMA_MODELS = [
    {"name": "openjev-gemma4-e4b",
     "description": "OpenJev decision logic on Gemma 4 E4B (Google, Apache-2.0), autoregressive, via Ollama on CPU. "
                    "Option-shuffle re-reads instead of diffusion noise re-reads.",
     "release_date": "2026-10-14"},
]


def served_models(backend):
    """(the model version a response names, the names a request may use, the /v1/models list)."""
    if backend == "ollama":
        m = OLLAMA_MODELS[0]
        return m["name"], {m["name"]} | SDK_ALIASES, list(OLLAMA_MODELS)
    if backend in ENCODER_MODELS:
        m = ENCODER_MODELS[backend]
        return m["name"], {m["name"]} | SDK_ALIASES, [m]
    return MODEL_VERSION, MODEL_ALIASES | SDK_ALIASES, MODELS
