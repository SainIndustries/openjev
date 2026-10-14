"""OpenJev reads on an autoregressive Gemma 4 served by Ollama (CPU), OPENJEV_BACKEND=ollama.

Added by SAIN Industries, 2026-10-14 (Apache-2.0, see NOTICE). DiffusionGemma reads every label
slot of a seeded canvas in one denoise step. An autoregressive model cannot do that, so this
backend keeps OpenJev's decision logic and replaces only the read:

- Same schema, labels, system prompt, "id: label" answer template and Jev answer shapes
  (Engine.build_schema / system_text / to_answer / confidence).
- A read teacher-forces the answer template: for question k the prompt ends with the chosen
  lines for questions 1..k-1 and "qk: ", and the label distribution is the next-token
  logprobs, restricted to that question's labels (Ollama returns the top 20 tokens; labels
  outside them get OpenJev's floor, min(top) - 5).
- OpenJev's re-read policy carries over: when a read's entropy passes OPENJEV_AUTO_THRESHOLD,
  up to OPENJEV_AUTO_MAX - 1 more reads run and the label probabilities are averaged. An
  autoregressive read at temperature 0 has no noise to redraw, so each re-read presents the
  options in a seeded, different order (option-shuffle averaging) and maps the answers back.
- `samples` asks for that many option-shuffled reads. `steps` has no meaning here and is
  ignored. `think` and images are refused with a 400 (use the GPU DiffusionGemma backend).

Prompt-structure tokens (<|turn>, <turn|>, <|channel>, ...) are removed from caller text before
the raw prompt is built, so a state cannot open or close a turn.
"""
import asyncio
import math
import random
import re
import time

import httpx

from .engine import Engine, Overloaded, SchemaError, Upstream, model_ns, to_answer  # noqa: F401

TOPK = 20  # Ollama's top_logprobs ceiling
MAX_LABELS = 20  # a label outside the top 20 is only a floor, so one question offers at most 20
SCAFFOLD = "<|channel>thought\n<channel|>"
_SPECIAL = re.compile(r"<\|[a-z_]*>|<[a-z_]*\|>|<bos>|<eos>|<pad>", re.I)


def clean(text):
    """Caller text with Gemma prompt-structure tokens removed."""
    return _SPECIAL.sub("", text or "")


def gemma_prompt(system, user, answer_prefix):
    return (f"<|turn>system\n{clean(system)}<turn|>\n<|turn>user\n{clean(user)}<turn|>\n"
            f"<|turn>model\n{SCAFFOLD}{answer_prefix}")


def label_distribution(top, labels):
    """Probabilities over `labels` from one position's top tokens (token -> logprob).
    Letter labels also take their lower-case token; yes/no take their capitalised form."""
    lp = {}
    for tok, v in top.items():
        t = tok.strip()
        for lab in labels:
            if t == lab or t.lower() == lab.lower():
                lp[lab] = math.log(math.exp(lp[lab]) + math.exp(v)) if lab in lp else v
    floor = (min(top.values()) if top else 0.0) - 5.0
    vals = [lp.get(lab, floor) for lab in labels]
    mx = max(vals)
    ex = [math.exp(x - mx) for x in vals]
    z = sum(ex)
    top_p = [math.exp(v) for v in top.values()]
    return {"probs": [e / z for e in ex], "entropy": -sum(p * math.log(p) for p in top_p if p > 0),
            "seen": len(lp)}


class OllamaEngine(Engine):
    def __init__(self, settings, tokenizer=None):
        self.s = settings
        self.tok = None
        self.client = httpx.AsyncClient(base_url=settings.ollama_url.rstrip("/"),
                                        timeout=httpx.Timeout(settings.ollama_timeout, connect=5.0))
        # CPU: one read at a time is fastest overall; more only queue inside Ollama
        self.slots = asyncio.Semaphore(settings.ollama_parallel)
        self.waiting = 0
        self.choice_labels = [chr(c) for c in range(ord("A"), ord("A") + MAX_LABELS)]
        self._templates = {}

    def build_schema(self, questions):
        schema = super().build_schema(questions)
        schema["format"] = "lines"  # no canvas to fit, so always the plain "id: label" lines
        return schema

    def groups(self, qs, fmt):
        return [qs]

    async def _generate(self, prompt):
        started = time.perf_counter_ns()
        try:
            async with self.slots:
                r = await self.client.post("/api/generate", json={
                    "model": self.s.ollama_model, "prompt": prompt, "raw": True, "stream": False,
                    "think": False, "logprobs": True, "top_logprobs": TOPK,
                    "keep_alive": self.s.ollama_keep_alive,
                    "options": {"temperature": 0, "num_predict": 1, "num_ctx": self.s.ollama_num_ctx}})
        finally:
            spent = model_ns.get()
            if spent is not None:
                spent[0] += time.perf_counter_ns() - started
        if 400 <= r.status_code < 500:
            raise Upstream(f"ollama refused the read ({r.status_code})")
        r.raise_for_status()
        d = r.json()
        rows = d.get("logprobs") or []
        top = {t["token"]: t["logprob"] for t in (rows[0].get("top_logprobs") or [])} if rows else {}
        return top, int(d.get("prompt_eval_count") or 0)

    async def _one_read(self, qs, state_text, perm_seed):
        """One teacher-forced pass over all questions. perm_seed None = options as listed."""
        rng = random.Random(perm_seed) if perm_seed is not None else None
        shown, orders = [], []
        for q in qs:
            order = list(range(len(q["choices"])))
            if rng is not None and q["type"] == "choice":
                rng.shuffle(order)  # yes/no and score levels keep their natural order
            orders.append(order)
            shown.append(dict(q, choices=[q["choices"][i] for i in order]))
        sys_text = self.system_text(shown, "lines")
        lines, out, billed = [], [], 0
        for q, order in zip(shown, orders):
            prefix = "".join(line + "\n" for line in lines) + f"{q['id']}: "
            top, n = await self._generate(gemma_prompt(sys_text, state_text, prefix))
            billed += n
            dist = label_distribution(top, q["labels"])
            # back to the original option order
            probs = [0.0] * len(order)
            for shown_i, orig_i in enumerate(order):
                probs[orig_i] = dist["probs"][shown_i]
            out.append({"probs": probs, "entropy": dist["entropy"]})
            best = max(range(len(dist["probs"])), key=dist["probs"].__getitem__)
            lines.append(f"{q['id']}: {q['labels'][best]}")
        return out, billed

    async def read_group(self, qs, fmt, sys_text, content, seed, opts, prefix=None, lead=""):
        if not isinstance(content, str):
            raise SchemaError("images need the GPU (DiffusionGemma) backend", ("body", "images"))
        if opts.get("think"):
            raise SchemaError("think needs the GPU (DiffusionGemma) backend", ("body", "think"))
        if opts.get("samples"):
            n = int(opts["samples"])
            results = [await self._one_read(qs, content, None if k == 0 else seed + k * 7919) for k in range(n)]
            reads = [r for r, _ in results]
            billed = sum(b for _, b in results)
        else:
            first, billed = await self._one_read(qs, content, None)
            reads = [first]
            if self.s.auto_max > 1 and max(r["entropy"] for r in first) > self.s.auto_threshold:
                for k in range(1, self.s.auto_max):  # sequential: CPU, one read at a time
                    more, _ = await self._one_read(qs, content, seed + k * 7919)
                    reads.append(more)  # re-reads are the server's own policy and are not billed
        means = [[sum(r[qi]["probs"][l] for r in reads) / len(reads) for l in range(len(q["labels"]))]
                 for qi, q in enumerate(qs)]
        return means, billed, 0

    async def _sequential(self, groups, fmt, all_qs, state_text, seed, opts):
        # one group always holds every question, so sequential is the normal read
        return [await self.read_group(all_qs, fmt, None, state_text, seed, dict(opts, think=0))]
