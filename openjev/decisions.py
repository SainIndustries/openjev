"""OpenAI-compatible Decisions API: POST /v1/decisions.

The request and answer shapes follow OpenAI's Decisions guide
(https://developers.openai.com/api/docs/guides/decisions), so the OpenAI SDKs
work against this server by pointing their base URL at it. A decision is a Jev
request in other words: ``input`` is the state, a ``predicate`` is a ``noul``,
a ``choice`` lists its options as ``{value, description}`` and a ``score`` its
levels as ``{label, description}``. Questions are answered in the order asked,
named when the request names them. No question is refused, so an answer is never
``refusal``. This module only translates; the read is
the one /v1/systemone makes, and the same request gives the same answers.
"""
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, StrictBool, StrictStr

from .engine import SchemaError

Described = Union[str, dict[str, Any], list[Any], None]


class PredicateQuestion(BaseModel):
    type: Literal["predicate"]
    name: str | None = None
    instructions: Described = None


class ChoiceOption(BaseModel):
    # a string and a boolean with the same text are different values
    value: Union[StrictBool, StrictStr]
    description: Described = None


class ChoiceQuestion(BaseModel):
    type: Literal["choice"]
    name: str | None = None
    instructions: Described = None
    choices: list[ChoiceOption] = Field(min_length=1)


class ScoreLevel(BaseModel):
    label: str
    description: Described = None


class ScoreQuestion(BaseModel):
    type: Literal["score"]
    name: str | None = None
    instructions: Described = None
    levels: list[ScoreLevel] = Field(min_length=1)


Question = Annotated[Union[PredicateQuestion, ChoiceQuestion, ScoreQuestion], Field(discriminator="type")]


class InputText(BaseModel):
    type: Literal["input_text"]
    text: str


class InputImage(BaseModel):
    type: Literal["input_image"]
    image_url: str | None = None
    file_id: str | None = None
    detail: str | None = None


Part = Annotated[Union[InputText, InputImage], Field(discriminator="type")]


class Message(BaseModel):
    role: str  # only "user"; checked in to_systemone so the error can say so
    content: Union[str, list[Part]]
    type: Literal["message"] | None = None


class DecisionRequest(BaseModel):
    model: str
    input: Union[str, list[Message]]
    questions: list[Question] = Field(min_length=1)
    safety_identifier: str | None = None  # accepted and unused: nothing is kept per user
    # OpenJev's own options, as on /v1/systemone; OpenAI's SDKs never send them
    steps: int | None = Field(default=None, ge=1, le=8)
    samples: int | None = Field(default=None, ge=1, le=32)
    think: int | None = Field(default=None, ge=0, le=4096)
    sequential: bool | None = None


def to_systemone(req):
    """A decision request -> the /v1/systemone body that asks the same questions."""
    texts, images = [], []
    if isinstance(req.input, str):
        texts.append(req.input)
    else:
        for i, m in enumerate(req.input):
            if m.role != "user":
                raise SchemaError(f"input messages must have role 'user', not {m.role!r}", ("body", "input", i, "role"))
            parts = [InputText(type="input_text", text=m.content)] if isinstance(m.content, str) else m.content
            for j, p in enumerate(parts):
                if p.type == "input_text":
                    texts.append(p.text)
                elif p.image_url and p.image_url.startswith("data:"):
                    images.append(p.image_url)
                else:
                    raise SchemaError("an input_image must be a base64 data URL; hosted URLs and file_id are not supported",
                                      ("body", "input", i, "content", j))
    questions, names = {}, set()
    for i, q in enumerate(req.questions):
        if q.name is not None:
            if q.name in names:
                raise SchemaError(f"question names must be unique: {q.name!r}", ("body", "questions", i, "name"))
            names.add(q.name)
        # Jev keys questions by id and OpenAI lists them; the position is the id
        key = question_key(i)
        if q.type == "predicate":
            questions[key] = {"type": "noul", "instructions": q.instructions}
        elif q.type == "choice":
            keys = choice_keys(q.choices)
            if len(set(keys)) < len(keys):
                raise SchemaError(f"choice values must be unique in question {q.name or i!r}",
                                  ("body", "questions", i, "choices"))
            questions[key] = {"type": "choice", "instructions": q.instructions,
                              "criteria": {k: c.description for k, c in zip(keys, q.choices)}}
        else:
            # the label names a level and the description says what it means; the
            # model reads both, as a Jev score reads its level descriptions
            questions[key] = {"type": "score", "instructions": q.instructions,
                              "criteria": [level_text(lv) for lv in q.levels]}
    body = {"model": req.model, "state": "\n\n".join(texts), "questions": questions}
    if images:
        body["images"] = images
    for opt in ("steps", "samples", "think", "sequential"):
        if getattr(req, opt) is not None:
            body[opt] = getattr(req, opt)
    return body


def question_key(i):
    return str(i)


def choice_keys(choices):
    """The name the model sees for each option: the value, as text. A boolean reads as
    true/false; when a string option already has that text, the boolean is marked."""
    texts = {c.value for c in choices if isinstance(c.value, str)}
    keys = []
    for c in choices:
        if isinstance(c.value, bool):
            text = "true" if c.value else "false"
            keys.append(f"{text} (boolean)" if text in texts else text)
        else:
            keys.append(c.value)
    return keys


def level_text(level):
    if level.description is None:
        return level.label
    if isinstance(level.description, str):
        return f"{level.label}: {level.description.strip()}"
    return {"label": level.label, "description": level.description}


def to_decision(req, result):
    """A /v1/systemone response -> the Decision object, answers in question order."""
    answers = []
    for i, q in enumerate(req.questions):
        a = result["answers"][question_key(i)]
        named = {"name": q.name} if q.name is not None else {}
        if q.type == "predicate":
            answers.append({"type": "predicate", **named, "probability": a["noul"]})
        elif q.type == "choice":
            keys = choice_keys(q.choices)
            answers.append({"type": "choice", **named, "choice": q.choices[keys.index(a["choice"])].value,
                            "probabilities": [{"value": c.value, "probability": a["probabilities"].get(k, 0.0)}
                                              for k, c in zip(keys, q.choices)],
                            "confidence": a["confidence"]})
        else:
            answers.append({"type": "score", **named, "score": a["score"],
                            "probabilities": [{"value": j, "label": lv.label,
                                               "probability": a["probabilities"].get(str(j), 0.0)}
                                              for j, lv in enumerate(q.levels)],
                            "confidence": a["confidence"]})
    usage = result.get("usage", {})
    # OpenJev caches nothing between requests; its only output tokens are a think thought
    input_tokens, output_tokens = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
    return {"model": result.get("model", req.model), "answers": answers,
            "usage": {"input_tokens": input_tokens,
                      "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                      "output_tokens": output_tokens,
                      "output_tokens_details": {"reasoning_tokens": output_tokens},
                      "total_tokens": input_tokens + output_tokens}}


OAI_TYPES = {400: "invalid_request_error", 401: "authentication_error", 403: "permission_error",
             404: "not_found_error", 413: "invalid_request_error", 422: "invalid_request_error",
             429: "rate_limit_error"}


def oai_error_body(status, jev_body):
    """A /v1/systemone error body (Jev's shapes) -> OpenAI's ``{"error": {...}}``."""
    detail = jev_body.get("detail") if isinstance(jev_body, dict) else None
    code, param = None, None
    if isinstance(detail, dict):
        message, code = detail.get("message", ""), detail.get("error_type")
    elif isinstance(detail, list) and detail:
        first = detail[0]
        message, param = first.get("msg", "Invalid request."), param_of(first.get("loc", ()))
    elif isinstance(detail, str):
        message = detail
    else:
        message = "Invalid request."
    return {"error": {"message": message, "type": OAI_TYPES.get(status, "server_error"), "param": param, "code": code}}


def param_of(loc):
    """A validation location -> OpenAI's dotted ``param``, without the leading "body"."""
    parts = [str(p) for p in loc if p != "body"]
    return ".".join(parts) or None
