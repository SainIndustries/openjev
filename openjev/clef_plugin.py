"""Registers Clef's vLLM model (openjev.clef_vllm) with vLLM, as a general plugin.

vLLM loads every installed `vllm.general_plugins` entry point in each of its processes, so this
module stays light: the model class is registered by name and imported only when a Clef
checkpoint is loaded. The DiffusionGemma server never imports it.
"""

ARCHITECTURE = "ClefForDecision"
TASK = "token_classify"


def register():
    from vllm import ModelRegistry

    if ARCHITECTURE not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(ARCHITECTURE, "openjev.clef_vllm:ClefForDecision")


def schema_layout(encoded):
    """What the pooler needs of an encode_record result: each question's type, its
    instruction span and its options' spans, in token positions."""
    return [[q.question_type, list(q.question_span), [list(s) for s in q.option_spans]] for q in encoded.questions]


def vllm_prompt(input_ids, pad_ids, images):
    """vLLM's prompt for an encode_record result. Clef's processor expands each image into a run
    of pad tokens; vLLM expands it again from the image, so each run goes in as one token. The
    positions then match Clef's, which the pooler checks (as simonlehmann/clef-NVFP4's
    clef_vllm.py does)."""
    ids = [t for i, t in enumerate(input_ids) if not (t in pad_ids and i and input_ids[i - 1] == t)]
    prompt = {"prompt_token_ids": ids}
    if images:
        prompt["multi_modal_data"] = {"image": images if len(images) > 1 else images[0]}
    return prompt
