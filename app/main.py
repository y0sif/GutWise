"""GutWise Gradio demo.

Two tabs:
  1. Chat — fine-tuned v2 model with a permanent disclaimer banner and a
     side-by-side baseline comparison button.
  2. About — pipeline overview, eval results, sources, limitations.

Loads the v2 LoRA adapter from Hugging Face Hub on top of `unsloth/gemma-4-E4B-it`.
Uses the ZeroGPU `@spaces.GPU` decorator when running on Hugging Face Spaces; falls
back to a normal CUDA/CPU code path locally.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import gradio as gr
import torch

try:
    import spaces  # Hugging Face Spaces ZeroGPU helper
except ImportError:  # local development

    class _NoOp:
        def GPU(self, *_args, **_kwargs):  # noqa: N802
            def decorator(fn):
                return fn

            return decorator

    spaces = _NoOp()  # type: ignore[assignment]


BASE_MODEL_ID = os.environ.get("GUTWISE_BASE", "unsloth/gemma-4-E4B-it")
ADAPTER_ID = os.environ.get("GUTWISE_ADAPTER", "y0sif/GutWise-v2")
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7

SYSTEM_PROMPT = (
    "You are GutWise, an IBS health education assistant. You provide evidence-based "
    "information about Irritable Bowel Syndrome to help users understand and manage "
    "their condition. You are not a doctor and cannot diagnose or prescribe. Always "
    "recommend consulting a healthcare provider for personal medical decisions."
)

DISCLAIMER = (
    "**GutWise provides educational information only — not medical advice.** "
    "If you have rectal bleeding, unintentional weight loss, fever, nocturnal "
    "symptoms, onset after age 50, or a family history of colorectal cancer or IBD, "
    "please see a clinician."
)

EXAMPLES = [
    "What's the difference between IBS and IBD?",
    "Can you walk me through the Rome IV criteria used to diagnose IBS?",
    "My doctor mentioned the low-FODMAP diet — what does that actually mean?",
    "Are probiotics worth trying for IBS?",
    # Red-flag example — meant to surface the safety-refusal behavior.
    "I've been losing weight without trying and there's blood in my stool. What could this be?",
]


@dataclass
class Models:
    tokenizer: object
    base: object
    # None while the LoRA adapter is being uploaded to the Hub.
    finetuned: object | None
    adapter_status: str  # "ready" | "missing" | "load_error: ..."


_models: Models | None = None


def _adapter_available() -> bool:
    """Probe the Hub for the adapter so we can fall back to base cleanly."""
    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

    try:
        HfApi().model_info(ADAPTER_ID)
        return True
    except (RepositoryNotFoundError, HfHubHTTPError):
        return False


def _load_models() -> Models:
    """Lazy-load base + LoRA on first inference call.

    If the adapter isn't on the Hub yet (Colab push pending), we fall back to
    base Gemma 4 so the Space stays usable and the user gets a clear message
    instead of a 500.
    """
    global _models
    if _models is not None:
        return _models

    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    base = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    finetuned: object | None = None
    status = "missing"
    if _adapter_available():
        try:
            finetuned = PeftModel.from_pretrained(base, ADAPTER_ID)
            status = "ready"
        except Exception as exc:  # pragma: no cover — Hub-flakiness path
            status = f"load_error: {exc.__class__.__name__}"

    _models = Models(
        tokenizer=tokenizer,
        base=base,
        finetuned=finetuned,
        adapter_status=status,
    )
    return _models


_ADAPTER_PENDING_MSG = (
    "**GutWise v2 adapter is not yet published to Hugging Face Hub.** "
    "The repo `y0sif/GutWise-v2` has not been pushed yet (the Colab notebook "
    "that does this hasn't been run). Until then this Space falls back to "
    "**base Gemma 4 E4B** so you can still try the UI — but the fine-tuned "
    "behavior you see in the writeup is not what's serving you right now."
)


def _format_messages(history: list[dict[str, str]], user_msg: str) -> list[dict[str, str]]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    msgs.extend(history)
    msgs.append({"role": "user", "content": user_msg})
    return msgs


def _generate(model, tokenizer, messages: list[dict[str, str]]) -> str:
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        out = model.generate(
            inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=0.9,
        )
    # Drop the prompt prefix
    decoded = tokenizer.decode(out[0, inputs.shape[-1] :], skip_special_tokens=True)
    return decoded.strip()


@spaces.GPU(duration=120)
def chat_finetuned(message: str, history: list[dict[str, str]]) -> str:
    try:
        models = _load_models()
    except Exception as exc:
        return f"Model failed to load: {exc.__class__.__name__}: {exc}"

    msgs = _format_messages(history, message)
    model = models.finetuned if models.finetuned is not None else models.base
    try:
        return _generate(model, models.tokenizer, msgs)
    except Exception as exc:
        return f"Generation error: {exc.__class__.__name__}: {exc}"


@spaces.GPU(duration=180)
def chat_compare(message: str) -> tuple[str, str]:
    try:
        models = _load_models()
    except Exception as exc:
        err = f"Model failed to load: {exc.__class__.__name__}: {exc}"
        return err, err

    msgs = _format_messages([], message)
    try:
        base_out = _generate(models.base, models.tokenizer, msgs)
    except Exception as exc:
        base_out = f"Generation error: {exc.__class__.__name__}: {exc}"

    if models.finetuned is None:
        ft_out = (
            "GutWise v2 adapter not yet on the Hub — side-by-side will activate "
            "after the Colab push to `y0sif/GutWise-v2`."
        )
    else:
        try:
            ft_out = _generate(models.finetuned, models.tokenizer, msgs)
        except Exception as exc:
            ft_out = f"Generation error: {exc.__class__.__name__}: {exc}"
    return base_out, ft_out


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="GutWise — IBS Education Assistant",
        theme=gr.themes.Soft(primary_hue="emerald"),
    ) as ui:
        gr.Markdown("# GutWise")
        gr.Markdown(
            "Offline-first IBS education assistant — a Gemma 4 E4B fine-tune grounded "
            "in Rome IV, ACG, NICE, BSG, StatPearls, NHS, and MedlinePlus."
        )
        gr.Markdown(f"> {DISCLAIMER}", elem_classes=["disclaimer"])

        # Probe once at UI build time so the banner reflects current state.
        # This call is cheap (single HEAD-ish request) and lets us avoid
        # surfacing a 500 if the adapter hasn't been pushed yet.
        if not _adapter_available():
            gr.Markdown(f"> ⚠️ {_ADAPTER_PENDING_MSG}")

        with gr.Tab("Chat"):
            chat = gr.ChatInterface(
                fn=chat_finetuned,
                type="messages",
                examples=EXAMPLES,
                cache_examples=False,
            )
            _ = chat

        with gr.Tab("Side-by-side: baseline vs v2"):
            gr.Markdown(
                "Same prompt, base Gemma 4 E4B on the left, GutWise v2 on the right. "
                "Compare empathy, completeness, and how each handles red-flag prompts."
            )
            q = gr.Textbox(
                label="Question",
                placeholder="e.g. 'I've had bloody stool and weight loss — could this be IBS?'",
                lines=2,
            )
            with gr.Row():
                base_box = gr.Textbox(label="Base Gemma 4 E4B", lines=10)
                ft_box = gr.Textbox(label="GutWise v2 (this fine-tune)", lines=10)
            btn = gr.Button("Compare", variant="primary")
            btn.click(chat_compare, inputs=q, outputs=[base_box, ft_box])
            gr.Examples(EXAMPLES, inputs=q)

        with gr.Tab("About"):
            gr.Markdown(_about_md())

    return ui


def _about_md() -> str:
    return """
## What GutWise is

A QLoRA fine-tune (r=8, lr=5e-5, eff batch 32, 1 epoch) of `unsloth/gemma-4-E4B-it`
on 659 audited IBS Q&A pairs.

## Results (v2, 3-run mean ± σ)

| Metric | Baseline E4B | GutWise v2 | Δ |
|---|---|---|---|
| Overall | 4.696 | **4.769 ± 0.028** | +0.073 |
| Empathy | 4.36 | 4.47 ± 0.08 | +0.11 |
| Completeness | 4.44 | 4.67 ± 0.04 | **+0.23** |
| Red-flag handling | 4/4 | **4/4 across all 3 runs** | ✓ |

Per-category gains: red_flag +0.22, safety_refusal +0.19.

## Pipeline

```
Collect (StatPearls, NICE, NHS, ACG, BSG, MedlinePlus, PubMed)
  → Chunk by topic
  → Generate (reference-grounded sub-agents)
  → Validate (medication whitelist + hallucination blocklist + LLM judge)
  → Train (QLoRA on Gemma 4 E4B)
  → Evaluate (3-run paired vs baseline, single judge batch)
```

## Sources & licenses

- StatPearls (CC-BY 4.0)
- NHS / NICE CG61 (Open Government Licence v3.0)
- MedlinePlus (Public Domain)
- ACG 2021, BSG 2021, Rome IV (cited)

GutWise itself: **Apache-2.0**.

## Limitations

- Educational only — not a diagnostic or prescribing tool.
- English only.
- Adults only (pediatric IBS criteria differ).
- Some hallucinations (e.g. invented strain names) are inherited from base Gemma 4.
- Built for the [Kaggle Gemma 4 Good Hackathon](https://www.kaggle.com/competitions/gemma-4-good-hackathon).
- Code: <https://github.com/y0sif/GutWise>
"""


if __name__ == "__main__":
    build_ui().launch()
