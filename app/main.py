"""GutWise Gradio demo.

Three tabs:
  1. Chat — fine-tuned model with a permanent disclaimer banner.
  2. Side-by-side baseline vs GutWise.
  3. About — pipeline overview, eval results, sources, limitations.

Loads the LoRA adapter from Hugging Face Hub on top of `unsloth/gemma-4-E4B-it`.
Uses the ZeroGPU `@spaces.GPU` decorator when running on Hugging Face Spaces; falls
back to CPU (bf16, low-memory) locally and on `cpu-basic` Spaces.
"""

from __future__ import annotations

import json
import logging
import os
import traceback
from dataclasses import dataclass
from pathlib import Path

import gradio as gr
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("gutwise")


def _format_exc_for_user(exc: BaseException, prefix: str) -> str:
    """User-visible error string. `exc!r` catches empty-message exceptions.

    Always logs the full traceback server-side (Spaces runtime log) so we
    never silently lose information again.
    """
    logger.exception("%s", prefix)
    frames = traceback.format_exception(type(exc), exc, exc.__traceback__)
    tb_tail = "".join(frames[-3:])
    return f"{prefix}: {type(exc).__name__}: {exc!r}\n\nTraceback (last 3 frames):\n{tb_tail}"


def _device_of(model) -> torch.device:
    """Robust device probe — `model.device` is unreliable on PeftModel + CPU + low_cpu_mem_usage."""
    return next(model.parameters()).device


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
ADAPTER_ID = os.environ.get("GUTWISE_ADAPTER", "y0sif/GutWise")
TEMPERATURE = 0.7

# Medical answers (Rome IV walkthroughs, full FODMAP explanations, etc.) need
# room. On GPU we let it run long; on cpu-basic we keep it bounded so a single
# turn still finishes in a few minutes rather than timing out.
_HAS_CUDA = torch.cuda.is_available()
MAX_NEW_TOKENS = int(os.environ.get("GUTWISE_MAX_TOKENS", "1024" if _HAS_CUDA else "512"))

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
    # None when the adapter isn't reachable on the Hub (push pending).
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

    bf16 on both CPU and GPU: modern PyTorch supports CPU bf16 ops, and fp32
    weights would push a 4 B Gemma over `cpu-basic`'s 16 GB. `low_cpu_mem_usage`
    avoids the transient peak during `from_pretrained` that often OOMs CPU.

    If the adapter isn't on the Hub yet, we fall back to base Gemma 4.
    """
    global _models
    if _models is not None:
        return _models

    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    base = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto" if _HAS_CUDA else None,
        low_cpu_mem_usage=True,
    )
    if not _HAS_CUDA:
        base = base.to("cpu")

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
    f"**GutWise adapter is not yet reachable on the Hub** (`{ADAPTER_ID}`). "
    "Until the Colab push completes, the Space serves **base Gemma 4 E4B** so "
    "the UI is still functional — but the fine-tuned behavior shown in the "
    "writeup is not what's serving you right now. Reload the page after the "
    "push to flip to the fine-tuned model."
)


def _format_messages(history: list[dict[str, str]], user_msg: str) -> list[dict[str, str]]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    msgs.extend(history)
    msgs.append({"role": "user", "content": user_msg})
    return msgs


def _generate(model, tokenizer, messages: list[dict[str, str]]) -> str:
    device = _device_of(model)
    # Render the chat template to text, then tokenize so we get an explicit
    # attention_mask and can pass pad_token_id to generate. Without those,
    # transformers floods the runtime log with "attention mask was not set".
    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    batch = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model.generate(
            **batch,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(out[0, batch["input_ids"].shape[-1] :], skip_special_tokens=True)
    return decoded.strip()


def _is_oom(exc: BaseException) -> bool:
    """Detect both CPU MemoryError and CUDA OOM across torch versions."""
    if isinstance(exc, MemoryError):
        return True
    cuda_oom = getattr(torch.cuda, "OutOfMemoryError", None)
    if cuda_oom is not None and isinstance(exc, cuda_oom):
        return True
    return "out of memory" in str(exc).lower()


_OOM_MSG = (
    "Out of memory. On `cpu-basic` (16 GB) Gemma 4 E4B can be tight on long prompts. "
    "Try a shorter question, or upgrade the Space hardware "
    "(Settings → Space hardware → ZeroGPU)."
)


@spaces.GPU(duration=120)
def chat_finetuned(message: str, history: list[dict[str, str]]) -> str:
    try:
        models = _load_models()
    except Exception as exc:
        return _format_exc_for_user(exc, "Model failed to load")

    msgs = _format_messages(history, message)
    model = models.finetuned if models.finetuned is not None else models.base
    try:
        return _generate(model, models.tokenizer, msgs)
    except Exception as exc:
        if _is_oom(exc):
            logger.exception("OOM during chat_finetuned")
            return _OOM_MSG
        return _format_exc_for_user(exc, "Generation error")


@spaces.GPU(duration=180)
def chat_compare(message: str) -> tuple[str, str]:
    try:
        models = _load_models()
    except Exception as exc:
        err = _format_exc_for_user(exc, "Model failed to load")
        return err, err

    msgs = _format_messages([], message)
    try:
        base_out = _generate(models.base, models.tokenizer, msgs)
    except Exception as exc:
        if _is_oom(exc):
            logger.exception("OOM during chat_compare (base)")
            base_out = _OOM_MSG
        else:
            base_out = _format_exc_for_user(exc, "Generation error")

    if models.finetuned is None:
        ft_out = (
            f"GutWise adapter not yet on the Hub — side-by-side will activate "
            f"after the Colab push to `{ADAPTER_ID}`."
        )
    else:
        try:
            ft_out = _generate(models.finetuned, models.tokenizer, msgs)
        except Exception as exc:
            if _is_oom(exc):
                logger.exception("OOM during chat_compare (finetuned)")
                ft_out = _OOM_MSG
            else:
                ft_out = _format_exc_for_user(exc, "Generation error")
    return base_out, ft_out


def _refresh_banner():
    """Runs on every page load. Reflects current Hub state without a Space restart."""
    if _adapter_available():
        return gr.update(visible=False, value="")
    return gr.update(visible=True, value=f"> ⚠️ {_ADAPTER_PENDING_MSG}")


def _load_cached_examples() -> list[dict]:
    """Read pre-generated demo responses from app/cached_examples.json if present.

    Used by the Quick Demo tab so we can record a video without waiting for
    a live CPU inference call to finish. Empty list = tab hidden.
    """
    path = Path(__file__).parent / "cached_examples.json"
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        logger.exception("Failed to read %s", path)
        return []


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

        banner = gr.Markdown(visible=False)
        ui.load(_refresh_banner, outputs=[banner])

        cached = _load_cached_examples()
        if cached:
            with gr.Tab("Quick demo (instant)"):
                gr.Markdown(
                    "Pre-generated outputs from this fine-tune — real responses, "
                    "captured on a GPU pass and shown instantly here so the demo "
                    "doesn't wait on CPU inference. For your own questions, use "
                    "the **Chat** tab (slow on `cpu-basic`)."
                )
                for ex in cached:
                    label = ex.get("label") or ex.get("prompt", "Example")
                    prompt = ex.get("prompt", "")
                    ft_resp = ex.get("finetuned", "")
                    base_resp = ex.get("base")
                    with gr.Accordion(label, open=False):
                        gr.Markdown(f"**Prompt**\n\n> {prompt}")
                        if base_resp:
                            with gr.Row():
                                gr.Textbox(
                                    value=base_resp,
                                    label="Base Gemma 4 E4B",
                                    lines=10,
                                    interactive=False,
                                )
                                gr.Textbox(
                                    value=ft_resp,
                                    label="GutWise",
                                    lines=10,
                                    interactive=False,
                                )
                        else:
                            gr.Textbox(
                                value=ft_resp,
                                label="GutWise response",
                                lines=10,
                                interactive=False,
                            )

        with gr.Tab("Chat"):
            chat = gr.ChatInterface(
                fn=chat_finetuned,
                type="messages",
                examples=EXAMPLES,
                cache_examples=False,
            )
            _ = chat

        with gr.Tab("Side-by-side: baseline vs GutWise"):
            gr.Markdown(
                "Same prompt, base Gemma 4 E4B on the left, GutWise on the right. "
                "Compare empathy, completeness, and how each handles red-flag prompts."
            )
            q = gr.Textbox(
                label="Question",
                placeholder="e.g. 'I've had bloody stool and weight loss — could this be IBS?'",
                lines=2,
            )
            with gr.Row():
                base_box = gr.Textbox(label="Base Gemma 4 E4B", lines=10)
                ft_box = gr.Textbox(label="GutWise (this fine-tune)", lines=10)
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

## Results (3-run mean ± σ)

| Metric | Baseline E4B | GutWise | Δ |
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
