# GutWise

**Offline-first IBS health education assistant powered by fine-tuned Gemma 4.**

Built for the [Kaggle Gemma 4 Good Hackathon](https://www.kaggle.com/competitions/gemma-4-good-hackathon) (Health & Sciences track).

> **Disclaimer**: GutWise provides educational information only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider.

## What is GutWise?

Irritable Bowel Syndrome (IBS) affects 10-15% of the global population, yet most people never receive a proper diagnosis. Access to gastroenterologists is severely limited in low-resource settings. GutWise is a fine-tuned Gemma 4 model that provides evidence-based IBS education, grounded in clinical guidelines (NICE, ACG, Rome IV), designed to run offline on consumer hardware.

## Features

- Evidence-based IBS education grounded in clinical guidelines
- Multimodal input: analyze meal photos for FODMAP guidance
- Function calling: structured symptom diary and food trigger tracking
- Offline-first: runs on consumer hardware without internet
- Safety-first: always defers to healthcare providers for diagnosis and treatment

## Quick Start

```bash
# Clone
git clone https://github.com/y0sif/GutWise.git
cd GutWise

# Install
uv sync --extra dev

# Run the demo app
uv run python -m app.main
```

## Pipeline

```
Collect (guidelines/papers) -> Chunk (by topic) -> Generate (synthetic Q&A) ->
Validate (LLM judge + medical filters) -> Train (QLoRA on Gemma 4) ->
Evaluate (accuracy + safety benchmarks) -> Demo (Gradio app)
```

## Project Structure

```
GutWise/
├── app/                    # Gradio demo application
├── datasets/
│   ├── sources/            # Raw collected source texts
│   ├── chunks/             # Topic-based chunks
│   ├── generated/          # Pre-validation instruction pairs
│   └── validated/          # Final training data
├── reference/              # Medical reference materials
│   ├── ibs_reference.md    # Verified IBS facts
│   ├── hallucination_blocklist.md
│   └── medication_whitelist.md
├── scripts/
│   ├── collect/            # Fetch open-access sources
│   ├── chunk/              # Split sources by topic
│   ├── generate/           # Synthetic Q&A generation
│   └── evaluate/           # Validation and benchmarking
├── notebooks/
│   ├── training/           # Fine-tuning notebooks
│   └── eval/               # Evaluation notebooks
├── tests/                  # Test suite
└── docs/                   # Documentation
```

## Development

```bash
# Install all extras
uv sync --extra dev --extra collect --extra generate

# Lint + format
uv run ruff check .
uv run ruff format .

# Run tests
uv run pytest
```

## License

Apache-2.0
