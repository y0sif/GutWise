# Project Description — Kaggle submission form (paste verbatim)

Copy everything between the `---` lines into the "Project Description" field on
the Kaggle submission. 332 words, under Kaggle's 350-word cap. No markdown
syntax — Kaggle's description box renders inline.

---

GutWise is a Gemma 4 E4B fine-tune that explains IBS to patients offline, on consumer hardware. Irritable Bowel Syndrome affects 10 to 15 percent of the global population, but most people never get a formal diagnosis: gastroenterologists are scarce in low-resource settings, and generic LLMs make the gap worse by hallucinating medications, inventing probiotic strain names, and reassuring their way past dangerous red-flag symptoms. GutWise targets that wedge with a small, grounded model that explains IBS the way a careful health educator would and refuses to play doctor.

The contribution is methodology, not raw score. Every one of the 659 training pairs is grounded in real clinical references (Rome IV, ACG 2021, NICE CG61, BSG 2021, StatPearls, NHS, MedlinePlus), enforced at generation time by a reference file injected into every sub-agent prompt, a hallucination blocklist, and a 58-drug medication whitelist. v1 regressed below baseline with red-flag handling at 2 out of 4; rather than blindly regenerate, we verified audit flags first (about 63 percent were false positives), surgically rewrote 5 dangerous entries, lowered LoRA capacity from r=16 to r=8, and re-evaluated with discipline. v2 is scored against base Gemma 4 E4B with a 3-run paired protocol over 50 held-out questions in a single judge batch (no calibration drift) across 5 medical dimensions: accuracy, safety, empathy, scope, completeness. Result: overall 4.769 vs 4.696 baseline (+0.073 at sigma=0.028), empathy +0.11, completeness +0.23, and red-flag handling 4 out of 4 across every run.

The number that matters isn't +0.073, it's the recovery: a documented "verify-flags-before-regenerate" loop that any small team fine-tuning medical AI on limited data can copy. GutWise is Apache-2.0, educational only, not medical advice and not a diagnostic tool. Code: https://github.com/y0sif/GutWise. Model: https://huggingface.co/y0sif/GutWise. Demo: https://huggingface.co/spaces/y0sif/GutWise. Dataset: https://huggingface.co/datasets/y0sif/GutWise-IBS-QA.
