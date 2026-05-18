# Demo video script — GutWise (Kaggle Gemma 4 Good Hackathon)

**Length target**: 2:30–3:00.
**Format**: screen capture + voiceover. Optional cold-open with you on camera (10 s) for warmth, then screen-only for the rest. Plain microphone — no music after the hook.

---

## SCENE 1 — Hook (0:00–0:15)

**Visual**: title card. White background. "GutWise" in a large serif. Subtitle: "An offline IBS education assistant on Gemma 4 E4B." Below: "Kaggle Gemma 4 Good Hackathon — Health & Sciences."

**Voiceover**:
> Irritable bowel syndrome affects one in seven adults worldwide. Most never see a gastroenterologist. The ones who do, wait months. GutWise is a fine-tuned Gemma 4 model that lives on a phone or laptop, explains what's happening, and — this is the part that matters — refuses to play doctor.

---

## SCENE 2 — The problem (0:15–0:30)

**Visual**: split screen. Left: generic chatbot answering "I have bloody stool and weight loss — could this be IBS?" with a reassuring "could be IBS, try peppermint oil…". Right: a stat card — "10–15% global prevalence. Months-long specialist wait times. Generic LLMs miss red flags."

**Voiceover**:
> Open the most popular general-purpose chatbot, type the same red-flag symptoms a real patient would type at 2 AM, and watch it reassure you past warning signs that need a doctor. That's not safe enough for medical education.

---

## SCENE 3 — Live demo: the chat (0:30–1:10)

**Visual**: HF Space, Chat tab. Type the question *"What's the low-FODMAP diet and how does it actually work for IBS?"*. Show the response streaming. Highlight: it cites real food categories, references the elimination → reintroduction phases, and mentions consulting a dietitian.

**Voiceover**:
> Here's GutWise on a low-FODMAP question. Notice three things: the structure of the answer comes from NICE and the ACG 2021 guidelines, not from training noise. It mentions reintroduction phases, which generic models often skip. And at the end, it sends the user to a dietitian — not because we hard-coded that, but because the training data is grounded in references that always do.

---

## SCENE 4 — The killer feature: red-flag handling (1:10–1:40)

**Visual**: Side-by-side tab. Type *"I've been losing weight without trying and there's blood in my stool. Could this be IBS?"*. Show base Gemma 4 on the left starting to discuss IBS subtypes. Show GutWise on the right opening with **"These symptoms are red flags — please see a clinician promptly. They are not typical of IBS."** Then context.

**Voiceover**:
> Same prompt to base Gemma 4, same prompt to GutWise. Base model starts discussing IBS subtypes. GutWise opens with red-flag escalation — bloody stool and weight loss are not IBS until a doctor rules out the things that they actually could be. On our held-out eval, GutWise hit 4 out of 4 red-flag prompts correctly across three independent runs. Our first attempt at this fine-tune regressed to 2 out of 4. Getting back to 4 out of 4 is what this submission is really about.

---

## SCENE 5 — How (45 s) (1:40–2:25)

**Visual**: pipeline diagram from the writeup, animated step by step. Then the results table fades in.

**Voiceover**:
> Six hundred and fifty-nine training pairs, every one grounded in Rome IV, the ACG, NICE, BSG, NHS, and StatPearls. Every prompt that generated those pairs was given a verified reference and a hallucination blocklist. We audited the data, verified the flags before regenerating — most of the flags turned out to be false positives — lowered LoRA capacity to match the dataset size, and ran a three-seed paired evaluation against the baseline in a single judge batch.
>
> Result: overall score up 0.073 over base Gemma 4 — small in absolute terms, but it beats baseline plus two sigma of variance. Empathy plus 0.11. Completeness plus 0.23. And safety preserved at 4 out of 4.

---

## SCENE 6 — Close (2:25–2:50)

**Visual**: title card again. URLs visible: github.com/y0sif/GutWise, huggingface.co/y0sif/GutWise, huggingface.co/spaces/y0sif/GutWise. Apache-2.0 badge.

**Voiceover**:
> GutWise is Apache-licensed. The model, the data, the eval set, the notebooks, and the demo are all in the repo. If you build medical assistants on small models, the methodology — audit first, ground every prompt, fine-tune small, evaluate paired with variance — is the part worth taking. Thanks for watching.

---

## Shot list / recording notes

- **Tools**: OBS (or screen.studio) at 1080p60, mic at 48 kHz mono. fish shell history visible? clear it first.
- **Browser**: clean profile, no extensions visible.
- **HF Space**: keep it warm before recording (one warmup query). Otherwise the first cold-start eats 30+ seconds.
- **Side-by-side prompt**: pre-warm both models with the FODMAP question, then clear and type the red-flag prompt live.
- **Cuts**: aggressively cut dead air during model generation. 2.5 minutes of voiceover, 30 seconds of model-output reveals.
- **Captions**: burn in via auto-editor or YouTube auto-caption + manual fixup pass. Required for accessibility judging.
- **End card**: hold for 3 seconds with URLs and the Apache-2.0 mark.

## Submission spots that re-use this script

- Kaggle writeup video embed
- Hugging Face Space "About" tab
- LinkedIn announcement
