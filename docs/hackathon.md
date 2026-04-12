# Hackathon Strategy

## Competition: Gemma 4 Good Hackathon
- **Track**: Health & Sciences
- **Deadline**: May 18, 2026
- **Prize pool**: $200,000
- **URL**: https://www.kaggle.com/competitions/gemma-4-good-hackathon

## Judging Criteria
- Innovation: 30%
- Impact Potential: 30%
- Technical Execution: 25%
- Accessibility: 15%

## Submission Requirements
1. Working demo (Gradio on HuggingFace Spaces)
2. Public code repository (this repo)
3. Technical writeup (Kaggle notebook)
4. Demo video (2-3 min, YouTube)

## Allowed Models
- Gemma 4 2B (edge/mobile)
- Gemma 4 7B (consumer)
- Gemma 4 27B (high-performance)
- Gemma 4 27B-IT (instruction-tuned)

## Differentiation Strategy

### What winners of past Gemma hackathons did:
1. Fine-tuned models (not just prompt engineering)
2. Offline-first / on-device deployment
3. Targeted specific underserved populations
4. Used multimodal + function calling
5. Grounded in real clinical guidelines

### Our differentiators:
- Fine-tuned Gemma 4 on IBS-specific data (not generic medical)
- Multimodal: meal photo analysis for FODMAP guidance
- Function calling: structured symptom diary + food trigger tracking
- Offline-first: runs without internet on consumer hardware
- Grounded in NICE/ACG/Rome IV guidelines
- Personal story: built by someone living with IBS

## Timeline (5 weeks)

### Week 1 (Apr 10-16): Data Collection + Reference
- [ ] Fetch all open-access sources
- [ ] Build chunk corpus
- [ ] Finalize reference documents
- [ ] Set up generation pipeline

### Week 2 (Apr 17-23): Data Generation
- [ ] Generate 800-1000 instruction pairs
- [ ] Run LLM judge validation
- [ ] Manual spot-check (20+ random entries)
- [ ] Build eval set (100+ questions)

### Week 3 (Apr 24-30): Training + Evaluation
- [ ] QLoRA fine-tune on Gemma 4
- [ ] Run eval benchmarks
- [ ] Iterate on data if needed
- [ ] Compare vs base model

### Week 4 (May 1-7): Demo + Polish
- [ ] Build Gradio app
- [ ] Add multimodal features
- [ ] Deploy to HuggingFace Spaces
- [ ] Create Kaggle notebook

### Week 5 (May 8-17): Submission
- [ ] Record demo video
- [ ] Write technical writeup
- [ ] Final testing and polish
- [ ] Submit before May 18
