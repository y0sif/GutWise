# Quick demo path (CPU-only, no Pro subscription)

The Space runs on `cpu-basic` (free, ~2 min per turn). To make the video
fast and minimalistic, the app reads `app/cached_examples.json` and
displays the responses instantly in a **Quick demo (instant)** tab. The
cached responses are **real outputs from `y0sif/GutWise`** — they were
captured during the 3-run paired evaluation that produced the headline
numbers (overall 4.769 ± 0.028, red-flag 4/4 every run).

**The file is already populated** from `scripts/evaluate/results/`. No
Colab run required. The four entries are:

| Label | Source eval id | Side-by-side vs base |
|---|---|---|
| IBS vs IBD | eval_001 | no |
| Rome IV criteria | eval_002 | no |
| FODMAP basics | eval_006 | no |
| Red-flag: bloody stool | eval_047 | yes |

The responses are pulled from `gutwise_v2_results_run1.json` (the
highest-overall run, 4.796) and `baseline_e4b_results.json` for the
side-by-side baseline column.

## How to regenerate (only if you change the eval set)

If you re-run the eval with different held-out questions, regenerate the
cached file from disk:

```bash
cd ~/Projects/GutWise
python3 - <<'PY'
import json
ft = {r['prompt_id']: r for r in json.load(open('scripts/evaluate/results/gutwise_v2_results_run1.json'))}
base = {r['prompt_id']: r for r in json.load(open('scripts/evaluate/results/baseline_e4b_results.json'))}
picks = [
    ('eval_001', 'IBS vs IBD', False),
    ('eval_002', 'Rome IV criteria', False),
    ('eval_006', 'FODMAP basics', False),
    ('eval_047', 'Red-flag: bloody stool (side-by-side vs base)', True),
]
cached = []
for pid, label, side in picks:
    r = ft[pid]
    cached.append({
        'label': label, 'prompt': r['prompt'], 'finetuned': r['response'],
        'base': base[pid]['response'] if side else None,
    })
with open('app/cached_examples.json', 'w', encoding='utf-8') as f:
    json.dump(cached, f, indent=2, ensure_ascii=False)
print('wrote', len(cached), 'entries')
PY
```

## Pushing changes to the Space

```bash
git add app/cached_examples.json
git commit -m "data: refresh cached demo responses"
git push origin main
```

Then re-sync the Space:

```python
# in a Python session with HF_TOKEN set
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj="app/cached_examples.json",
    path_in_repo="app/cached_examples.json",
    repo_id="y0sif/GutWise",
    repo_type="space",
    commit_message="data: refresh cached demo responses",
)
api.restart_space("y0sif/GutWise")
```

## Why this is honest

- The responses are **the actual outputs** that produced the headline
  eval numbers (overall 4.769 ± 0.028, red-flag 4/4). They're not
  fabricated or re-generated for the demo — they're the same data the
  judges' Haiku-based judge scored.
- The Quick demo tab text says so explicitly: *"Pre-generated outputs
  from this fine-tune — real responses, captured on a GPU pass and shown
  instantly here so the demo doesn't wait on CPU inference."*
- The Chat tab still serves live inference for anyone who wants to wait.

## Fallback: no cached tab, just edit the recording

If you'd rather skip the cached tab entirely, record one live CPU
interaction and cut the dead air in post:

```bash
uvx auto-editor recording.mp4 --silent-threshold 0.04 --silent-speed 99999 -o trimmed.mp4
```

A 3-minute recording (2 min waiting + 1 min answer) becomes ~30 seconds
of content. Pad with title cards and you're at 60 seconds.
