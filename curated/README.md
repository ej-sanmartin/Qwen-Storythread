# Curated PIPPA subset

The 80 PIPPA conversations used in the pilot, so you can skip
[step 2](../README.md#2-curate-pippa) and rebuild the exact training data.

The files contain mature content, so they are hosted as a Hugging Face dataset
marked not-for-all-audiences, not in this Git repository:

```bash
hf download HF_USER/storythread-pippa-subset --repo-type dataset --local-dir curated
```

| File | What it is |
|---|---|
| `pippa-subset.jsonl` | 80 PIPPA rows in PIPPA's shape, cut to the fields `prepare` reads |
| `review.jsonl` | the 80 approved review rows: `scene_id`, `start`/`end` turn window, `system` card, `group`, `tags` |

Use them in place of the full PIPPA file and your own `review.jsonl`:

```bash
python -m storythread.data prepare curated/pippa-subset.jsonl curated/review.jsonl \
  authored/scenes.jsonl work/datasets --valid-share 0.1 --authored-share 0.25
```

This rebuilds `work/datasets` byte for byte (checked on the pilot: all
`train`, `valid`, and `manifest` files and `summary.json` are identical to the
ones the adapters were trained on). `tests/test_publish.py` checks the same
property on the fixture.

`python -m storythread.publish SOURCE REVIEW OUTDIR` regenerates the two files
from a full PIPPA file and a review file. Use it to share your own subset.

## Source and license

PIPPA (Gosling, Dale, Zheng), `pippa_deduped.jsonl`, Apache-2.0 per its
[dataset card](https://huggingface.co/datasets/PygmalionAI/PIPPA). Rows are
cut to a few fields; conversation text is not edited. See `NOTICE`.

## Content warning

PIPPA is user-written fiction between people and chatbots. This subset
includes mature themes. About ten of the 80 windows are sexually explicit in
the character card, the conversation, or both, and some include violence.
Nothing here is suited to minors. Do not use it in a setting that needs
all-ages content, and filter or replace windows (step 2, path B) if so.

## What was checked

- Each of the 80 windows is marked `approved: true` in `review.jsonl`.
- A keyword scan after approval looked for emails, phone numbers, `@handles`,
  and URLs. It found none, plus one false positive ("Awww...").
- The same scan flagged sexual, violence, and minor-related terms. The flagged
  passages were read in context. The scan is not a review of every line.
- PIPPA applies its own redaction and flags the dataset as not for all
  audiences. This subset adds no further redaction.

If you find personal data or content that should not be here, open an issue
and the row can be removed (this changes the reproduced datasets).
