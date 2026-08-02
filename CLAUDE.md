## Project info


## Environment

Python venv at `.venv/`. Activate with `.venv\Scripts\activate` (Windows). CUDA is available and used automatically (`DEVICE = "cuda" if torch.cuda.is_available() else "cpu"`).

## Subagents

```
Do NOT spawn subagents unless I explicitly ask for parallel work. Do all tasks inline.
```

Use agents only for:
- Parallel searches across unrelated areas
- Risky experiments you want isolated (worktree)
- Genuinely independent tasks that benefit from concurrent execution

Never for: sequential steps, code review of your own recent edits, single-file tasks.

## File/folder index

- Refer `index.md`

## Current State

**Done:**

- Literature survey - production, calibration, research models
- Three candidate plans drafted and verified against primary sources (`docs/reference/2026-08-02-*`)
- SSAFE read at the paper body; corrections recorded in `docs/reference/2026-08-02-ssafe-primary-read.md` §5
- Bottleneck inventory: `docs/notes/2026-08-02-bottlenecks.md`
- **`0004` written and accepted** — `docs/decisions/0004-adaptation-hypothesis-demo-build.md`. The claim is now **"adaptation costs N images"**, not "it generalizes". Supersedes `0003` §4.3, §5, most of §6
- Evidence for it: `docs/research/2026-08-02-patch-inference-and-encoder-choice.md`
- Execution plan: `docs/notes/2026-08-02-build-plan.md` - **start here**

**Next:**

- Execute the build plan in stage order. **S1 (data) is the highest risk and comes first**; S1-S5 are the demo and are never cut
- Nothing is on disk yet: `checkpoints/`, `dataset/`, `data_raw/`, `outputs/`, `runs/` all absent
- First code change: `data/download.py:68` takes all seven COCO_AI columns instead of three. Second: drop the person-caption filter
- Deferred, not on the critical path: DDA (arXiv:2505.14359) at the body; Chai et al. at the body; feature-layer selection

**Backbone is settled — do not reopen.** Frozen CLIP ViT-L/14 @224. DINOv2 and PE-Core dropped with reasons in `0004` §10; SigLIP2 is a stretch arm only, strictly after E1-E3. The frozen backbone is load-bearing: fine-tuning would falsify the headline claim, not merely cost time.

**Standing constraint:** no data addition may put a real source into training without matched fakes at comparable resolution and content domain (`bottlenecks.md` §2.4). Breaking the COCO_AI 1:1 pairing is the fastest way to a good validation number and a failed demo.

