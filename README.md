# Failure-Guided Sim2Real2Sim on a Low-Cost Manipulator (SO-101)

UT Dallas course project. The proposal is [Project_Proposal_Doc.pdf](Project_Proposal_Doc.pdf).

| Path | What it is |
|---|---|
| [`s2r2s/`](s2r2s/README.md) | **Current code.** Milestone 1: SO-101 Push-T learned with RL in MuJoCo (96.6 % on held-out scenes), the trained policy, and tools for evaluation, video and camera studies. Start with [s2r2s/README.md](s2r2s/README.md) and [s2r2s/ROADMAP.md](s2r2s/ROADMAP.md). |
| `src/so101_m1/`, `scripts/`, `tests/`, `STATUS.md`, `PLAN.md`, `*_CHECKPOINT.md`, ... | Earlier *Push-Turn-Park* attempt (WSL + mjlab + GPU physics), kept for reference. |
| `artifacts/` | That attempt's diagnostic plots and JSON. Bulky `.npz`/`.pt` files are not in git. |
| `assets/menagerie/` | MuJoCo Menagerie SO-101 model. Not in git: `s2r2s/scripts/setup.ps1` fetches it at the pinned revision. |

## Fresh clone (Windows, PowerShell)

```powershell
git clone <repository-url> Sim2Real2Sim
cd Sim2Real2Sim\s2r2s
.\scripts\setup.ps1                        # fetches the SO-101 model, creates .venv, runs the tests
.\scripts\play.ps1 runs\tee_v1\best.pt     # watch the trained policy
```
