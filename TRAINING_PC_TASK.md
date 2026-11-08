# Task for the training PC: finish the missing values

**Run this on the PC that trained the models** (WSL2, RTX 3050, user `MustafaJK`, conda env at
`/root/miniconda3`). That PC is the only one that has `outputs/cgan_*/checkpoints/progress.json`,
`outputs/cgan_*/history/loss_history.csv` and the original
`outputs/cgan_B_128mp/checkpoints/landmark_regressor.weights.h5`. The laptop has none of them.

The laptop already finished items 5–9, 15 and 16 (see `missing-values-answers.pdf`). This task closes
the rest: **1, 2, 3, 4, 5 (original weights), 6, 10–12, 13 and 14**.

Estimated time: about 10 minutes, no training.

---

## Step 1 — Get the branch (inside WSL)

```bash
cd /path/to/arasl-sign-language-gan        # the repo folder that contains outputs/
git stash -u                               # only if there are local changes; keep outputs/ untouched
git fetch origin
git checkout reviewer-missing-values
git pull origin reviewer-missing-values
```

`outputs/` is gitignored, so checking out the branch does not touch it.

## Step 2 — Check the files exist before running

```bash
ls -l outputs/cgan_{A_128,B_128mp,C_128struct,F_128fusion,G_128plus}/checkpoints/progress.json
ls -l outputs/cgan_{A_128,B_128mp,C_128struct,F_128fusion,G_128plus}/history/loss_history.csv
ls -l outputs/cgan_B_128mp/checkpoints/landmark_regressor.weights.h5 outputs/cgan_B_128mp/landmarks_128px.npy
conda env list                              # note the name of the MediaPipe env
```

If a file is missing, write down which one and continue. The script still collects everything else.

If the MediaPipe env is **not** called `mp`, change `conda activate mp` near the end of
`collect_training_machine_facts.sh` to the real name.

## Step 3 — Run the collector

```bash
bash collect_training_machine_facts.sh 2>&1 | tee reports/paper/logs/collect_training_machine.log
```

It writes:

| Output | Answers item |
|---|---|
| `reports/paper/results/training_machine/cgan_*/progress.json`, `loss_history.csv` | 1 (all five histories, including A) |
| `reports/paper/results/training_machine/progress_summary.txt` | 2 (A epoch timing: mean, min, max, total s, total h), 3 (A `gd_ratio`), 4 (F/G steps per epoch → batch size) |
| `reports/paper/results/landmark_regressor_eval.json` + `training_machine/landmark_regressor_eval.log` | 5 (original regressor MSE/MAE, normalised + pixels, vs constant baseline) |
| `training_machine/regressor_file_times.txt` | 6 (when the regressor weights were written) |
| `training_machine/environment.txt` | 10–12 (CPU, RAM, GPU, driver, CUDA, cuDNN, Keras, WSL, kernel, Windows build), 4 (shell history `ARASL_BATCH`), 14 (MediaPipe version, parquet date) |
| `requirements_tf.txt`, `requirements_mp.txt` (repo root) | 13 |

## Step 4 — Extra evidence the script does not collect

Run these and paste the output into `reports/paper/results/training_machine/extra.txt`:

```bash
{
  echo "## Batch size evidence for F and G (item 4)"
  grep -h ARASL_BATCH ~/.bash_history ~/.zsh_history 2>/dev/null
  grep -rn "ARASL_BATCH\|batch" outputs/cgan_F_128fusion/*.log outputs/cgan_G_128plus/*.log 2>/dev/null | head
  echo "## Regressor training time (item 6)"
  grep -rn -i "regressor" outputs/cgan_B_128mp/*.log reports/paper/logs/*B*.log 2>/dev/null | head -40
  echo "## Dataset download date (item 14)"
  ls -l --time-style=full-iso data/arasl.parquet ~/.cache/huggingface/hub 2>/dev/null
  ls -l --time-style=full-iso ~/.cache/huggingface/hub/datasets--pain--ArASL_Database_Grayscale 2>/dev/null
} > reports/paper/results/training_machine/extra.txt 2>&1
cat reports/paper/results/training_machine/extra.txt
```

## Step 5 — Commit and push the results

```bash
git add reports/paper/results/training_machine reports/paper/results/landmark_regressor_eval.json \
        reports/paper/logs/collect_training_machine.log requirements_tf.txt requirements_mp.txt
git commit -m "Add training-PC histories, timing, regressor eval and environment"
git push origin reviewer-missing-values
```

Do **not** add `outputs/`, `*.npy` or `*.h5` — they are large and gitignored on purpose.

## Step 6 — Report back

After pushing, the laptop will pull and update `missing-values-answers.html/.pdf`. Report:

1. Which files in Step 2 were missing, if any.
2. Any error in `collect_training_machine.log`.

---

### Done when

- [ ] `progress_summary.txt` shows Model A `epoch_seconds` (item 2) and `gd_ratio` (item 3)
- [ ] F and G batch size is confirmed by steps per epoch or shell history (item 4)
- [ ] `landmark_regressor_eval.json` exists for the **original** weights (item 5)
- [ ] `environment.txt` has real CPU, RAM, driver, CUDA, cuDNN, WSL and Windows values (items 10–12)
- [ ] `requirements_tf.txt` and `requirements_mp.txt` are committed, with protobuf versions (item 13)
- [ ] MediaPipe version and dataset date are recorded (item 14)
- [ ] Everything is pushed to `reviewer-missing-values`
