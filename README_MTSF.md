# MTSF: Forecasting multivariate time series with missing values

A refactored, self-contained codebase for this project. The old staged pipeline
(`scripts/`, `configs/`, `baselines/`, `runs/`) is left as it was for
provenance. All new work lives in:

| path | content |
|---|---|
| `mtsf/data.py` | S4M-protocol data pipeline: block-missingness generation, 0.7/0.1/0.2 split, train-stat z-score |
| `mtsf/models/mars.py` | **MARS**, the proposed model |
| `mtsf/models/baselines.py` | PatchTST, iTransformer, DLinear (applied to zero-filled input) |
| `mtsf/models/crib_wrap.py` | wrapper around the official CRIB code (2025 SOTA, not vendored) |
| `scripts_mtsf/s4m_baseline.py` | runs the **official S4M code** on exactly the same masks |
| `mtsf/train.py` | shared trainer: early stopping on validation loss over observed targets |
| `mtsf/queue.py` | resumable single-GPU job queue |
| `mtsf/summarize.py`, `mtsf/paper_numbers.py` | result tables, with the S4M paper numbers for reference |
| `configs_mtsf/base.yaml` | default configuration |

## Protocol (same as S4M, ICLR 2025)

* Missingness is injected into the whole series before splitting. Each time step
  (*point*) or each (time step, variable) pair (*variable*) starts a block of
  5 missing steps with probability `r`. With `r=0.06` the realised missing rate
  is about 26%, consistent with Table 8 of the S4M paper.
* The split is chronological 0.7/0.1/0.2. Data are z-scored with statistics of
  the partially observed training part. The horizon is 96.
* Training and validation losses use **observed future values only**. No
  ground truth that would be missing is ever used for training or model
  selection.
* Test MSE/MAE are computed against the **complete** future in normalised
  space, over all test windows (no `drop_last`).
* The official S4M masks/seeds were never published. Every model here,
  including the official S4M code, is run on the same fixed masks
  (`mask_seed=2024`, cached in `data/masks/`).

## MARS (Mask-Aware Reliability-weighted patch tranSformer)

1. **ObsNorm**: instance normalisation from observed entries only.
2. **Gap-aware patch tokens**: [linear interpolation, mask, log distance-to-observation] per patch.
3. **Reliability-biased attention**: each head learns how strongly to down-weight poorly observed patches.
4. **Variate mixing**: zero-initialised cross-variable attention on pooled per-variable states, after every layer.
5. **Missingness-consistent masking (MCM)**: during training, extra blocks with the
   dataset's own missing pattern are hidden. The model forecasts from the reduced
   input and reconstructs the hidden *observed* values.

Each component has a switch in the config (`use_obsnorm`, `use_gap_tokens`,
`use_rel_bias`, `use_var_mix`, `mcm_rate`) for ablations.

## Usage

```bash
conda activate mtsf            # torch 2.5.1 + cu121
export PYTHONNOUSERSITE=1
# data: thuml/Time-Series-Library on Hugging Face -> data/raw/{ETTh1,weather,electricity,traffic}.csv
python -m mtsf.cli --set model=mars dataset=etth1 pattern=point rate=0.06 seq_len=96
python -m mtsf.queue jobs/round1.txt --workers 2      # resumable batch
python -m mtsf.summarize --prefix r1
# official S4M on the same masks
python scripts_mtsf/s4m_baseline.py prepare
python scripts_mtsf/s4m_baseline.py run --dataset etth1 --lr 0.005
```
