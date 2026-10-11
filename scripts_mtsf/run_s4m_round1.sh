#!/bin/bash
# Official S4M on the same masks (L=96, point, r=0.06). Small lr sweep, chosen by validation loss.
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mtsf
cd "$(dirname "$0")/.."
run() { [ -f "results_mtsf/$1.json" ] || python scripts_mtsf/s4m_baseline.py run --name "$@"; }
run s4m__etth1__point__L96__lr0.005 --dataset etth1 --lr 0.005
run s4m__etth1__point__L96__lr0.001 --dataset etth1 --lr 0.001
run s4m__weather__point__L96__lr0.001 --dataset weather --lr 0.001
