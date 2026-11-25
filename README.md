# Behavioral Selection and Blending through Noise Fields as an Abstract Model of Neuromodulation

Code for Figures 1–6 and 8 of the paper "Behavioral Selection and Blending through Noise Fields as an Abstract Model of Neuromodulation". 

## Setup

Tested on Ubuntu (WSL), Python 3.12.3, PyTorch 2.6.0 (CUDA 12.4). Run all commands from the repository root.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

Training and evaluation use the GPU by default. Add `--device cpu` to run on the CPU. Figure 1 picks the device by itself.

## Figure 1

A 1D test that switches between sin and cos by the noise condition. The first line trains and evaluates. The second line makes the figure.

```bash
python -B fig01_sin_cos/neuromod_shape_axis.py --out-dir fig01_sin_cos/results
python -B fig01_sin_cos/plot_shape_axis.py --data fig01_sin_cos/results/shape_axis.npz
```

## Figures 2 and 3

The environment layout and the reference noise fields. No training is needed.

```bash
python -B fig02_environment/plot.py
python -B fig03_reference_noise_fields/plot.py
```

## Figures 4, 5, 6 and 8

Train and evaluate first. Interpolation (Figure 6) and lesion (Figure 8) use the models and results of the intensity experiment (Figure 5), so keep this order.

```bash
run_name=reproduce01
python -B fig04_behavior/run.py train --run-name "$run_name"
python -B fig04_behavior/run.py evaluate --run-name "$run_name"
python -B fig05_intensity/run.py train --run-name "$run_name"
python -B fig05_intensity/run.py evaluate --run-name "$run_name"
python -B fig06_interpolation/run.py evaluate --run-name "$run_name"
python -B fig08_lesion/run.py evaluate --run-name "$run_name"
```

Results are saved in `runs/<run name>/` under `01_behavior`, `02_intensity`, `03_interpolation` and `04_lesion`. Then make the figures from the saved results.

```bash
python -B paper_figures/plot_distribution_trace.py \
  --run-dir "01_behavior/runs/$run_name/evaluations/c1_run01" \
  --out-dir "figures/$run_name/behavior"
python -B fig05_intensity/plot.py \
  --run-dir "02_intensity/runs/$run_name/evaluations/main" \
  --out-dir "figures/$run_name/intensity"
python -B paper_figures/neuromod_draft_figs.py --figure interpolation \
  --interpolation-csv "03_interpolation/runs/$run_name/evaluations/main/results.csv" \
  --out-dir "figures/$run_name/interpolation"
python -B paper_figures/neuromod_draft_figs.py --figure lesion \
  --lesion-csv "04_lesion/runs/$run_name/evaluations/main/results.csv" \
  --out-dir "figures/$run_name/lesion"
```

## Notes

- Figures are saved under `figures/` as PNG and PDF. Figure 1 is PNG only by default.
- Training and evaluation results are never overwritten. Use a new run name to run again.
- A saved model can be evaluated only with the same environment (Python, NumPy, PyTorch, CUDA, GPU) and the same code as in training.

## License

MIT. See `LICENSE`.
