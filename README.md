# MRI-based tau prediction

**Kim, D. et. al., Deep learning-based prediction of regional brain Tau PET burden from MRI, Alzheimer's & Dementia 2026, https://doi.org/10.1002/alz.71911**

Source code is available. The public release of pretrained weights is pending internal institutional review.

Please contact the corresponding authors for any questions or requests.# tau_prediction


Train a 3D DenseNet264 on MPRAGE and FLAIR scans to predict eight regional tau measures.

## Setup
 Install the dependencies with `python -m pip install -r requirements.txt` and use a CUDA-capable PyTorch build for training.

For each cohort (`adni`, `oasis`, `a4`, and `habs`), use the synthetic examples in `dataset/<cohort>/fold_0/` as a guide. Save your authorized manifests as `train.json` and `valid.json` alongside the templates. Replace the example entries with your visit metadata, Braak targets, and paths to accessible preprocessed MPRAGE and FLAIR volumes. Relative image paths are resolved from the repository root. Git ignores the local manifests and includes the `*.template.json` examples.

## Train

Run from the repository root.

```sh
python train.py -fold 0 -model dense -gpu [GPU_ID] -loss_type mse_corr_residual_comp
```

 Set `TAU_OUTPUT_DIR` to use a different location. To enable W&B logging, set `WANDB_MODE=online` and provide your W&B credentials and entity.
