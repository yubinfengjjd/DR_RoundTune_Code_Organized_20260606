# DR RoundTune

Code for quality-aware diabetic retinopathy grading from color fundus photographs.

## Methodology

1. `01_data_preparation`: image preprocessing, manifest construction, and retinal feature extraction.
2. `02_model_training`: quality pretraining, LoRA adaptation, and image-level DR grading.
3. `03_primary_evaluation`: internal and external evaluation with confidence intervals.
4. `04_ablation_and_sota`: seed/rank ablations and comparison with reference methods.
5. `05_quality_pretraining`: image-quality auditing and quality-stratified evaluation.
6. `06_uncertainty_triage`: test-time augmentation, uncertainty estimation, and referral triage.
7. `07_interpretability`: case analysis, attribution summaries, and interpretability figures.
8. `08_calibration_and_dca`: probability calibration and decision-curve analysis.

## Environment

```bash
pip install -r requirements.txt
```

## Training

Quality pretraining:

```bash
python 02_model_training/train_roundtune_cpf.py \
  --task qual \
  --project_root <project-root> \
  --qual_manifest <quality-manifest.csv>
```

DR grading:

```bash
python 02_model_training/train_roundtune_cpf.py \
  --task grade \
  --project_root <project-root> \
  --grade_manifest <grading-manifest.csv>
```

## License

This project is released under the MIT License. See `LICENSE`.
