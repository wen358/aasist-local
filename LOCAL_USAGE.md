# Local AASIST Baseline Notes

This folder is cloned from the official AASIST implementation:

https://github.com/clovaai/aasist

The repository includes pretrained AASIST weights under `models/weights/`.

## Why This Baseline

- It is the official implementation for "AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks".
- The README reports ASVspoof 2019 LA eval performance around `EER: 0.83%` for AASIST.
- It is a stronger anti-spoofing baseline than the exploratory ResNet18 spectrogram classifier.

## Local Adaptations

Two local files were added:

- `eval_manifest.py`: evaluates pretrained AASIST on the CSV manifests used by the CDBD project.
- `local_configs/AASIST_manifest_eval.conf`: tracked config for manifest-based evaluation on local, Kaggle, or Colab.
- `summarize_scores_by_codec.py`: summarizes an existing `eval_manifest.py --output-scores` CSV by codec without rerunning AASIST inference.
- `codec_calibration_experiment.py`: runs held-out codec-aware score-shift calibration on an existing score CSV.

`eval_manifest.py` avoids extra dependencies such as `soundfile` by decoding non-WAV files through `ffmpeg` and reading WAV data with the Python standard library.
For manifest-based evaluation, `database_path` is not used because each audio path comes from the CSV manifest.

## Smoke Test

From this directory:

```powershell
& "D:\audio deepfake detecation\.venv\Scripts\python.exe" eval_manifest.py `
  --manifest "D:\audio deepfake detecation\github_upload\cdbd-audio\manifests\cross_2021_df_direct_2.csv" `
  --config local_configs\AASIST_manifest_eval.conf `
  --weights models\weights\AASIST.pth `
  --device cpu `
  --batch-size 1
```

## Evaluate Existing Manifests

ASVspoof 2021 DF direct 500-per-class subset:

```powershell
& "D:\audio deepfake detecation\.venv\Scripts\python.exe" eval_manifest.py `
  --manifest "D:\audio deepfake detecation\github_upload\cdbd-audio\manifests\cross_2021_df_direct_500.csv" `
  --config local_configs\AASIST_manifest_eval.conf `
  --weights models\weights\AASIST.pth `
  --output-scores "D:\audio deepfake detecation\github_upload\cdbd-audio\outputs\metrics\aasist_2021_df_direct_500_scores.csv" `
  --device cpu `
  --batch-size 4
```

ASVspoof 2019 LA compressed dev subset:

```powershell
& "D:\audio deepfake detecation\.venv\Scripts\python.exe" eval_manifest.py `
  --manifest "D:\audio deepfake detecation\github_upload\cdbd-audio\manifests\local_500_per_class_dev_compressed.csv" `
  --config local_configs\AASIST_manifest_eval.conf `
  --weights models\weights\AASIST.pth `
  --output-scores "D:\audio deepfake detecation\github_upload\cdbd-audio\outputs\metrics\aasist_2019_dev_compressed_scores.csv" `
  --device cpu `
  --batch-size 4
```

CPU evaluation is expected to be slow. Use Kaggle or Colab GPU for full-scale runs.

## Summarize Existing Scores By Codec

If `eval_manifest.py` was already run with `--output-scores`, compute codec-level metrics without rerunning the model:

```powershell
& "D:\audio deepfake detecation\.venv\Scripts\python.exe" summarize_scores_by_codec.py `
  --scores "D:\audio deepfake detecation\github_upload\cdbd-audio\outputs\metrics\aasist_2021_df_direct_500_scores.csv" `
  --output "D:\audio deepfake detecation\github_upload\cdbd-audio\outputs\metrics\aasist_2021_df_direct_500_by_codec.csv"
```

## Codec-Aware Calibration Experiment

Use a stratified calibration split from an existing score CSV, learn score shifts for selected codecs, and evaluate on the held-out split:

```powershell
& "D:\audio deepfake detecation\.venv\Scripts\python.exe" codec_calibration_experiment.py `
  --scores "D:\audio deepfake detecation\github_upload\cdbd-audio\outputs\metrics\aasist_2021_df_direct_500_scores.csv" `
  --output-json "D:\audio deepfake detecation\github_upload\cdbd-audio\outputs\metrics\codec_calibration_2021_df_500_each_target_codecs.json" `
  --output-shifted-scores "D:\audio deepfake detecation\github_upload\cdbd-audio\outputs\metrics\codec_calibration_2021_df_500_each_shifted_scores.csv" `
  --target-codecs high_ogg high_mp3 low_m4a `
  --cal-per-class 20 `
  --seed 0
```

This is an analysis tool, not a training script. Report both the `before` and `after` held-out metrics; do not treat calibration as useful unless held-out EER improves.

## Next Step

1. Verify pretrained AASIST on a small manifest locally.
2. Run pretrained AASIST on `cross_2021_df_direct_500.csv`.
3. Move training/evaluation to Kaggle/Colab GPU.
4. Only after AASIST baseline is reliable, add CDBD-style probe logic on top.
