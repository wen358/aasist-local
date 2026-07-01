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

## Next Step

1. Verify pretrained AASIST on a small manifest locally.
2. Run pretrained AASIST on `cross_2021_df_direct_500.csv`.
3. Move training/evaluation to Kaggle/Colab GPU.
4. Only after AASIST baseline is reliable, add CDBD-style probe logic on top.
