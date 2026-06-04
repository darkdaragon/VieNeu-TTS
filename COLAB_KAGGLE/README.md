Repo-native Kaggle bundle.

There are now 2 Kaggle paths inside this folder:

1. Private repo path
- Notebook: `run_vieneutts_kaggle_t4x2_fast_standard_v29_longtext_modes_segments_segment_srt_refinfer.ipynb`
- Uses Kaggle User Secrets `GITHUB_TOKEN` and optionally `GITHUB_USERNAME`
- Clones private branch `darkdaragon/VieNeu-TTS:feature/my-feature`

2. Zip fallback path
- Notebook: `run_vieneutts_kaggle_t4x2_fast_standard_v29_longtext_modes_segments_segment_srt_refinfer_zip_bundle.ipynb`
- Preferred bundle zip: `vieneutts_myfeature_kaggle_bundle.zip`
- Legacy bundle zip: `vieneutts_kaggle_repo_bundle.zip`
- Nested package zip: `vieneutts_pure_colab_no_gradio_package.zip`
- Does not need git clone on Kaggle
- `vieneutts_myfeature_kaggle_bundle.zip` is a full snapshot of the current private branch `feature/my-feature`, excluding git metadata/cache/temp files and nested zip files
- Its top-level folder is `VIENEU_MYFEATURE_REPO`, so Kaggle auto-extract is less likely to collide with existing `COLAB_KAGGLE` or `VieNeu-TTS` dataset folders
- If Kaggle auto-extracts the bundle instead of exposing the `.zip` file, the zip fallback notebook will detect the extracted repo folder and copy it to `/kaggle/working/VieNeu-TTS`

Both paths default to JSON segment mode `segments_native_infer`, which uses `FastVieNeuTTS.infer_segments()` for:

- narration -> southern female narration route
- female dialogue -> southern female dialogue route
- male dialogue -> southern male dialogue route

Per-segment SRT output is preserved.
