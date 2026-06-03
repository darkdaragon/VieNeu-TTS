Repo-native Kaggle bundle.

There are now 2 Kaggle paths inside this folder:

1. Private repo path
- Notebook: `run_vieneutts_kaggle_t4x2_fast_standard_v29_longtext_modes_segments_segment_srt_refinfer.ipynb`
- Uses Kaggle User Secrets `GITHUB_TOKEN` and optionally `GITHUB_USERNAME`
- Clones private branch `darkdaragon/VieNeu-TTS:feature/my-feature`

2. Zip fallback path
- Notebook: `run_vieneutts_kaggle_t4x2_fast_standard_v29_longtext_modes_segments_segment_srt_refinfer_zip_bundle.ipynb`
- Bundle zip: `vieneutts_kaggle_repo_bundle.zip`
- Nested package zip: `vieneutts_pure_colab_no_gradio_package.zip`
- Does not need git clone on Kaggle
- `vieneutts_kaggle_repo_bundle.zip` is intended to be a full snapshot of the current private branch `feature/my-feature`, excluding only git metadata/cache/temp files

Both paths default to JSON segment mode `segments_native_infer`, which uses `FastVieNeuTTS.infer_segments()` for:

- narration -> southern female narration route
- female dialogue -> southern female dialogue route
- male dialogue -> southern male dialogue route

Per-segment SRT output is preserved.
