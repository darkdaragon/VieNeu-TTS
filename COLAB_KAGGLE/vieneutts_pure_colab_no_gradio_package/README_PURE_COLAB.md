# VieNeuTTS Pure Colab Package

No Gradio. Use the companion notebook to choose files with Colab upload widget, select engine preset, and render using the same manifest/cache/final_parts logic from app.py. Success outputs copy only audio + subtitle; reports are kept only for incomplete/debug.


## 2026-06 update: optimized Full Quality v2 cache/engine

### Engine modes

Set these environment variables in Colab config:

```python
VIENEU_DEVICE_MODE = "gpu"
VIENEU_QUALITY_MODE = "full_quality_v2"
VIENEU_ENGINE_STRATEGY = "shared_single_engine"   # safest
# or:
VIENEU_ENGINE_STRATEGY = "per_worker_engine"      # true parallel; engines load sequentially to reduce OOM risk
VIENEU_MEMORY_UTIL = "0.30"                       # recommended when per_worker_engine + 2 workers
VIENEU_TP = "1"
VIENEU_EMOTION = "storytelling"                  # stable long-form narration; use "natural" for casual speech
```

`VIENEU_EMOTION` is passed into `Vieneu(...)`, so changing it invalidates segment cache keys and prevents old natural/storytelling audio from mixing.

`per_worker_engine` creates one VieNeu/LMDeploy engine per Python worker thread, but engine loading is serialized under a lock. This avoids multiple workers converting/loading the model at the exact same time.

### Compact cache policy

- `failed/` markers are kept.
- `success/` raw segment mp3/json are temporary working files only.
- After each render pass, contiguous successful ranges are built into `final_parts/`.
- Single isolated successful segments are also stored as `final_parts/part_x_x`.
- Raw `success/` files covered by `final_parts/` are deleted to keep Drive light.
- If failed segments remain, final audio is not built. Retry failed first, then final build uses cached `final_parts/` plus newly fixed gaps.
- Success output folder contains only audio + subtitle. Debug reports are written only for incomplete/failed runs.
