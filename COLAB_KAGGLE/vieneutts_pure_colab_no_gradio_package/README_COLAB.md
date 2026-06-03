# VieNeuTTS Colab High Quality v2 / LMDeploy/Fast Package

Bản này dùng `Vieneu(mode="fast")` để đi backend GPU/High Quality v2 / LMDeploy. Không dùng `mode="standard"` vì standard sẽ rơi về GGUF/llama_cpp.

Colab nên chạy app bằng `uv run python app.py` từ thư mục `/content/VieNeu-TTS` để dùng đúng env GPU được tạo bởi `uv sync --group gpu`.

Nếu log còn hiện `.gguf` hoặc `llama_context`, tức là đang sai backend.

# VieNeuTTS Colab HF-style package

Chạy toàn bộ trên Google Colab T4, không cần local.

## Cách dùng nhanh

1. Bật Runtime > Change runtime type > GPU > T4.
2. Upload file zip này lên Colab.
3. Chạy:

```bash
!unzip -o vieneutts_colab_hf_style_package.zip -d /content/vieneutts_app
%cd /content/vieneutts_app
!bash run_colab.sh
```

App Gradio sẽ mở link public giống Hugging Face Space. FFmpeg chạy trực tiếp trong Colab để concat audio, đo duration, silencedetect và build subtitle.

## Lưu output/cache vào Google Drive

Trong notebook, mount Drive trước khi chạy app:

```python
from google.colab import drive
drive.mount('/content/drive')
import os
os.environ['GOOGLE_DRIVE_OUTPUT_ROOT'] = '/content/drive/MyDrive/VieNeuTTS_Output'
```

Nếu không dùng Hugging Face Dataset, app sẽ fallback lưu outputs/cache vào Google Drive/local folder này.

## Dùng Hugging Face Dataset thay Drive

Set env trước khi chạy app:

```python
import os
os.environ['HF_TOKEN'] = 'hf_xxx'
os.environ['DATASET_REPO'] = 'username/dataset_name'
```

## Voice mapping

- `doan`, `Đoan`, `hoaimy`, `Hoài My` -> Đoan nữ miền Nam
- `vinh`, `Vĩnh`, `namminh`, `Nam Minh` -> Vĩnh nam miền Nam

Pitch/rate giữ trong JSON/cache/report nhưng VieNeu render không dùng pitch/rate như Edge TTS.


## High Quality default
This package defaults to `mode="fast"` with `VIENEU_BACKBONE_REPO=pnnbao-ump/VieNeu-TTS-v2` for full VieNeu-TTS-v2 high-quality GPU/LMDeploy inference.
