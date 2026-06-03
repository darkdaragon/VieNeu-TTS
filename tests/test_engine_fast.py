import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from vieneu.fast import FastVieNeuTTS

@pytest.fixture
def mock_fast_tts():
    with patch("lmdeploy.pipeline") as mock_pipeline, \
         patch("lmdeploy.GenerationConfig"), \
         patch.object(FastVieNeuTTS, '_warmup_model'), \
         patch("vieneu.standard.BaseVieneuTTS._load_codec") as mock_codec:
        
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.return_value = [MagicMock(text="codes"), MagicMock(text="codes")]
        mock_pipeline.return_value = mock_pipeline_instance
        
        with patch.object(FastVieNeuTTS, '_load_codec'):
            tts = FastVieNeuTTS(backbone_device="cuda")
            tts.codec = MagicMock()
            tts.codec.device = "cuda"
            tts.codec.decode_code.return_value = np.zeros((1, 1, 1000))
            return tts

def test_fast_init(mock_fast_tts):
    assert mock_fast_tts.backbone is not None
    assert mock_fast_tts.device == "cuda"

def test_fast_infer(mock_fast_tts):
    with patch("vieneu_utils.phonemize_text.phonemize_with_dict", return_value="phonemes"), \
         patch.object(mock_fast_tts, '_decode', return_value=np.zeros(1000)):
        audio = mock_fast_tts.infer("Xin chào", ref_codes=[1, 2], ref_text="ref")
        assert isinstance(audio, np.ndarray)
        assert len(audio) == 1000

def test_fast_infer_batch(mock_fast_tts):
    texts = ["Text 1", "Text 2"]
    with patch("vieneu.fast.phonemize_batch", return_value=["p1", "p2"]) as mock_ph_batch, \
         patch.object(mock_fast_tts, '_decode', return_value=np.zeros(1000)):
        results = mock_fast_tts.infer_batch(texts, ref_codes=[1], ref_text="ref")
        assert len(results) == 2
        mock_ph_batch.assert_called_once()

def test_fast_infer_stability_mode_updates_generation_config(mock_fast_tts):
    with patch("vieneu_utils.phonemize_text.phonemize_with_dict", return_value="phonemes"), \
         patch.object(mock_fast_tts, '_decode', return_value=np.zeros(1000)):
        _ = mock_fast_tts.infer("Xin chÃ o", ref_codes=[1, 2], ref_text="ref", stability_mode="locked_safe")
        assert mock_fast_tts.gen_config.temperature == pytest.approx(0.28)
        assert mock_fast_tts.gen_config.top_k == 8
        assert mock_fast_tts.gen_config.top_p == pytest.approx(0.70)
        assert mock_fast_tts.gen_config.repetition_penalty == pytest.approx(1.06)
        assert mock_fast_tts.gen_config.do_sample is True

def test_fast_infer_segments_groups_three_voice_routes(mock_fast_tts):
    narrator_voice = {"codes": [1], "text": "narrator"}
    female_voice = {"codes": [2], "text": "female"}
    male_voice = {"codes": [3], "text": "male"}
    segments = [
        {"segment_type": "narration", "text": "Má»™t ngÃ y mÆ°a."},
        {"segment_type": "narration", "text": "Khung cáº£nh yÃªn áº¯ng."},
        {"segment_type": "dialogue", "text": "Em Ä‘áº¿n Ä‘Ã¢y.", "performed_voice_persona": {"gender": "female"}},
        {"segment_type": "dialogue", "text": "Anh biáº¿t rá»“i.", "performed_voice_persona": {"gender": "male"}},
    ]

    with patch.object(
        mock_fast_tts,
        "infer",
        side_effect=[np.zeros(90), np.zeros(60), np.zeros(120)],
    ) as mock_infer:
        result = mock_fast_tts.infer_segments(
            segments,
            narrator_voice=narrator_voice,
            female_voice=female_voice,
            male_voice=male_voice,
            apply_watermark=False,
            return_metadata=True,
        )

    assert len(result["wavs"]) == 4
    assert [group["route"] for group in result["groups"]] == ["narration", "dialogue_female", "dialogue_male"]
    assert result["groups"][0]["segment_indexes"] == [0, 1]
    assert result["groups"][1]["segment_indexes"] == [2]
    assert result["groups"][2]["segment_indexes"] == [3]
    assert mock_infer.call_count == 3
    assert mock_infer.call_args_list[0].kwargs["voice"] == narrator_voice
    assert mock_infer.call_args_list[0].kwargs["emotion_tag"] is None
    assert mock_infer.call_args_list[0].kwargs["stability_mode"] == "locked_safe"
    assert mock_infer.call_args_list[1].kwargs["voice"] == female_voice
    assert mock_infer.call_args_list[1].kwargs["emotion_tag"] == "<|emotion_0|>"
    assert mock_infer.call_args_list[1].kwargs["stability_mode"] == "stable"
    assert mock_infer.call_args_list[2].kwargs["voice"] == male_voice
    assert mock_infer.call_args_list[2].kwargs["emotion_tag"] == "<|emotion_0|>"
    assert mock_infer.call_args_list[2].kwargs["stability_mode"] == "stable"

def test_fast_voice_aliases_use_sample_presets(mock_fast_tts):
    mock_fast_tts._preset_voices = {
        "Doan": {"codes": [1], "text": "doan"},
        "Ly": {"codes": [2], "text": "ly"},
        "Vinh": {"codes": [3], "text": "vinh"},
    }
    assert mock_fast_tts._normalize_preset_voice_name("doan") == "Doan"
    assert mock_fast_tts._normalize_preset_voice_name("sample_female") == "Ly"
    assert mock_fast_tts._normalize_preset_voice_name("sample_male") == "Vinh"
