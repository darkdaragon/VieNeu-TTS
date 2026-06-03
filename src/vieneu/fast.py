from pathlib import Path
from typing import Optional, Union, List, Generator, Any, Dict, Tuple
import numpy as np
import torch
import gc
import logging
import re
from collections import defaultdict
from .base import BaseVieneuTTS
from .utils import _compile_codec_with_triton, extract_speech_ids, _linear_overlap_add, normalize_device
from vieneu_utils.phonemize_text import phonemize_batch
from vieneu_utils.core_utils import split_text_into_chunks, join_audio_chunks

logger = logging.getLogger("Vieneu.Fast")

_FAST_STABILITY_PRESETS: Dict[str, Dict[str, Any]] = {
    "locked_safe": {"temperature": 0.28, "top_k": 8, "top_p": 0.70, "do_sample": True, "repetition_penalty": 1.06},
    "locked": {"temperature": 0.30, "top_k": 12, "top_p": 0.72, "do_sample": True, "repetition_penalty": 1.08},
    "stable": {"temperature": 0.55, "top_k": 20, "top_p": 0.80, "do_sample": True, "repetition_penalty": 1.10},
    "balanced": {"temperature": 0.70, "top_k": 35, "top_p": 0.88, "do_sample": True, "repetition_penalty": 1.15},
    "creative": {"temperature": 1.00, "top_k": 50, "top_p": 0.95, "do_sample": True, "repetition_penalty": 1.20},
}

_FAST_SAMPLE_ROUTE_DEFAULTS = {
    "narrator": "Doan",
    "dialogue_female": "Ly",
    "dialogue_male": "Vinh",
}

_FAST_VOICE_ALIASES = {
    "binh": "Binh",
    "tuyen": "Tuyen",
    "vinh": "Vinh",
    "doan": "Doan",
    "ly": "Ly",
    "ngoc": "Ngoc",
    "sample_narrator": "Doan",
    "sample_female": "Ly",
    "sample_male": "Vinh",
    "narrator": "Doan",
    "female": "Ly",
    "male": "Vinh",
}

class FastVieNeuTTS(BaseVieneuTTS):
    """
    GPU-optimized VieNeu-TTS using LMDeploy TurbomindEngine.
    """

    def __init__(
        self,
        backbone_repo: str = "pnnbao-ump/VieNeu-TTS",
        backbone_device: str = "cuda",
        codec_repo: str = "neuphonic/distill-neucodec",
        codec_device: str = "cuda",
        memory_util: float = 0.3,
        tp: int = 1,
        enable_prefix_caching: bool = False,
        quant_policy: int = 0,
        enable_triton: bool = True,
        max_batch_size: int = 4,
        hf_token: Optional[str] = None,
    ):
        super().__init__()
        self.device = backbone_device

        if backbone_device != "cuda" and not backbone_device.startswith("cuda:"):
            raise ValueError("LMDeploy backend requires CUDA device")

        # Streaming configuration
        self.streaming_overlap_frames = 1
        self.streaming_frames_per_chunk = 50
        self.streaming_lookforward = 5
        self.streaming_lookback = 50
        self.streaming_stride_samples = self.streaming_frames_per_chunk * self.hop_length

        self.max_batch_size = max_batch_size
        self._ref_cache: Dict[str, Any] = {}
        self.stored_dict = defaultdict(dict)

        self._is_onnx_codec = False
        self._triton_enabled = False

        self.use_chat_format = backbone_repo.rstrip("/").endswith("pnnbao-ump/VieNeu-TTS")

        self._load_backbone_lmdeploy(backbone_repo, memory_util, tp, enable_prefix_caching, quant_policy, hf_token)
        self._load_codec(codec_repo, codec_device, enable_triton)
        self._load_voices(backbone_repo, hf_token)
        self._warmup_model()

        logger.info("✅ FastVieNeuTTS with optimizations loaded successfully!")
        logger.info(f"   Max batch size: {self.max_batch_size}")

    def _resolve_generation_settings(self, temperature: float, top_k: int, **kwargs: Any) -> Dict[str, Any]:
        settings = {
            "temperature": temperature,
            "top_k": top_k,
        }
        stability_mode = str(kwargs.get("stability_mode") or "").strip().lower()
        preset = dict(_FAST_STABILITY_PRESETS.get(stability_mode, {}))

        if preset:
            if temperature == 1.0 and "temperature" in preset:
                settings["temperature"] = float(preset["temperature"])
            if top_k == 50 and "top_k" in preset:
                settings["top_k"] = int(preset["top_k"])
            for key in ("top_p", "do_sample", "repetition_penalty"):
                if key in preset:
                    settings[key] = preset[key]

        for key in ("top_p", "do_sample", "repetition_penalty", "min_new_tokens", "max_new_tokens"):
            if key in kwargs and kwargs[key] is not None:
                settings[key] = kwargs[key]
        return settings

    def _apply_generation_settings(self, settings: Dict[str, Any]) -> None:
        self.gen_config.temperature = float(settings.get("temperature", self.gen_config.temperature))
        self.gen_config.top_k = int(settings.get("top_k", self.gen_config.top_k))
        for key in ("top_p", "do_sample", "repetition_penalty", "min_new_tokens", "max_new_tokens"):
            if key in settings and hasattr(self.gen_config, key):
                setattr(self.gen_config, key, settings[key])

    def _normalize_preset_voice_name(self, voice_name: str) -> str:
        raw = str(voice_name or "").strip()
        if not raw:
            return raw
        if raw in self._preset_voices:
            return raw
        alias = _FAST_VOICE_ALIASES.get(raw.lower())
        if alias and alias in self._preset_voices:
            return alias
        for preset_name in self._preset_voices:
            if preset_name.lower() == raw.lower():
                return preset_name
        return raw

    def _resolve_voice_like(self, voice_like: Optional[Union[str, Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
        if voice_like is None:
            return None
        if isinstance(voice_like, dict):
            return voice_like
        if isinstance(voice_like, str):
            return self.get_preset_voice(self._normalize_preset_voice_name(voice_like))
        raise TypeError("Voice must be None, a preset voice name, or a voice dict.")

    def _segment_gender(self, segment: Dict[str, Any]) -> str:
        for key in ("performed_voice_persona", "speaker_source"):
            info = segment.get(key) or {}
            gender = str(info.get("gender") or "").strip().lower()
            if gender in {"male", "female"}:
                return gender
        return "female"

    def _segment_route(
        self,
        segment: Dict[str, Any],
        narrator_voice: Dict[str, Any],
        female_voice: Dict[str, Any],
        male_voice: Dict[str, Any],
        narrator_emotion_tag: Optional[str],
        dialogue_emotion_tag: Optional[str],
    ) -> Tuple[str, Dict[str, Any], Optional[str]]:
        seg_type = str(segment.get("segment_type") or "").strip().lower()
        if seg_type == "dialogue":
            if self._segment_gender(segment) == "male":
                return "dialogue_male", male_voice, dialogue_emotion_tag
            return "dialogue_female", female_voice, dialogue_emotion_tag
        return "narration", narrator_voice, narrator_emotion_tag

    def _estimate_segment_weight(self, segment: Dict[str, Any]) -> float:
        text = re.sub(r"\s+", " ", str(segment.get("text", "") or "")).strip()
        if not text:
            return 0.1
        pause_after_ms = int(segment.get("pause_after_ms", 0) or 0)
        chars = max(1, len(text))
        words = max(1, len(re.findall(r"\w+", text, flags=re.UNICODE)))
        pause_bonus = min(max(pause_after_ms, 0), 500) / 1000.0 * 0.35
        punctuation_bonus = 0.25 if text.endswith(("!", "?", ".")) else 0.12 if text.endswith((",", ";", ":")) else 0.0
        return max(0.2, (chars / 11.5) + (words * 0.015) + pause_bonus + punctuation_bonus)

    def _split_group_wav(self, wav: np.ndarray, segments: List[Dict[str, Any]]) -> List[np.ndarray]:
        if len(segments) <= 1:
            return [wav]
        total_samples = int(len(wav))
        if total_samples <= 0:
            return [np.array([], dtype=np.float32) for _ in segments]

        weights = [self._estimate_segment_weight(segment) for segment in segments]
        weight_sum = float(sum(weights)) or float(len(segments))

        boundaries = [0]
        consumed = 0.0
        for idx, weight in enumerate(weights[:-1], start=1):
            consumed += weight
            boundary = int(round((consumed / weight_sum) * total_samples))
            min_boundary = boundaries[-1] + 1
            max_boundary = total_samples - (len(segments) - idx)
            boundary = max(min_boundary, min(boundary, max_boundary))
            boundaries.append(boundary)
        boundaries.append(total_samples)

        chunks: List[np.ndarray] = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            chunks.append(wav[start:end].copy())
        return chunks

    def infer_segments(
        self,
        segments: List[Dict[str, Any]],
        narrator_voice: Optional[Union[str, Dict[str, Any]]] = None,
        female_voice: Optional[Union[str, Dict[str, Any]]] = None,
        male_voice: Optional[Union[str, Dict[str, Any]]] = None,
        narrator_emotion_tag: Optional[str] = None,
        dialogue_emotion_tag: Optional[str] = "<|emotion_0|>",
        narrator_stability_mode: str = "locked_safe",
        dialogue_stability_mode: str = "stable",
        group_max_chars: int = 1200,
        max_chars: int = 256,
        silence_p: float = 0.15,
        crossfade_p: float = 0.0,
        skip_normalize: bool = False,
        apply_watermark: bool = True,
        return_metadata: bool = False,
        **kwargs: Any,
    ) -> Union[List[np.ndarray], Dict[str, Any]]:
        """
        Render stage7-style segments with 3 long-form voice routes:
        narrator/storytelling, female dialogue/natural, male dialogue/natural.

        Adjacent segments that resolve to the same route are grouped into one long
        text render first, then the waveform is split back into per-segment clips.
        This mirrors the more stable long-text path and helps reduce voice drift
        across short dialogue-heavy segment lists.
        """
        if not segments:
            return {"wavs": [], "groups": []} if return_metadata else []

        default_narrator_name = self._normalize_preset_voice_name(_FAST_SAMPLE_ROUTE_DEFAULTS["narrator"])
        default_female_name = self._normalize_preset_voice_name(_FAST_SAMPLE_ROUTE_DEFAULTS["dialogue_female"])
        default_male_name = self._normalize_preset_voice_name(_FAST_SAMPLE_ROUTE_DEFAULTS["dialogue_male"])

        narrator_voice_obj = self._resolve_voice_like(narrator_voice) or self.get_preset_voice(default_narrator_name if default_narrator_name in self._preset_voices else None)
        female_voice_obj = self._resolve_voice_like(female_voice) or self.get_preset_voice(default_female_name if default_female_name in self._preset_voices else default_narrator_name if default_narrator_name in self._preset_voices else None)
        male_voice_obj = self._resolve_voice_like(male_voice) or self.get_preset_voice(default_male_name if default_male_name in self._preset_voices else default_narrator_name if default_narrator_name in self._preset_voices else None)

        base_kwargs = dict(kwargs)

        ordered_wavs: List[np.ndarray] = [np.array([], dtype=np.float32) for _ in segments]
        groups: List[Dict[str, Any]] = []
        current_group: List[Dict[str, Any]] = []
        current_key: Optional[Tuple[str, Optional[str], str]] = None
        current_chars = 0

        def flush_group() -> None:
            nonlocal current_group, current_key, current_chars
            if not current_group:
                return
            groups.append({
                "route": current_key[0] if current_key else "narration",
                "emotion_tag": current_key[1] if current_key else narrator_emotion_tag,
                "stability_mode": current_key[2] if current_key else narrator_stability_mode,
                "segments": current_group,
            })
            current_group = []
            current_key = None
            current_chars = 0

        for idx, segment in enumerate(segments):
            text = re.sub(r"\s+", " ", str(segment.get("text", "") or "")).strip()
            if not text:
                continue
            route, voice_obj, emotion_tag = self._segment_route(
                segment,
                narrator_voice_obj,
                female_voice_obj,
                male_voice_obj,
                narrator_emotion_tag,
                dialogue_emotion_tag,
            )
            stability_mode = narrator_stability_mode if route == "narration" else dialogue_stability_mode
            group_key = (route, emotion_tag, stability_mode)
            text_len = len(text)
            if current_group and (group_key != current_key or (group_max_chars > 0 and current_chars + text_len > group_max_chars)):
                flush_group()
            current_group.append({
                "index": idx,
                "segment": segment,
                "text": text,
                "voice": voice_obj,
            })
            current_key = group_key
            current_chars += text_len
        flush_group()

        group_meta: List[Dict[str, Any]] = []
        for group_idx, group in enumerate(groups, start=1):
            group_items = group["segments"]
            group_text = "\n\n".join(item["text"] for item in group_items).strip()
            if not group_text:
                continue
            group_kwargs = dict(base_kwargs)
            group_kwargs["stability_mode"] = group["stability_mode"]
            group_wav = self.infer(
                group_text,
                voice=group_items[0]["voice"],
                emotion_tag=group["emotion_tag"],
                max_chars=max_chars,
                silence_p=silence_p,
                crossfade_p=crossfade_p,
                skip_normalize=skip_normalize,
                apply_watermark=False,
                **group_kwargs,
            )
            split_wavs = self._split_group_wav(group_wav, [item["segment"] for item in group_items])
            for item, split_wav in zip(group_items, split_wavs):
                final_wav = self._apply_watermark(split_wav) if apply_watermark else split_wav
                ordered_wavs[item["index"]] = final_wav
            group_meta.append({
                "group_index": group_idx,
                "route": group["route"],
                "emotion_tag": group["emotion_tag"],
                "stability_mode": group["stability_mode"],
                "segment_indexes": [item["index"] for item in group_items],
                "segment_count": len(group_items),
                "text_length": len(group_text),
                "audio_samples": int(len(group_wav)),
            })

        if return_metadata:
            return {
                "wavs": ordered_wavs,
                "groups": group_meta,
            }
        return ordered_wavs

    def _load_backbone_lmdeploy(self, repo, memory_util, tp, enable_prefix_caching, quant_policy, hf_token=None):
        logger.info(f"Loading backbone with LMDeploy from: {repo}")
        if hf_token:
            import os
            os.environ["HF_TOKEN"] = hf_token

        try:
            from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
        except ImportError as e:
            raise ImportError(
                "Failed to import `lmdeploy`. Install with: pip install vieneu[gpu]"
            ) from e

        backend_config = TurbomindEngineConfig(
            cache_max_entry_count=memory_util,
            tp=tp,
            enable_prefix_caching=enable_prefix_caching,
            dtype='bfloat16',
            quant_policy=quant_policy
        )
        self.backbone = pipeline(repo, backend_config=backend_config)
        self.gen_config = GenerationConfig(
            top_p=0.95, top_k=50, temperature=1.0, max_new_tokens=2048,
            repetition_penalty=1.2,
            do_sample=True, min_new_tokens=40,
        )

    def _load_codec(self, codec_repo: str, codec_device: str, enable_triton: bool) -> None:
        super()._load_codec(codec_repo, codec_device)

        if enable_triton and not getattr(self, "_is_onnx_codec", False) and codec_device != "cpu":
            self._triton_enabled = _compile_codec_with_triton(self.codec)

    def _warmup_model(self):
        logger.info("🔥 Warming up model...")
        try:
            dummy_codes = list(range(10))
            dummy_prompt = self._format_prompt(dummy_codes, "warmup", "test", use_chat_format=self.use_chat_format)
            _ = self.backbone([dummy_prompt], gen_config=self.gen_config, do_preprocess=False)
            logger.info("   ✅ Warmup complete")
        except Exception as e:
            logger.warning(f"   ⚠️ Warmup failed: {e}")

    def _decode(self, codes_str: str) -> np.ndarray:
        speech_ids = extract_speech_ids(codes_str)
        if not speech_ids:
            raise ValueError(
                "No valid speech tokens found in the output. "
                "Lỗi này có thể do GPU của bạn không hỗ trợ định dạng bfloat16 (ví dụ: dòng T4, RTX 20-series) "
                "dẫn đến sai số khi tính toán. Bạn hãy thử chuyển sang dùng phiên bản VieNeu-TTS-0.3B nếu vẫn muốn dùng LmDeploy hoặc "
                "bỏ chọn 'LMDeploy' trong Tùy chọn nâng cao. Nếu vẫn gặp lỗi này, hãy thông báo với chúng tôi tại: https://discord.com/invite/yJt8kzjzWZ"
            )

        if self._is_onnx_codec:
            codes = np.array(speech_ids, dtype=np.int32)[np.newaxis, np.newaxis, :]
            recon = self.codec.decode_code(codes)
        else:
            with torch.no_grad():
                codes = torch.tensor(speech_ids, dtype=torch.long)[None, None, :].to(self.codec.device)
                recon = self.codec.decode_code(codes).cpu().numpy()
        return recon[0, 0, :]


    def infer(self, text: str, ref_audio: Optional[Union[str, Path]] = None, ref_codes: Optional[Union[np.ndarray, torch.Tensor]] = None, ref_text: Optional[str] = None, max_chars: int = 256, silence_p: float = 0.15, crossfade_p: float = 0.0, voice: Optional[Dict[str, Any]] = None, temperature: float = 1.0, top_k: int = 50, skip_normalize: bool = False, apply_watermark: bool = True, **kwargs) -> np.ndarray:

        ref_codes, ref_text = self._resolve_ref_voice(voice, ref_audio, ref_codes, ref_text)

        if not skip_normalize:
            text = self.normalizer.normalize(text)

        generation_settings = self._resolve_generation_settings(temperature, top_k, **kwargs)
        self._apply_generation_settings(generation_settings)

        chunks = split_text_into_chunks(text, max_chars=max_chars)
        if not chunks:
            return np.array([], dtype=np.float32)

        if len(chunks) == 1:
            prompt = self._format_prompt(ref_codes, ref_text, chunks[0], 
                                        use_chat_format=self.use_chat_format,
                                        emotion_tag=kwargs.get('emotion_tag'))
            responses = self.backbone([prompt], gen_config=self.gen_config, do_preprocess=False)
            wav = self._decode(responses[0].text)
            if apply_watermark:
                wav = self._apply_watermark(wav)
        else:
            all_wavs = self.infer_batch(
                chunks,
                ref_codes=ref_codes,
                ref_text=ref_text,
                voice=voice,
                temperature=temperature,
                top_k=top_k,
                skip_normalize=True,
                apply_watermark=False,
                **kwargs,
            )
            wav = join_audio_chunks(all_wavs, self.sample_rate, silence_p, crossfade_p)
            if apply_watermark:
                wav = self._apply_watermark(wav)

        return wav

    def infer_batch(self, texts: List[str], ref_audio: Optional[Union[str, Path]] = None, ref_codes: Optional[Union[np.ndarray, torch.Tensor]] = None, ref_text: Optional[str] = None, voice: Optional[Dict[str, Any]] = None, temperature: float = 1.0, top_k: int = 50, skip_normalize: bool = False, apply_watermark: bool = True, max_batch_size: Optional[int] = None, **kwargs) -> List[np.ndarray]:

        if not skip_normalize:
            texts = [self.normalizer.normalize(t) for t in texts]

        max_batch_size = max_batch_size or self.max_batch_size

        ref_codes, ref_text = self._resolve_ref_voice(voice, ref_audio, ref_codes, ref_text)

        # Pre-phonemize all for performance
        ref_phonemes = self.get_ref_phonemes(ref_text)
        chunk_phonemes = phonemize_batch(texts, skip_normalize=True)

        generation_settings = self._resolve_generation_settings(temperature, top_k, **kwargs)
        self._apply_generation_settings(generation_settings)

        all_wavs = []
        for i in range(0, len(texts), max_batch_size):
            batch_texts = texts[i : i + max_batch_size]
            batch_phonemes = chunk_phonemes[i : i + max_batch_size]
            prompts = [self._format_prompt(ref_codes, ref_text, text, ref_phonemes=ref_phonemes, 
                                          input_phonemes=ph, use_chat_format=self.use_chat_format,
                                          emotion_tag=kwargs.get('emotion_tag'))
                      for text, ph in zip(batch_texts, batch_phonemes)]
            responses = self.backbone(prompts, gen_config=self.gen_config, do_preprocess=False)
            batch_codes = [response.text for response in responses]
            batch_wavs = [self._decode(codes) for codes in batch_codes]
            if apply_watermark:
                batch_wavs = [self._apply_watermark(w) for w in batch_wavs]
            all_wavs.extend(batch_wavs)
        return all_wavs

    def infer_stream(self, text: str, ref_audio: Optional[Union[str, Path]] = None, ref_codes: Optional[Union[np.ndarray, torch.Tensor]] = None, ref_text: Optional[str] = None, max_chars: int = 256, voice: Optional[Dict[str, Any]] = None, temperature: float = 1.0, top_k: int = 50, skip_normalize: bool = False, **kwargs) -> Generator[np.ndarray, None, None]:

        ref_codes, ref_text = self._resolve_ref_voice(voice, ref_audio, ref_codes, ref_text)

        if not skip_normalize:
            text = self.normalizer.normalize(text)

        generation_settings = self._resolve_generation_settings(temperature, top_k, **kwargs)
        self._apply_generation_settings(generation_settings)

        chunks = split_text_into_chunks(text, max_chars=max_chars)
        for chunk in chunks:
            yield from self._infer_stream_single(chunk, ref_codes, ref_text, emotion_tag=kwargs.get('emotion_tag'))

    def _infer_stream_single(self, text: str, ref_codes: Any, ref_text: str, emotion_tag: Optional[str] = None) -> Generator[np.ndarray, None, None]:
        ref_codes_list = self.to_list(ref_codes)
        prompt = self._format_prompt(ref_codes_list, ref_text, text, use_chat_format=self.use_chat_format, emotion_tag=emotion_tag)
        audio_cache = []
        token_cache = [f"<|speech_{idx}|>" for idx in ref_codes_list]
        n_decoded_samples = 0
        n_decoded_tokens = len(ref_codes_list)

        for response in self.backbone.stream_infer([prompt], gen_config=self.gen_config, do_preprocess=False):
            output_str = response.text
            new_tokens = output_str[len("".join(token_cache[len(ref_codes_list):])):] if len(token_cache) > len(ref_codes_list) else output_str
            if new_tokens:
                token_cache.append(new_tokens)

            if len(token_cache[n_decoded_tokens:]) >= self.streaming_frames_per_chunk + self.streaming_lookforward:
                tokens_start = max(n_decoded_tokens - self.streaming_lookback - self.streaming_overlap_frames, 0)
                tokens_end = n_decoded_tokens + self.streaming_frames_per_chunk + self.streaming_lookforward + self.streaming_overlap_frames
                sample_start = (n_decoded_tokens - tokens_start) * self.hop_length
                sample_end = sample_start + (self.streaming_frames_per_chunk + 2 * self.streaming_overlap_frames) * self.hop_length
                curr_codes = token_cache[tokens_start:tokens_end]
                recon = self._decode("".join(curr_codes))
                recon = self._apply_watermark(recon)
                recon = recon[sample_start:sample_end]
                audio_cache.append(recon)

                processed_recon = _linear_overlap_add(audio_cache, stride=self.streaming_stride_samples)
                new_samples_end = len(audio_cache) * self.streaming_stride_samples
                processed_recon = processed_recon[n_decoded_samples:new_samples_end]
                n_decoded_samples = new_samples_end
                n_decoded_tokens += self.streaming_frames_per_chunk
                yield processed_recon

        remaining_tokens = len(token_cache) - n_decoded_tokens
        if remaining_tokens > 0:
            tokens_start = max(len(token_cache) - (self.streaming_lookback + self.streaming_overlap_frames + remaining_tokens), 0)
            sample_start = (len(token_cache) - tokens_start - remaining_tokens - self.streaming_overlap_frames) * self.hop_length
            curr_codes = token_cache[tokens_start:]
            recon = self._decode("".join(curr_codes))
            recon = self._apply_watermark(recon)
            recon = recon[sample_start:]
            audio_cache.append(recon)
            processed_recon = _linear_overlap_add(audio_cache, stride=self.streaming_stride_samples)
            processed_recon = processed_recon[n_decoded_samples:]
            yield processed_recon

    def cleanup_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def get_optimization_stats(self) -> Dict[str, Any]:
        return {
            'triton_enabled': self._triton_enabled,
            'max_batch_size': self.max_batch_size,
            'cached_references': len(self._ref_cache),
            'active_sessions': len(self.stored_dict),
            'prefix_caching': False,
        }
