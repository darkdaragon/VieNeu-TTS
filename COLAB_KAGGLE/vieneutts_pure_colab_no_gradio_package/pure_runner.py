
import asyncio
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace


def _load_app(app_dir: str):
    app_path = Path(app_dir) / "app.py"
    if not app_path.exists():
        hits = list(Path(app_dir).rglob("app.py"))
        if not hits:
            raise FileNotFoundError(f"Cannot find app.py under {app_dir}")
        app_path = hits[0]
    spec = importlib.util.spec_from_file_location("vieneu_pure_app", app_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, app_path.parent


def _copy_file(src, dst_dir: Path):
    if not src:
        return None
    src = Path(src)
    if not src.exists():
        return None
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    shutil.copyfile(src, dst)
    return str(dst)


def _progress_factory(label, file_index=1, file_total=1, file_state=None):
    """Progress renderer for Colab/terminal.

    Shows one compact progress bar with:
      - current file index / total files
      - completed files ok/failed
      - current manifest phase
      - segment done/total, ok, failed
      - current segment/detail

    In Colab subprocess output, carriage-return only progress can disappear.
    Default is one carriage-return progress bar. Set VIENEU_PROGRESS_NEWLINE=1 only when your UI cannot display "\r" updates.
    """
    import re
    import os

    file_state = file_state if file_state is not None else {"ok": 0, "failed": 0}
    newline_mode = os.getenv("VIENEU_PROGRESS_NEWLINE", "0").strip().lower() in {"1", "true", "yes", "on"}
    throttle_sec = float(os.getenv("VIENEU_PROGRESS_THROTTLE_SEC", "0.5") or 0.5)

    last = {
        "ts": 0.0,
        "line": "",
        "phase": "",
        "printed": False,
        "last_done": -1,
    }

    def _to_int(value, default=0):
        try:
            if value in (None, ""):
                return default
            return int(value)
        except Exception:
            return default

    def _bar(done, total, width=24):
        if total <= 0:
            return "░" * width, 0.0
        pct = max(0.0, min(1.0, done / total))
        filled = int(round(pct * width))
        return "█" * filled + "░" * (width - filled), pct * 100.0

    def _current_label(detail):
        detail = str(detail or "").strip()
        m = re.search(r"(seg_\d{3,})", detail)
        if m:
            return m.group(1)
        return detail[:100]

    def cb(**updates):
        if updates.pop("_finish", False):
            if last["printed"] and not newline_mode:
                print("", flush=True)
                last["printed"] = False
            return

        now = time.time()
        phase = str(updates.get("phase", "") or "")
        status = str(updates.get("status", "") or "")
        done = _to_int(updates.get("done", 0))
        total = _to_int(updates.get("total", 0))
        ok = _to_int(updates.get("ok", 0))
        failed = _to_int(updates.get("failed", 0))
        detail = _current_label(updates.get("detail", ""))

        bar, pct = _bar(done, total)
        total_text = str(total) if total > 0 else "?"
        current = f" | current={detail}" if detail else ""
        # One-line progress bar only. Engine load/drop logs still print as normal lines.
        # Example:
        # [████████░░░░░░░░░░░░░░░░] 33.3% | file 1/3 | files ok=0 fail=0 | seg 120/360 ok=120 fail=0 | current=seg_000120
        line = (
            f"[{bar}] {pct:5.1f}% | file {file_index}/{file_total} "
            f"| files ok={file_state.get('ok', 0)} fail={file_state.get('failed', 0)} "
            f"| {label} | {status} {phase} "
            f"| seg {done}/{total_text} ok={ok} fail={failed}{current}"
        )

        must_print = (
            line != last["line"]
            and (
                now - last["ts"] >= throttle_sec
                or phase != last["phase"]
                or status in {"COMPLETE", "INCOMPLETE", "FAILED"}
                or done == total
                or last["last_done"] < 0
            )
        )
        if not must_print:
            return

        if newline_mode:
            print(line, flush=True)
            last["printed"] = False
        else:
            padded = line + " " * max(0, len(last["line"]) - len(line))
            end = "\n" if status in {"COMPLETE", "INCOMPLETE", "FAILED"} else ""
            print("\r" + padded, end=end, flush=True)
            last["printed"] = end == ""

        last["line"] = line
        last["phase"] = phase
        last["ts"] = now
        last["last_done"] = done

    return cb

def _select_voice_for_plain(app, quality_mode: str):
    # Keep plain mode simple. Manifest mode ignores this and uses segment voice mapping.
    if quality_mode in {"fast", "fast_0_3b"}:
        return "doan"
    return "doan"


# ==============================================================================
# v25: plain TXT -> single-voice manifest + final audio speed control
# ==============================================================================

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？…])\s+")


def _cfg_float(cfg, key, default=1.0):
    try:
        value = cfg.get(key)
        if value in (None, ""):
            value = os.getenv(key, default)
        return float(value)
    except Exception:
        return float(default)


def _cfg_int(cfg, key, default=0):
    try:
        value = cfg.get(key)
        if value in (None, ""):
            value = os.getenv(key, default)
        return int(value)
    except Exception:
        return int(default)


def _cfg_str(cfg, key, default=""):
    value = cfg.get(key)
    if value in (None, ""):
        value = os.getenv(key, default)
    return str(value if value is not None else default)


def split_plain_text_for_manifest(text, max_chars=420, min_chars=180):
    """Split raw .txt into large but safe TTS segments.

    It keeps a single voice/emotion route and only splits for stability.
    max_chars is a soft max; the splitter prefers sentence/paragraph boundaries.
    """
    text = str(text or "").replace("\ufeff", "")
    text = re.sub(r"\r\n?", "\n", text)
    paragraphs = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    units = []
    for para in paragraphs or [re.sub(r"\s+", " ", text).strip()]:
        if not para:
            continue
        parts = _SENTENCE_SPLIT_RE.split(para)
        for part in parts:
            part = part.strip()
            if part:
                units.append(part)

    chunks = []
    buf = ""

    def flush():
        nonlocal buf
        if buf.strip():
            chunks.append(buf.strip())
        buf = ""

    for unit in units:
        if len(unit) > max_chars:
            flush()
            words = unit.split()
            sub = ""
            for w in words:
                cand = (sub + " " + w).strip()
                if sub and len(cand) > max_chars:
                    chunks.append(sub.strip())
                    sub = w
                else:
                    sub = cand
            if sub:
                chunks.append(sub.strip())
            continue

        cand = (buf + " " + unit).strip()
        if buf and len(cand) > max_chars and len(buf) >= min_chars:
            flush()
            buf = unit
        else:
            buf = cand
    flush()
    return [c for c in chunks if c.strip()]


def build_single_voice_manifest_from_txt(app, file_path: Path, cfg: dict):
    raw_text = Path(file_path).read_text(encoding="utf-8", errors="ignore")
    max_chars = _cfg_int(cfg, "plain_text_max_chars", _cfg_int(cfg, "PLAIN_TEXT_MAX_CHARS", 420))
    min_chars = _cfg_int(cfg, "plain_text_min_chars", _cfg_int(cfg, "PLAIN_TEXT_MIN_CHARS", 180))
    voice = _cfg_str(cfg, "plain_text_voice", _cfg_str(cfg, "PLAIN_TEXT_VOICE", "doan"))
    emotion = _cfg_str(cfg, "plain_text_emotion", _cfg_str(cfg, "PLAIN_TEXT_EMOTION", "storytelling"))
    pause_after_ms = _cfg_int(cfg, "plain_text_pause_after_ms", _cfg_int(cfg, "PLAIN_TEXT_PAUSE_AFTER_MS", 180))
    rate_pct = _cfg_int(cfg, "plain_text_rate_pct", _cfg_int(cfg, "PLAIN_TEXT_RATE_PCT", 0))
    pitch_hz = _cfg_int(cfg, "plain_text_pitch_hz", _cfg_int(cfg, "PLAIN_TEXT_PITCH_HZ", 0))

    chunks = split_plain_text_for_manifest(raw_text, max_chars=max_chars, min_chars=min_chars)
    if not chunks:
        raise ValueError(f"TXT input is empty after cleanup: {file_path}")

    segments = []
    for idx, chunk in enumerate(chunks, start=1):
        segments.append({
            "order": idx,
            "line_index": idx,
            "segment_id": f"seg_{idx:06d}",
            "segment_type": "narration",
            "text": chunk,
            "voice": voice,
            "voice_mode": "natural",
            "rate_pct": rate_pct,
            "pitch_hz": pitch_hz,
            "pause_after_ms": pause_after_ms,
            "speaker_source": {"gender": "unknown", "age_tone": "unknown", "role_rank": "unknown"},
            "performed_voice_persona": {"gender": "unknown", "age_tone": "unknown", "role_rank": "unknown"},
            "detection": {"source": "plain_txt_single_voice"},
            "emotion": emotion,
        })

    manifest = {
        "schema_version": "audio_segments.v2",
        "source_file": Path(file_path).name,
        "source_mode": "plain_txt_single_voice",
        "defaults": {
            "default_voice": voice,
            "narrator_default_voice": voice,
            "dialogue_default_voice": voice,
            "default_rate_pct": rate_pct,
            "default_pitch_hz": pitch_hz,
            "default_pause_after_ms": pause_after_ms,
            "default_emotion": emotion,
        },
        "watermark_plan": app.default_watermark_plan() if hasattr(app, "default_watermark_plan") else {},
        "segments": segments,
    }
    manifest = app.normalize_manifest(manifest, source_name=Path(file_path).name)
    return manifest



# ==============================================================================
# v29: long-text render modes
# ==============================================================================
# RENDER_INPUT_MODE:
#   "segments"            -> keep original segment-by-segment behavior
#   "segments_ref_chunks" -> JSON segments input; infer adjacent non-dialogue together, then split back to original segments for SRT
#   "segments_single_voice_ref_chunks" -> JSON segments input; force one voice, infer adjacent/all segments together, then split back to original segments for SRT
#   "segments_native_infer" -> JSON segments input; send original segments into FastVieNeuTTS.infer_segments() for native long-term routing
#   "tag_long_text"       -> JSON: dialogue stays tagged, non-dialogue consecutive runs become long-text blocks
#   "txt_long_text"       -> TXT: one single-voice long text segment; VieNeu SDK handles splitting/joining internally
#
# Aliases:
#   "tag", "tag_author", "longtext_tag" -> tag_long_text
#   "txt", "plain_txt", "plain_txt_long_text" -> txt_long_text
#   "segment", "original" -> segments
#   "segments_ref", "segment_ref_chunks", "segments_ref_chunks" -> segments_ref_chunks
#   "segments_single_voice", "segments_one_voice" -> segments_single_voice_ref_chunks
#   "segments_native", "native_infer_segments", "segments_native_infer" -> segments_native_infer

def get_render_input_mode(cfg, suffix=None):
    raw = (
        cfg.get("render_input_mode")
        or cfg.get("RENDER_INPUT_MODE")
        or os.getenv("RENDER_INPUT_MODE")
        or ""
    )
    raw = str(raw or "").strip().lower()
    aliases = {
        "": "segments",
        "segment": "segments",
        "segments": "segments",
        "original": "segments",
        "keep_segments": "segments",
        "segments_ref": "segments_ref_chunks",
        "segment_ref": "segments_ref_chunks",
        "segment_ref_chunks": "segments_ref_chunks",
        "segments_ref_chunks": "segments_ref_chunks",
        "json_segments_ref_chunks": "segments_ref_chunks",
        "segments_single_voice": "segments_single_voice_ref_chunks",
        "segment_single_voice": "segments_single_voice_ref_chunks",
        "segments_one_voice": "segments_single_voice_ref_chunks",
        "one_voice_segments": "segments_single_voice_ref_chunks",
        "single_voice_segments": "segments_single_voice_ref_chunks",
        "segments_single_voice_ref_chunks": "segments_single_voice_ref_chunks",
        "segments_native": "segments_native_infer",
        "segment_native": "segments_native_infer",
        "native_infer_segments": "segments_native_infer",
        "segments_native_infer": "segments_native_infer",
        "tag": "tag_long_text",
        "tag_author": "tag_long_text",
        "tag_longtext": "tag_long_text",
        "tag_long_text": "tag_long_text",
        "longtext_tag": "tag_long_text",
        "txt": "txt_long_text",
        "plain_txt": "txt_long_text",
        "plain_txt_long_text": "txt_long_text",
        "txt_longtext": "txt_long_text",
        "txt_long_text": "txt_long_text",
    }
    mode = aliases.get(raw, raw)
    if suffix == ".txt" and mode in {"segments", "tag_long_text"}:
        # TXT has no dialogue tags, so it is always a single voice mode.
        return "txt_long_text" if _cfg_str(cfg, "plain_text_use_author_split", os.getenv("PLAIN_TEXT_USE_AUTHOR_SPLIT", "1")).lower() not in {"0", "false", "no"} else "segments"
    return mode if mode in {"segments", "segments_ref_chunks", "segments_single_voice_ref_chunks", "segments_native_infer", "tag_long_text", "txt_long_text"} else "segments"


def _segment_gender_for_tag_mode(seg):
    seg = seg or {}
    candidates = [
        seg.get("gender"),
        seg.get("speaker_gender"),
        seg.get("voice_gender"),
    ]
    for field in ("speaker_source", "performed_voice_persona", "detection"):
        obj = seg.get(field)
        if isinstance(obj, dict):
            candidates.append(obj.get("gender"))
    for value in candidates:
        raw = str(value or "").strip().lower()
        if raw in {"male", "nam", "man", "m", "boy", "adult_male", "elder_male"}:
            return "male"
        if raw in {"female", "nữ", "nu", "woman", "f", "girl", "adult_female", "elder_female"}:
            return "female"
    # explicit voice can also indicate male/female
    voice = str(seg.get("voice") or "").strip().lower()
    if voice in {"vinh", "vĩnh", "male", "namminh", "nam minh", "nam_minh"}:
        return "male"
    if voice in {"doan", "đoan", "female", "hoaimy", "hoài my", "hoai_my"}:
        return "female"
    return "unknown"


def _make_manifest_segment(idx, text, segment_type="narration", voice="doan", emotion="storytelling", pause_after_ms=180, source_ids=None, gender="unknown"):
    return {
        "order": idx,
        "line_index": idx,
        "segment_id": f"seg_{idx:06d}",
        "segment_type": segment_type,
        "text": str(text or "").strip(),
        "voice": voice,
        "voice_mode": "natural",
        "rate_pct": 0,
        "pitch_hz": 0,
        "pause_after_ms": int(pause_after_ms or 0),
        "speaker_source": {"gender": gender, "age_tone": "unknown", "role_rank": "unknown"},
        "performed_voice_persona": {"gender": gender, "age_tone": "unknown", "role_rank": "unknown"},
        "detection": {"source": "v29_long_text_mode", "source_segment_ids": source_ids or []},
        "emotion": emotion,
    }


def build_single_voice_longtext_manifest_from_txt(app, file_path: Path, cfg: dict):
    raw_text = Path(file_path).read_text(encoding="utf-8", errors="ignore")
    raw_text = str(raw_text or "").replace("\ufeff", "")
    raw_text = re.sub(r"\r\n?", "\n", raw_text)
    raw_text = re.sub(r"[ \t]+", " ", raw_text)
    raw_text = re.sub(r"\n{3,}", "\n\n", raw_text).strip()
    if not raw_text:
        raise ValueError(f"TXT input is empty after cleanup: {file_path}")

    voice = _cfg_str(cfg, "plain_text_voice", _cfg_str(cfg, "PLAIN_TEXT_VOICE", "doan"))
    emotion = _cfg_str(cfg, "plain_text_emotion", _cfg_str(cfg, "PLAIN_TEXT_EMOTION", "storytelling"))
    pause_after_ms = _cfg_int(cfg, "plain_text_pause_after_ms", _cfg_int(cfg, "PLAIN_TEXT_PAUSE_AFTER_MS", 180))

    manifest = {
        "schema_version": "audio_segments.v2",
        "source_file": Path(file_path).name,
        "source_mode": "plain_txt_long_text_sdk_split",
        "defaults": {
            "default_voice": voice,
            "narrator_default_voice": voice,
            "dialogue_default_voice": voice,
            "default_rate_pct": 0,
            "default_pitch_hz": 0,
            "default_pause_after_ms": pause_after_ms,
            "default_emotion": emotion,
        },
        "watermark_plan": app.default_watermark_plan() if hasattr(app, "default_watermark_plan") else {},
        "segments": [
            _make_manifest_segment(
                1,
                raw_text,
                segment_type="narration",
                voice=voice,
                emotion=emotion,
                pause_after_ms=pause_after_ms,
                source_ids=["plain_txt_all"],
                gender="unknown",
            )
        ],
    }
    return app.normalize_manifest(manifest, source_name=Path(file_path).name)




def _copy_manifest_preserve_segments(manifest):
    data = dict(manifest or {})
    data["segments"] = [dict(seg) for seg in ((manifest or {}).get("segments") or [])]
    data["audio_strategy"] = dict((manifest or {}).get("audio_strategy") or {})
    return data


def prepare_segments_ref_chunks_manifest_from_json_manifest(app, manifest: dict, cfg: dict):
    """Keep original segment list/SRT, but enable shared-ref runtime infer.

    This does NOT merge manifest segments. It only marks the manifest so app.py will
    infer adjacent non-dialogue segments together, then cut the real audio back into
    original segment clips for final concat/SRT.
    """
    new_manifest = _copy_manifest_preserve_segments(manifest)
    non_dialogue_voice = _cfg_str(cfg, "segments_ref_chunks_non_dialogue_voice", _cfg_str(cfg, "SEGMENTS_REF_CHUNKS_NON_DIALOGUE_VOICE", "doan"))
    non_dialogue_emotion = _cfg_str(cfg, "segments_ref_chunks_non_dialogue_emotion", _cfg_str(cfg, "SEGMENTS_REF_CHUNKS_NON_DIALOGUE_EMOTION", "storytelling"))
    dialogue_emotion = _cfg_str(cfg, "segments_ref_chunks_dialogue_emotion", _cfg_str(cfg, "SEGMENTS_REF_CHUNKS_DIALOGUE_EMOTION", "natural"))
    pause_after_ms = _cfg_int(cfg, "segments_ref_chunks_non_dialogue_pause_after_ms", _cfg_int(cfg, "SEGMENTS_REF_CHUNKS_NON_DIALOGUE_PAUSE_AFTER_MS", 180))

    for seg in new_manifest.get("segments") or []:
        seg_type = str(seg.get("segment_type") or "").strip().lower()
        if seg_type != "dialogue":
            seg["voice"] = non_dialogue_voice
            seg["emotion"] = non_dialogue_emotion
            seg["pause_after_ms"] = int(seg.get("pause_after_ms", pause_after_ms) or pause_after_ms)
        else:
            seg["emotion"] = seg.get("emotion") or dialogue_emotion

    audio_strategy = new_manifest.setdefault("audio_strategy", {})
    audio_strategy["segment_infer_strategy"] = "adjacent_non_dialogue_shared_ref"
    audio_strategy["default_voice"] = non_dialogue_voice
    audio_strategy["narrator_default_voice"] = non_dialogue_voice
    audio_strategy["default_emotion"] = non_dialogue_emotion
    audio_strategy["default_pause_after_ms"] = pause_after_ms
    new_manifest["source_mode"] = "json_segments_ref_chunks_runtime_split"
    return app.normalize_manifest(new_manifest, source_name=str(new_manifest.get("source_file") or "segments_ref_chunks"))


def prepare_segments_single_voice_ref_chunks_manifest_from_json_manifest(app, manifest: dict, cfg: dict):
    """Keep original segment list/SRT, force one voice, infer together, split back.

    This is the segments version of TXT single voice: every JSON segment uses the
    same voice/emotion reference, but SRT remains per original segment.
    """
    new_manifest = _copy_manifest_preserve_segments(manifest)
    voice = _cfg_str(cfg, "segments_single_voice", _cfg_str(cfg, "SEGMENTS_SINGLE_VOICE", _cfg_str(cfg, "plain_text_voice", "doan")))
    emotion = _cfg_str(cfg, "segments_single_voice_emotion", _cfg_str(cfg, "SEGMENTS_SINGLE_VOICE_EMOTION", _cfg_str(cfg, "plain_text_emotion", "storytelling")))
    pause_after_ms = _cfg_int(cfg, "segments_single_voice_pause_after_ms", _cfg_int(cfg, "SEGMENTS_SINGLE_VOICE_PAUSE_AFTER_MS", _cfg_int(cfg, "plain_text_pause_after_ms", 180)))
    for seg in new_manifest.get("segments") or []:
        seg["voice"] = voice
        seg["emotion"] = emotion
        seg["pause_after_ms"] = int(seg.get("pause_after_ms", pause_after_ms) or pause_after_ms)
    audio_strategy = new_manifest.setdefault("audio_strategy", {})
    audio_strategy["segment_infer_strategy"] = "all_segments_single_voice_shared_ref"
    audio_strategy["default_voice"] = voice
    audio_strategy["narrator_default_voice"] = voice
    audio_strategy["dialogue_default_voice"] = voice
    audio_strategy["default_emotion"] = emotion
    audio_strategy["default_pause_after_ms"] = pause_after_ms
    new_manifest["source_mode"] = "json_segments_single_voice_ref_chunks_runtime_split"
    return app.normalize_manifest(new_manifest, source_name=str(new_manifest.get("source_file") or "segments_single_voice"))


def prepare_segments_native_infer_manifest_from_json_manifest(app, manifest: dict, cfg: dict):
    """Keep original segment list/SRT and let FastVieNeuTTS.infer_segments() route voices natively."""
    new_manifest = _copy_manifest_preserve_segments(manifest)
    non_dialogue_voice = _cfg_str(cfg, "tag_non_dialogue_voice", os.getenv("VIENEU_NON_DIALOGUE_VOICE", "doan"))
    female_voice = _cfg_str(cfg, "tag_dialogue_female_voice", os.getenv("VIENEU_DIALOGUE_FEMALE_VOICE", "doan"))
    male_voice = _cfg_str(cfg, "tag_dialogue_male_voice", os.getenv("VIENEU_DIALOGUE_MALE_VOICE", "vinh"))
    narration_emotion = _cfg_str(cfg, "tag_non_dialogue_emotion", os.getenv("VIENEU_NARRATION_EMOTION", "storytelling"))
    dialogue_emotion = _cfg_str(cfg, "tag_dialogue_emotion", os.getenv("VIENEU_DIALOGUE_EMOTION", "natural"))

    audio_strategy = new_manifest.setdefault("audio_strategy", {})
    audio_strategy["segment_infer_strategy"] = "vieneu_native_infer_segments"
    audio_strategy["non_dialogue_voice"] = non_dialogue_voice
    audio_strategy["dialogue_female_voice"] = female_voice
    audio_strategy["dialogue_male_voice"] = male_voice
    audio_strategy["narrator_default_voice"] = non_dialogue_voice
    audio_strategy["dialogue_default_voice"] = female_voice
    audio_strategy["default_voice"] = non_dialogue_voice
    audio_strategy["default_emotion"] = narration_emotion
    new_manifest["source_mode"] = "json_segments_native_infer"

    for seg in new_manifest.get("segments") or []:
        seg_type = str(seg.get("segment_type") or "").strip().lower()
        if seg_type == "dialogue":
            seg["emotion"] = seg.get("emotion") or dialogue_emotion
        else:
            seg["emotion"] = seg.get("emotion") or narration_emotion

    return app.normalize_manifest(new_manifest, source_name=str(new_manifest.get("source_file") or "segments_native_infer"))


def build_tag_longtext_manifest_from_json_manifest(app, manifest: dict, cfg: dict):
    """Build a tag-aware long-text manifest from JSON segments.

    The goal is to reduce prosody drift by letting VieNeu SDK split/join long
    narration/non-dialogue blocks internally, while keeping male/female dialogue tags.
    Consecutive non-dialogue segments are merged. Dialogue segments are kept separate
    by default so male/female routing remains controllable.
    """
    original_segments = list(manifest.get("segments") or [])
    if not original_segments:
        return manifest

    non_dialogue_voice = _cfg_str(cfg, "tag_non_dialogue_voice", os.getenv("VIENEU_NON_DIALOGUE_VOICE", "doan"))
    female_voice = _cfg_str(cfg, "tag_dialogue_female_voice", os.getenv("VIENEU_DIALOGUE_FEMALE_VOICE", "doan"))
    male_voice = _cfg_str(cfg, "tag_dialogue_male_voice", os.getenv("VIENEU_DIALOGUE_MALE_VOICE", "vinh"))

    narration_emotion = _cfg_str(cfg, "tag_non_dialogue_emotion", os.getenv("VIENEU_NARRATION_EMOTION", "storytelling"))
    dialogue_emotion = _cfg_str(cfg, "tag_dialogue_emotion", os.getenv("VIENEU_DIALOGUE_EMOTION", "natural"))

    non_dialogue_pause = _cfg_int(cfg, "tag_non_dialogue_pause_after_ms", _cfg_int(cfg, "PLAIN_TEXT_PAUSE_AFTER_MS", 180))

    keep_dialogue_separate = _cfg_str(cfg, "tag_keep_dialogue_separate", os.getenv("TAG_KEEP_DIALOGUE_SEPARATE", "1")).lower() not in {"0", "false", "no"}
    # If false, consecutive dialogue with same resolved gender/voice are grouped too.
    new_segments = []
    idx = 1
    buf_texts = []
    buf_ids = []
    buf_pause = non_dialogue_pause

    def flush_non_dialogue():
        nonlocal idx, buf_texts, buf_ids, buf_pause
        if not buf_texts:
            return
        text = "\n".join(t.strip() for t in buf_texts if str(t).strip()).strip()
        if text:
            new_segments.append(_make_manifest_segment(
                idx,
                text,
                segment_type="narration",
                voice=non_dialogue_voice,
                emotion=narration_emotion,
                pause_after_ms=buf_pause,
                source_ids=list(buf_ids),
                gender="female",
            ))
            idx += 1
        buf_texts = []
        buf_ids = []
        buf_pause = non_dialogue_pause

    last_dialogue_key = None
    dlg_texts = []
    dlg_ids = []
    dlg_pause = 180

    def flush_dialogue():
        nonlocal idx, last_dialogue_key, dlg_texts, dlg_ids, dlg_pause
        if not dlg_texts or not last_dialogue_key:
            dlg_texts = []; dlg_ids = []; last_dialogue_key = None
            return
        voice, gender = last_dialogue_key
        text = "\n".join(t.strip() for t in dlg_texts if str(t).strip()).strip()
        if text:
            new_segments.append(_make_manifest_segment(
                idx,
                text,
                segment_type="dialogue",
                voice=voice,
                emotion=dialogue_emotion,
                pause_after_ms=dlg_pause,
                source_ids=list(dlg_ids),
                gender=gender,
            ))
            idx += 1
        dlg_texts = []
        dlg_ids = []
        last_dialogue_key = None
        dlg_pause = 180

    for seg in sorted(original_segments, key=lambda s: int(s.get("order", 0) or 0)):
        text = str(seg.get("text") or "").strip()
        if not text:
            continue
        seg_type = str(seg.get("segment_type") or "").strip().lower()
        seg_id = str(seg.get("segment_id") or f"orig_{seg.get('order', '')}")
        if seg_type != "dialogue":
            flush_dialogue()
            buf_texts.append(text)
            buf_ids.append(seg_id)
            try:
                buf_pause = int(seg.get("pause_after_ms", buf_pause))
            except Exception:
                pass
            continue

        flush_non_dialogue()
        gender = _segment_gender_for_tag_mode(seg)
        voice = male_voice if gender == "male" else female_voice
        gender_out = "male" if gender == "male" else "female"

        if keep_dialogue_separate:
            flush_dialogue()
            new_segments.append(_make_manifest_segment(
                idx,
                text,
                segment_type="dialogue",
                voice=voice,
                emotion=dialogue_emotion,
                pause_after_ms=int(seg.get("pause_after_ms", 180) or 180),
                source_ids=[seg_id],
                gender=gender_out,
            ))
            idx += 1
        else:
            key = (voice, gender_out)
            if last_dialogue_key != key:
                flush_dialogue()
                last_dialogue_key = key
            dlg_texts.append(text)
            dlg_ids.append(seg_id)
            dlg_pause = int(seg.get("pause_after_ms", dlg_pause) or dlg_pause)

    flush_non_dialogue()
    flush_dialogue()

    new_manifest = dict(manifest)
    new_manifest["source_mode"] = "json_tag_long_text_sdk_split"
    new_manifest["source_file"] = str(manifest.get("source_file") or "tag_long_text")
    new_manifest["segments"] = new_segments
    new_manifest["defaults"] = {
        "default_voice": non_dialogue_voice,
        "narrator_default_voice": non_dialogue_voice,
        "dialogue_default_voice": female_voice,
        "default_rate_pct": 0,
        "default_pitch_hz": 0,
        "default_pause_after_ms": non_dialogue_pause,
        "default_emotion": narration_emotion,
    }
    return app.normalize_manifest(new_manifest, source_name=new_manifest["source_file"])


def _ffmpeg_atempo_filter(speed):
    speed = float(speed)
    if speed <= 0:
        return "atempo=1.0"
    factors = []
    remaining = speed
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    factors.append(remaining)
    return ",".join(f"atempo={x:.6g}" for x in factors)


def apply_audio_speed_to_file(audio_path, speed):
    speed = float(speed or 1.0)
    if abs(speed - 1.0) < 0.001:
        return audio_path
    audio_path = Path(audio_path)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print(f"⚠️ AUDIO_SPEED={speed} requested but ffmpeg not found; keeping original audio.", flush=True)
        return str(audio_path)
    tmp = audio_path.with_name(audio_path.stem + f"_speed_{speed:g}" + audio_path.suffix)
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(audio_path), "-filter:a", _ffmpeg_atempo_filter(speed), str(tmp)]
    subprocess.run(cmd, check=True)
    shutil.move(str(tmp), str(audio_path))
    return str(audio_path)


def _scale_timestamp_srt(ts, speed):
    h, m, rest = ts.split(":")
    s, ms = rest.split(",")
    total = (int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0) / float(speed)
    total = max(0.0, total)
    hh = int(total // 3600); total -= hh * 3600
    mm = int(total // 60); total -= mm * 60
    ss = int(total); msec = int(round((total - ss) * 1000))
    if msec >= 1000:
        ss += 1; msec -= 1000
    return f"{hh:02d}:{mm:02d}:{ss:02d},{msec:03d}"


def _scale_timestamp_vtt(ts, speed):
    h, m, rest = ts.split(":")
    s, ms = rest.split(".")
    total = (int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0) / float(speed)
    total = max(0.0, total)
    hh = int(total // 3600); total -= hh * 3600
    mm = int(total // 60); total -= mm * 60
    ss = int(total); msec = int(round((total - ss) * 1000))
    if msec >= 1000:
        ss += 1; msec -= 1000
    return f"{hh:02d}:{mm:02d}:{ss:02d}.{msec:03d}"


def apply_subtitle_speed_to_file(subtitle_path, speed):
    speed = float(speed or 1.0)
    if abs(speed - 1.0) < 0.001 or not subtitle_path:
        return subtitle_path
    p = Path(subtitle_path)
    if not p.exists():
        return subtitle_path
    text = p.read_text(encoding="utf-8", errors="ignore")
    if p.suffix.lower() == ".srt":
        pattern = re.compile(r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})")
        text = pattern.sub(lambda m: f"{_scale_timestamp_srt(m.group(1), speed)} --> {_scale_timestamp_srt(m.group(2), speed)}", text)
    elif p.suffix.lower() == ".vtt":
        pattern = re.compile(r"(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})")
        text = pattern.sub(lambda m: f"{_scale_timestamp_vtt(m.group(1), speed)} --> {_scale_timestamp_vtt(m.group(2), speed)}", text)
    p.write_text(text, encoding="utf-8")
    return str(p)


def maybe_apply_output_speed(audio_path, subtitle_path, cfg):
    speed = _cfg_float(cfg, "audio_speed", _cfg_float(cfg, "AUDIO_SPEED", 1.0))
    if abs(speed - 1.0) < 0.001:
        return audio_path, subtitle_path
    print(f"🎚️ Applying final AUDIO_SPEED={speed:g} to audio/subtitle output", flush=True)
    audio_path = apply_audio_speed_to_file(audio_path, speed) if audio_path else audio_path
    subtitle_path = apply_subtitle_speed_to_file(subtitle_path, speed) if subtitle_path else subtitle_path
    return audio_path, subtitle_path


async def render_one(app, file_path: Path, cfg: dict, output_run_dir: Path, file_index=1, file_total=1, file_state=None):
    suffix = file_path.suffix.lower()
    source_name = file_path.name
    work_dir = Path(tempfile.mkdtemp(prefix=f"pure_{file_path.stem}_"))
    final_dir = output_run_dir / file_path.stem
    debug_dir = output_run_dir / "_debug_reports"
    label = file_path.name
    print(f"\n===== START file {file_index}/{file_total}: {label} =====", flush=True)
    print(f"Render order={cfg.get('render_order_strategy')} | workers={cfg.get('worker_count')} | timeline={cfg.get('timeline_mode')} | subtitle={cfg.get('subtitle_format')}", flush=True)
    print("Extra pause: ENABLED — pause_after_ms is read from each JSON segment and added during final concat/final_parts.", flush=True)
    print("Timeline accurate: SRT/VTT uses real rendered audio duration when timeline_mode=accurate.", flush=True)
    try:
        if suffix in {".json", ".txt"}:
            render_input_mode = get_render_input_mode(cfg, suffix=suffix)
            if suffix == ".json":
                manifest = app.load_manifest(str(file_path))
                if render_input_mode == "segments_ref_chunks":
                    manifest = prepare_segments_ref_chunks_manifest_from_json_manifest(app, manifest, cfg)
                    source_mode = "json_segments_ref_chunks_runtime_split"
                    print(
                        f"SEGMENTS ref-chunks mode: original_segments={len(manifest.get('segments') or [])} "
                        f"| adjacent non-dialogue shares one infer/ref | audio is split back per segment for SRT",
                        flush=True,
                    )
                elif render_input_mode == "segments_native_infer":
                    manifest = prepare_segments_native_infer_manifest_from_json_manifest(app, manifest, cfg)
                    source_mode = "json_segments_native_infer"
                    print(
                        f"SEGMENTS native infer mode: original_segments={len(manifest.get('segments') or [])} "
                        f"| FastVieNeuTTS.infer_segments routes narration/doan + dialogue female/doan + dialogue male/vinh "
                        f"| per-segment SRT is preserved",
                        flush=True,
                    )
                elif render_input_mode == "segments_single_voice_ref_chunks":
                    manifest = prepare_segments_single_voice_ref_chunks_manifest_from_json_manifest(app, manifest, cfg)
                    source_mode = "json_segments_single_voice_ref_chunks_runtime_split"
                    print(
                        f"SEGMENTS single-voice ref-chunks mode: original_segments={len(manifest.get('segments') or [])} "
                        f"| one voice={cfg.get('segments_single_voice') or cfg.get('plain_text_voice') or 'doan'} "
                        f"| shared infer then split back per segment for SRT",
                        flush=True,
                    )
                elif render_input_mode == "tag_long_text":
                    manifest = build_tag_longtext_manifest_from_json_manifest(app, manifest, cfg)
                    source_mode = "json_tag_long_text_sdk_split"
                    print(
                        f"TAG long-text mode: render_units={len(manifest.get('segments') or [])} "
                        f"| non_dialogue=Doan/storytelling | dialogue male/female tagged | SDK handles split/join inside each long block",
                        flush=True,
                    )
                else:
                    source_mode = "json_manifest_segments"
            else:
                if render_input_mode == "txt_long_text":
                    manifest = build_single_voice_longtext_manifest_from_txt(app, file_path, cfg)
                    source_mode = "plain_txt_long_text_sdk_split"
                    print(
                        f"TXT long-text single-voice mode: render_units={len(manifest.get('segments') or [])} "
                        f"| voice={cfg.get('plain_text_voice') or os.getenv('PLAIN_TEXT_VOICE', 'doan')} "
                        f"| emotion={cfg.get('plain_text_emotion') or os.getenv('PLAIN_TEXT_EMOTION', 'storytelling')} "
                        f"| SDK handles split/join internally",
                        flush=True,
                    )
                else:
                    manifest = build_single_voice_manifest_from_txt(app, file_path, cfg)
                    source_mode = "plain_txt_single_voice_chunked"
                    print(
                        f"TXT chunked single-voice mode: segments={len(manifest.get('segments') or [])} "
                        f"| voice={cfg.get('plain_text_voice') or os.getenv('PLAIN_TEXT_VOICE', 'doan')} "
                        f"| emotion={cfg.get('plain_text_emotion') or os.getenv('PLAIN_TEXT_EMOTION', 'storytelling')} "
                        f"| max_chars={cfg.get('plain_text_max_chars') or os.getenv('PLAIN_TEXT_MAX_CHARS', '420')}",
                        flush=True,
                    )
            progress_cb = _progress_factory(label, file_index=file_index, file_total=file_total, file_state=file_state)
            result = await app.render_manifest_to_outputs(
                manifest=manifest,
                subtitle_format=cfg["subtitle_format"],
                upload=False,
                source_name=source_name,
                work_dir=str(work_dir),
                cache_name=source_name,
                tts_timeout_sec=int(cfg["tts_timeout_sec"]),
                force_rerender=bool(cfg["force_rerender"]),
                progress_callback=progress_cb,
                rescue_repeated_short_failures=bool(cfg["rescue_repeated_short_failures"]),
                rescue_short_segments_now=bool(cfg.get("rescue_short_segments_now", False)),
                repo_cache_only=False,
                segment_worker_count=int(cfg["worker_count"]),
                timeline_mode=cfg["timeline_mode"],
                tts_semaphore=app.AdaptiveTTSLimiter(int(cfg["worker_count"]), step=None),
                cache_only=bool(cfg.get("cache_only", False)),
                retry_failed_only=bool(cfg.get("retry_failed_only", False)),
            )
            complete = bool(result.get("complete"))
            copied = []
            progress_cb(_finish=True)
            if complete:
                final_audio = result.get("final_audio")
                final_subtitle = result.get("local_subtitle")
                final_audio, final_subtitle = maybe_apply_output_speed(final_audio, final_subtitle, cfg)
                audio = _copy_file(final_audio, final_dir)
                if audio:
                    copied.append(audio)
                sub = _copy_file(final_subtitle, final_dir)
                if sub:
                    copied.append(sub)
                print(f"✅ COMPLETE {label} ({source_mode})")
                for path in copied:
                    print(f"  output: {path}")
            else:
                rep = _copy_file(result.get("local_report"), debug_dir)
                print(f"⚠️ INCOMPLETE {label}. Final audio/subtitle not built. Debug report: {rep}")
                copied = [rep] if rep else []
            return {"file": str(file_path), "complete": complete, "outputs": copied}

        print(f"SKIP unsupported file type: {file_path}")
        return {"file": str(file_path), "complete": False, "outputs": [], "skipped": True}
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


async def main(config_path: str):
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    os.environ.update(cfg.get("env", {}))

    # v29 locked_safe stability patch:
    # Keep generation sampling consistent for all 3 render options:
    #   segments / tag_long_text / txt_long_text
    # Config values are mirrored into env because app.py reads stability from env.
    mode_for_env = str(cfg.get("render_input_mode") or cfg.get("RENDER_INPUT_MODE") or os.getenv("RENDER_INPUT_MODE", "segments"))
    os.environ.setdefault("RENDER_INPUT_MODE", mode_for_env)
    os.environ.setdefault("VIENEU_RENDER_INPUT_MODE", mode_for_env)
    cfg_to_env = {
        "vieneu_voice_stability_mode": "VIENEU_VOICE_STABILITY_MODE",
        "vieneu_infer_temperature": "VIENEU_INFER_TEMPERATURE",
        "vieneu_infer_top_k": "VIENEU_INFER_TOP_K",
        "vieneu_infer_top_p": "VIENEU_INFER_TOP_P",
        "vieneu_infer_repetition_penalty": "VIENEU_INFER_REPETITION_PENALTY",
        "vieneu_infer_do_sample": "VIENEU_INFER_DO_SAMPLE",
    }
    for cfg_key, env_key in cfg_to_env.items():
        if cfg.get(cfg_key) not in (None, ""):
            os.environ.setdefault(env_key, str(cfg[cfg_key]))

    # Same temp/top-k/top-p are also exposed as per-mode knobs. If you want
    # different behavior later, set only the mode-specific env you need.
    if os.getenv("VIENEU_INFER_TEMPERATURE"):
        os.environ.setdefault("VIENEU_SEGMENT_TEMPERATURE", os.getenv("VIENEU_INFER_TEMPERATURE"))
        os.environ.setdefault("VIENEU_TAG_TEMPERATURE", os.getenv("VIENEU_INFER_TEMPERATURE"))
        os.environ.setdefault("VIENEU_LONG_TEXT_TEMPERATURE", os.getenv("VIENEU_INFER_TEMPERATURE"))
    if os.getenv("VIENEU_INFER_TOP_K"):
        os.environ.setdefault("VIENEU_SEGMENT_TOP_K", os.getenv("VIENEU_INFER_TOP_K"))
        os.environ.setdefault("VIENEU_TAG_TOP_K", os.getenv("VIENEU_INFER_TOP_K"))
        os.environ.setdefault("VIENEU_LONG_TEXT_TOP_K", os.getenv("VIENEU_INFER_TOP_K"))
    if os.getenv("VIENEU_INFER_TOP_P"):
        os.environ.setdefault("VIENEU_SEGMENT_TOP_P", os.getenv("VIENEU_INFER_TOP_P"))
        os.environ.setdefault("VIENEU_TAG_TOP_P", os.getenv("VIENEU_INFER_TOP_P"))
        os.environ.setdefault("VIENEU_LONG_TEXT_TOP_P", os.getenv("VIENEU_INFER_TOP_P"))

    app, app_dir = _load_app(cfg["app_dir"])

    # Put cache state on Drive so reruns can reuse it without Gradio/HF.
    cache_root = Path(cfg["output_root"]) / "chunkcache"
    pending_root = Path(cfg["output_root"]) / "pending_upload"
    silence_root = Path(cfg["output_root"]) / "silence_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    pending_root.mkdir(parents=True, exist_ok=True)
    silence_root.mkdir(parents=True, exist_ok=True)
    app.CHUNK_CACHE_ROOT = cache_root
    app.PENDING_UPLOAD_ROOT = pending_root
    app.SYNC_STATE_PATH = pending_root / "_sync_state.json"
    app.SILENCE_CACHE_ROOT = silence_root
    app.FINAL_PART_MIN_SEGMENTS = 1

    run_name = cfg.get("run_name") or time.strftime("run_%Y%m%d_%H%M%S")
    output_run_dir = Path(cfg["output_root"]) / "runs" / run_name
    output_run_dir.mkdir(parents=True, exist_ok=True)

    print("Pure Colab render config:")
    print(json.dumps({k: v for k, v in cfg.items() if k not in {"env"}}, ensure_ascii=False, indent=2))
    print("Engine env:")
    print(json.dumps(cfg.get("env", {}), ensure_ascii=False, indent=2))
    print("App dir:", app_dir)
    print("Output run dir:", output_run_dir)

    if hasattr(app, "preload_vieneu_engines"):
        preload_result = app.preload_vieneu_engines()
        print("Preload result:", json.dumps(preload_result, ensure_ascii=False, indent=2))

    results = []
    input_files = cfg.get("input_files") or []
    file_state = {"ok": 0, "failed": 0}
    file_total = len(input_files)
    for file_index, raw in enumerate(input_files, start=1):
        path = Path(raw)
        result = await render_one(app, path, cfg, output_run_dir, file_index=file_index, file_total=file_total, file_state=file_state)
        results.append(result)
        if result.get("complete"):
            file_state["ok"] += 1
        else:
            file_state["failed"] += 1
        print(f"📌 FILE SUMMARY: {file_index}/{file_total} done | files_ok={file_state['ok']} files_failed={file_state['failed']}", flush=True)

    complete = sum(1 for r in results if r.get("complete"))
    incomplete = len(results) - complete
    print("\n===== ALL DONE =====")
    print(f"complete={complete}, incomplete={incomplete}, files={len(results)}")
    if incomplete:
        debug_dir = output_run_dir / "_debug_reports"
        debug_dir.mkdir(parents=True, exist_ok=True)
        summary_path = debug_dir / "_summary.json"
        summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print("debug summary:", summary_path)
    else:
        print("all files complete; no success report/summary was written.")
    print("success outputs are under:", output_run_dir)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python pure_runner.py /path/to/config.json")
    asyncio.run(main(sys.argv[1]))
