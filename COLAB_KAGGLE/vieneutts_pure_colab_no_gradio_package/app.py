import asyncio
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import threading
from copy import deepcopy
from pathlib import Path

# VieNeuTTS engine replacement for the original Edge-TTS runtime.
# The old file imported edge_tts here; this version keeps an edge_tts-compatible
# placeholder only for old log lines, while all real synthesis goes through VieNeu.
class _EngineVersion:
    __version__ = "vieneu-adapter-v2"

edge_tts = _EngineVersion()

try:
    from vieneu import Vieneu
except ImportError:  # pragma: no cover - Colab first run may install later
    Vieneu = None

try:
    import gradio as gr
except ImportError:  # pragma: no cover - exercised in local test environments without UI deps
    gr = None

try:
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download, CommitOperationDelete
except ImportError:  # pragma: no cover - exercised in local test environments without upload deps
    HfApi = None
    hf_hub_download = None
    snapshot_download = None
    CommitOperationDelete = None


HF_TOKEN = os.getenv("HF_TOKEN")
# Optional. If empty or HF_TOKEN is missing, upload/cache sync falls back to Google Drive/local folder.
DATASET_REPO = os.getenv("DATASET_REPO", "").strip()
GOOGLE_DRIVE_OUTPUT_ROOT = Path(os.getenv("GOOGLE_DRIVE_OUTPUT_ROOT", "/content/drive/MyDrive/VieNeuTTS_Output"))
USE_GOOGLE_DRIVE_WHEN_NO_HF = os.getenv("USE_GOOGLE_DRIVE_WHEN_NO_HF", "1").strip().lower() not in {"0", "false", "no"}
WORKERS = 2
MANIFEST_WORKERS = 4
SEGMENT_RENDER_WORKERS = 4
SEGMENT_TTS_MAX_CHARS = 2000
SEGMENT_TTS_RETRIES = 3
SEGMENT_RETRY_BACKOFF_SEC = 1.5
SEGMENT_TTS_TIMEOUT_SEC = 60
# Subtitle timing policy for manifest mode.
# 0 = do not add a global artificial offset. Raise to 80-180 only if your video renderer still shows text early.
SUBTITLE_GLOBAL_DELAY_MS = 0
# If Edge TTS reports the last SentenceBoundary shorter than the real segment MP3, extend the last subtitle
# to the real speech/audio end so the subtitle timeline is anchored to measured audio, not Edge metadata only.
SUBTITLE_EXTEND_LAST_EVENT_TO_AUDIO_END = True
# ffmpeg silencedetect threshold used to estimate actual speech bounds inside each segment MP3.
SILENCE_DETECT_NOISE_DB = -45
SILENCE_DETECT_MIN_DURATION_SEC = 0.03
MANIFEST_MIN_SEGMENT_CHARS = 24
MANIFEST_MIN_SEGMENT_WORDS = 3
CHUNK_CACHE_ROOT = Path("chunkcache")
PENDING_UPLOAD_ROOT = Path("pending_hf_upload")
SYNC_STATE_PATH = PENDING_UPLOAD_ROOT / "_sync_state.json"


# Global fixed limiter: Worker count is the maximum total active Edge TTS calls.
# Auto downgrade/downrate is disabled. The configured worker count stays stable.

class AdaptiveTTSLimiter:
    """Backward-compatible fixed concurrency limiter.

    Kept under the old class name so the rest of the code can keep using
    the same interface, but note_success/note_failure are intentionally no-op.
    """
    def __init__(self, initial_limit, min_limit=1, step=None, threshold=None):
        self.target_limit = max(1, int(initial_limit or 1))
        self.min_limit = 1
        self.step = 0
        self.threshold = 0
        self.active = 0
        self.failure_score = 0
        self.downgrade_events = []
        self._cond = asyncio.Condition()

    async def __aenter__(self):
        async with self._cond:
            while self.active >= self.target_limit:
                await self._cond.wait()
            self.active += 1
        return self

    async def __aexit__(self, exc_type, exc, tb):
        async with self._cond:
            self.active = max(0, self.active - 1)
            self._cond.notify_all()
        return False

    def current_limit(self):
        return int(self.target_limit)

    def note_success(self):
        return None

    def note_failure(self, reason="vieneu_tts_failure"):
        return None

SILENCE_CACHE_ROOT = Path("silence_cache")
CACHE_INDEX_FILENAME = "_cache_index.json"
TIMELINE_FAST = "fast"
TIMELINE_ACCURATE = "accurate"
TIMELINE_MODES = {TIMELINE_FAST, TIMELINE_ACCURATE}
DEFAULT_TIMELINE_MODE = TIMELINE_ACCURATE
SHORT_SEGMENT_TIMEOUT_SEC = 45
STALE_PROGRESS_LOG_SEC = 180

SUCCESS_CACHE_DIRNAME = "success"
FAILED_CACHE_DIRNAME = "failed"
FINAL_PARTS_DIRNAME = "final_parts"

RAW_SUCCESS_UPLOAD_DISABLED = True


def should_upload_remote_cache_rel(rel_path):
    """Only ship compact cache payloads to the repo.

    Remote cache keeps only _cache_index.json, failed/, and final_parts/.
    Raw per-segment success files under success/ stay local for the current runtime,
    but are not pushed to the repo because they can exceed the repo limit on files
    per directory. Legacy flat raw success files are also skipped.
    """
    rel = str(rel_path or '').replace('\\', '/').strip('/')
    if not rel:
        return False
    if rel == CACHE_INDEX_FILENAME:
        return True
    if rel.startswith(FAILED_CACHE_DIRNAME + '/'):
        return True
    if rel.startswith(FINAL_PARTS_DIRNAME + '/'):
        return True
    return False


def iter_uploadable_cache_files(manifest_cache_dir):
    cache_dir = Path(manifest_cache_dir)
    if not cache_dir.exists():
        return
    for cache_file in cache_dir.rglob('*'):
        if not cache_file.is_file():
            continue
        rel_cache = cache_file.relative_to(cache_dir).as_posix()
        if should_upload_remote_cache_rel(rel_cache):
            yield cache_file, rel_cache


def build_filtered_manifest_cache_snapshot(manifest_cache_dir, dest_dir):
    cache_dir = Path(manifest_cache_dir)
    dest_dir = Path(dest_dir)
    copied = 0
    skipped = 0
    for cache_file in cache_dir.rglob('*'):
        if not cache_file.is_file():
            continue
        rel_cache = cache_file.relative_to(cache_dir).as_posix()
        if not should_upload_remote_cache_rel(rel_cache):
            skipped += 1
            continue
        target = dest_dir / rel_cache
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cache_file, target)
        copied += 1
    return {'copied': copied, 'skipped': skipped}


def clear_remote_raw_success_payload(cache_name):
    """Delete raw success payload from the repo before uploading compact cache.

    This removes chunkcache/<manifest>/success/** and legacy flat success files.
    failed/, final_parts/, and _cache_index.json are preserved.
    """
    if api is None:
        return {'deleted': 0, 'paths': []}
    ensure_dataset_exists()
    cache_name = build_manifest_cache_name(cache_name)
    remote_prefix = build_remote_cache_prefix(cache_name).strip('/')
    repo_files = api.list_repo_files(repo_id=DATASET_REPO, repo_type='dataset', token=HF_TOKEN)
    success_folder = remote_prefix + '/' + SUCCESS_CACHE_DIRNAME
    legacy_delete_paths = []
    success_folder_exists = False
    for path in repo_files:
        if not path.startswith(remote_prefix + '/'):
            continue
        rel = path[len(remote_prefix) + 1:]
        if rel.startswith(SUCCESS_CACHE_DIRNAME + '/'):
            success_folder_exists = True
            continue
        if '/' not in rel and (rel.endswith('.mp3') or (rel.endswith('.json') and not rel.endswith('.failed.json') and rel != CACHE_INDEX_FILENAME)):
            legacy_delete_paths.append(path)
    operations = []
    if CommitOperationDelete is not None and success_folder_exists:
        operations.append(CommitOperationDelete(path_in_repo=success_folder, is_folder=True))
    if CommitOperationDelete is not None:
        operations.extend(CommitOperationDelete(path_in_repo=p) for p in sorted(set(legacy_delete_paths)))
        if operations:
            api.create_commit(
                repo_id=DATASET_REPO,
                repo_type='dataset',
                operations=operations,
                commit_message=f'Prune raw success cache payload: {cache_name}',
                token=HF_TOKEN,
            )
        return {'deleted': (1 if success_folder_exists else 0) + len(set(legacy_delete_paths)), 'paths': ([success_folder] if success_folder_exists else []) + sorted(set(legacy_delete_paths))[:12]}

    # Fallback for old huggingface_hub: delete individual files only.
    delete_paths = []
    for path in repo_files:
        if path.startswith(success_folder + '/') or path in legacy_delete_paths:
            delete_paths.append(path)
    for path in sorted(set(delete_paths)):
        try:
            api.delete_file(path_in_repo=path, repo_id=DATASET_REPO, repo_type='dataset', commit_message=f'Prune raw success cache payload: {cache_name}', token=HF_TOKEN)
        except Exception:
            pass
    return {'deleted': len(set(delete_paths)), 'paths': sorted(set(delete_paths))[:12]}

FINAL_PART_MIN_SEGMENTS = 1


def has_local_final_parts_for_manifest(cache_name):
    cache_dir = build_manifest_cache_dir(cache_name)
    parts_dir = cache_dir / FINAL_PARTS_DIRNAME
    return parts_dir.exists() and any(parts_dir.glob('*.mp3')) and any(parts_dir.glob('*.json'))


def clear_remote_raw_success_payload_if_migrated(cache_name):
    """Only delete remote raw success after local final_parts exist.

    Old cache is migrated safely: first run with the manifest can download raw
    success, convert contiguous success ranges to final_parts/, commit those compact
    parts, and only then prune raw success from the repo.
    """
    if not has_local_final_parts_for_manifest(cache_name):
        return {'deleted': 0, 'paths': [], 'skipped': True, 'reason': 'local final_parts not ready; raw success kept'}
    result = clear_remote_raw_success_payload(cache_name)
    result['skipped'] = False
    return result

CACHE_SCOPE_ALL = "all"
CACHE_SCOPE_FAILED_ONLY = "failed_only"
CACHE_SCOPE_SUCCESS_ONLY = "success_only"

DEFAULT_PLAIN_MODE = "plain_txt_single_voice"
DEFAULT_MANIFEST_MODE = "audio_manifest_multi_voice"
PLAIN_FILE_TYPES = [".txt"]
MANIFEST_FILE_TYPES = [".json"]

SCHEMA_VERSION = "audio_segments.v2"
SEGMENT_TYPES = {"narration", "dialogue", "inner_monologue", "seo_tag"}
VOICE_MODES = {"natural", "imitated", "disguised", "projected", "quoted"}
GENDERS = {"male", "female", "neutral", "unknown"}
AGE_TONES = {"child", "teen", "young_adult", "adult", "elder", "unknown"}
ROLE_RANKS = {"low", "neutral", "high", "royal", "master", "system", "artifact", "unknown"}

VOICE_PROFILE_DEFAULTS = {
    "narrator_female_main": {"voice": "vi-VN-HoaiMyNeural", "rate_pct": 10, "pitch_hz": -8, "pause_after_ms": 220},
    "narrator_male_main": {"voice": "vi-VN-HoaiMyNeural", "rate_pct": 10, "pitch_hz": -8, "pause_after_ms": 220},
    "dialogue_female_child": {"voice": "vi-VN-HoaiMyNeural", "rate_pct": 10, "pitch_hz": -8, "pause_after_ms": 220},
    "dialogue_male_child": {"voice": "vi-VN-NamMinhNeural", "rate_pct": 10, "pitch_hz": -8, "pause_after_ms": 220},
    "dialogue_female_teen": {"voice": "vi-VN-HoaiMyNeural", "rate_pct": 10, "pitch_hz": -8, "pause_after_ms": 220},
    "dialogue_male_teen": {"voice": "vi-VN-NamMinhNeural", "rate_pct": 10, "pitch_hz": -8, "pause_after_ms": 220},
    "dialogue_female_young": {"voice": "vi-VN-HoaiMyNeural", "rate_pct": 10, "pitch_hz": -8, "pause_after_ms": 220},
    "dialogue_male_young": {"voice": "vi-VN-NamMinhNeural", "rate_pct": 10, "pitch_hz": -8, "pause_after_ms": 220},
    "dialogue_female_adult": {"voice": "vi-VN-HoaiMyNeural", "rate_pct": 10, "pitch_hz": -8, "pause_after_ms": 220},
    "dialogue_male_adult": {"voice": "vi-VN-NamMinhNeural", "rate_pct": 10, "pitch_hz": -8, "pause_after_ms": 220},
    "dialogue_female_elder": {"voice": "vi-VN-HoaiMyNeural", "rate_pct": 10, "pitch_hz": -8, "pause_after_ms": 220},
    "dialogue_male_elder": {"voice": "vi-VN-NamMinhNeural", "rate_pct": 10, "pitch_hz": -8, "pause_after_ms": 220},
    "inner_female_soft": {"voice": "vi-VN-HoaiMyNeural", "rate_pct": 10, "pitch_hz": -8, "pause_after_ms": 220},
    "inner_male_soft": {"voice": "vi-VN-HoaiMyNeural", "rate_pct": 10, "pitch_hz": -8, "pause_after_ms": 220},
    "system_mechanical": {"voice": "vi-VN-HoaiMyNeural", "rate_pct": 10, "pitch_hz": -8, "pause_after_ms": 220},
    "artifact_mystic": {"voice": "vi-VN-HoaiMyNeural", "rate_pct": 10, "pitch_hz": -8, "pause_after_ms": 220},
    "watermark_neutral": {"voice": "vi-VN-HoaiMyNeural", "rate_pct": 10, "pitch_hz": -8, "pause_after_ms": 220},
    "quoted_soft": {"voice": "vi-VN-HoaiMyNeural", "rate_pct": 10, "pitch_hz": -8, "pause_after_ms": 220},
}

VOICE_FALLBACKS = {
    "vi-VN-NamMinhNeural": "vi-VN-HoaiMyNeural",
}

DEFAULT_MANIFEST_STRATEGY = {
    "narrator_default_voice": "vi-VN-HoaiMyNeural",
    "dialogue_default_voice": "vi-VN-HoaiMyNeural",
    "watermark_default_voice": "vi-VN-HoaiMyNeural",
    "default_voice": "vi-VN-HoaiMyNeural",
    "default_rate_pct": 10,
    "default_pitch_hz": -8,
    "default_pause_after_ms": 220,
}

# New manifest-render policy:
# - Keep incoming JSON schema as-is.
# - Do not infer voice from speaker_source/performed_voice_persona/voice_profile_key.
# - TTS only trusts segment.voice, segment.rate_pct, segment.pitch_hz and segment.pause_after_ms.
# - Edge SentenceBoundary is used as a timing hint only; subtitle text always comes from segment["text"].
VOICE_ALIASES = {
    "hoaimy": "vi-VN-HoaiMyNeural",
    "hoai my": "vi-VN-HoaiMyNeural",
    "hoài my": "vi-VN-HoaiMyNeural",
    "vi-vn-hoaimyneural": "vi-VN-HoaiMyNeural",
    "vi-VN-HoaiMyNeural": "vi-VN-HoaiMyNeural",
    "namminh": "vi-VN-NamMinhNeural",
    "nam minh": "vi-VN-NamMinhNeural",
    "vi-vn-namminhneural": "vi-VN-NamMinhNeural",
    "vi-VN-NamMinhNeural": "vi-VN-NamMinhNeural",
}

SHORT_SEGMENT_MAX_WORDS = 3
SHORT_SEGMENT_MAX_CHARS = 16
RESCUE_CARRIER_GROUP_MAX_SEGMENTS = 20
RESCUE_CARRIER_GROUP_MAX_CHARS = 500
RESCUE_CUT_PAD_BEFORE_SEC = 0.04
RESCUE_CUT_PAD_AFTER_SEC = 0.08
RESCUE_BOUNDARY_SNAP_WINDOW_SEC = 0.22

PREVIEW_CSS = """
#preview_manifest_box {
    max-height: 440px;
    overflow: hidden;
}
#preview_manifest_box .cm-editor,
#preview_manifest_box textarea,
#preview_manifest_box pre {
    max-height: 440px !important;
    overflow: auto !important;
}
#batch_logs textarea {
    max-height: 360px !important;
    overflow: auto !important;
}
"""

api = HfApi(token=HF_TOKEN) if (HfApi is not None and HF_TOKEN and DATASET_REPO) else None
_DATASET_READY = False


def ensure_dataset_exists():
    global _DATASET_READY
    if _DATASET_READY:
        return
    if api is None:
        print("huggingface_hub is unavailable; dataset checks are skipped in this environment.")
        return
    try:
        api.repo_info(repo_id=DATASET_REPO, repo_type="dataset")
        print(f"Dataset {DATASET_REPO} is ready.")
        _DATASET_READY = True
    except Exception:
        try:
            api.create_repo(repo_id=DATASET_REPO, repo_type="dataset", private=False)
            print(f"Created dataset: {DATASET_REPO}")
            _DATASET_READY = True
        except Exception as exc:
            print(f"Unable to create dataset: {exc}")
CHUNK_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
PENDING_UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
SILENCE_CACHE_ROOT.mkdir(parents=True, exist_ok=True)


async def get_voices():
    if edge_tts is None:
        raise RuntimeError("edge_tts is not installed in this environment.")
    voices = await edge_tts.list_voices()
    return {
        f"{v['ShortName']} - {v['Locale']} ({v['Gender']})": v["ShortName"]
        for v in voices
    }


def sanitize_filename(name):
    stem = os.path.splitext(os.path.basename(name or ""))[0]
    stem = re.sub(r'[\\/*?:"<>|]', "", stem)
    return stem.strip() or "audio"


def extract_story_prefix(filename: str) -> str:
    stem = os.path.splitext(os.path.basename(filename or ""))[0]
    match = re.match(r"([A-Za-z0-9]+)", stem)
    if match:
        return match.group(1)
    fallback = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return fallback or "story"


def build_output_subdir(filename_or_source_file: str) -> str:
    return f"outputs/{extract_story_prefix(filename_or_source_file)}"


def upload_outputs_grouped(local_file, story_prefix, remote_name):
    if api is None:
        raise RuntimeError("huggingface_hub is not installed in this environment.")
    ensure_dataset_exists()
    api.upload_file(
        path_or_fileobj=local_file,
        path_in_repo=f"outputs/{story_prefix}/{remote_name}",
        repo_id=DATASET_REPO,
        repo_type="dataset",
    )


def build_remote_cache_prefix(cache_name):
    return f"chunkcache/{sanitize_filename(cache_name)}"


def build_manifest_remote_cache_prefix_from_filename(manifest_filename):
    return build_remote_cache_prefix(build_manifest_cache_name(os.path.basename(manifest_filename)))


def collect_remote_cache_prefixes_for_files(files, mode):
    if mode != DEFAULT_MANIFEST_MODE:
        return []
    prefixes = []
    for file in files or []:
        filename = os.path.basename(getattr(file, "name", "") or "")
        if not filename.lower().endswith(".json"):
            continue
        prefix = build_manifest_remote_cache_prefix_from_filename(filename).strip("/")
        if prefix and prefix not in prefixes:
            prefixes.append(prefix)
    return prefixes


def clear_remote_chunkcache_prefixes(prefixes, commit_message=None):
    """Delete committed chunkcache folders from the Hugging Face Dataset.

    This is used only when Force rerender is enabled. Local cache is already
    cleared separately; this removes the remote cache so a Space restart cannot
    resume from stale dataset cache. Multiple folders are deleted in one commit
    when the installed huggingface_hub version supports create_commit.
    """
    prefixes = [str(prefix or "").replace("\\", "/").strip("/") for prefix in (prefixes or [])]
    prefixes = sorted({prefix for prefix in prefixes if prefix})
    if not prefixes:
        return "Remote chunkcache clear skipped: no manifest cache prefixes."
    if api is None:
        raise RuntimeError("huggingface_hub is not installed; cannot clear remote chunkcache.")
    ensure_dataset_exists()

    repo_files = api.list_repo_files(repo_id=DATASET_REPO, repo_type="dataset", token=HF_TOKEN)
    existing_prefixes = []
    for prefix in prefixes:
        folder_prefix = prefix.rstrip("/") + "/"
        if any(path == prefix or path.startswith(folder_prefix) for path in repo_files):
            existing_prefixes.append(prefix)
    if not existing_prefixes:
        return "Remote chunkcache clear: no existing dataset cache folders matched this batch."

    if CommitOperationDelete is not None:
        operations = [CommitOperationDelete(path_in_repo=prefix, is_folder=True) for prefix in existing_prefixes]
        api.create_commit(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            operations=operations,
            commit_message=commit_message or f"Force clear VieNeu TTS chunkcache ({len(existing_prefixes)} folder(s))",
            token=HF_TOKEN,
        )
        return "Remote chunkcache cleared in one commit: " + ", ".join(existing_prefixes)

    # Fallback for old huggingface_hub versions. This may create one commit per folder.
    for prefix in existing_prefixes:
        api.delete_folder(
            path_in_repo=prefix,
            repo_id=DATASET_REPO,
            repo_type="dataset",
            commit_message=commit_message or f"Force clear VieNeu TTS chunkcache: {prefix}",
            token=HF_TOKEN,
        )
    return "Remote chunkcache cleared with fallback delete_folder: " + ", ".join(existing_prefixes)



def sync_remote_cache_folder_to_local(cache_name, clear_local_first=True):
    """Refresh one manifest cache folder from the Hugging Face Dataset.

    When repo-cache-only is enabled, this function makes the Dataset cache the
    source of truth: local chunkcache/<manifest> is cleared first, then refreshed
    from chunkcache/<manifest> in the repo in one folder-level pass. This avoids
    slow per-segment lazy downloads during render.
    """
    cache_name = build_manifest_cache_name(cache_name)
    remote_prefix = build_remote_cache_prefix(cache_name).strip("/")
    local_dir = build_manifest_cache_dir(cache_name)
    if clear_local_first and local_dir.exists():
        shutil.rmtree(local_dir, ignore_errors=True)
    local_dir.mkdir(parents=True, exist_ok=True)

    if api is None:
        return {
            "enabled": True,
            "status": "skipped",
            "reason": "huggingface_hub is unavailable",
            "files": 0,
            "remote_prefix": remote_prefix,
            "local_dir": str(local_dir.resolve()),
        }
    ensure_dataset_exists()
    try:
        repo_files = api.list_repo_files(repo_id=DATASET_REPO, repo_type="dataset", token=HF_TOKEN)
    except Exception as exc:
        return {
            "enabled": True,
            "status": "failed",
            "reason": f"list_repo_files failed: {exc}",
            "files": 0,
            "remote_prefix": remote_prefix,
            "local_dir": str(local_dir.resolve()),
        }
    matched = [path for path in repo_files if path.startswith(remote_prefix + "/")]
    if not matched:
        return {
            "enabled": True,
            "status": "empty",
            "reason": "no remote cache folder matched",
            "files": 0,
            "remote_prefix": remote_prefix,
            "local_dir": str(local_dir.resolve()),
        }

    copied = 0
    if snapshot_download is not None:
        try:
            snapshot_dir = snapshot_download(
                repo_id=DATASET_REPO,
                repo_type="dataset",
                allow_patterns=[remote_prefix + "/*"],
                token=HF_TOKEN,
            )
            source_dir = Path(snapshot_dir) / remote_prefix
            if source_dir.exists():
                for src in source_dir.rglob("*"):
                    if not src.is_file():
                        continue
                    rel = src.relative_to(source_dir)
                    dest = local_dir / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src, dest)
                    copied += 1
                return {
                    "enabled": True,
                    "status": "synced",
                    "method": "snapshot_download_folder",
                    "files": copied,
                    "remote_prefix": remote_prefix,
                    "local_dir": str(local_dir.resolve()),
                }
        except Exception:
            # Fall back to individual hf_hub_download below. It is slower, but keeps compatibility.
            copied = 0

    if hf_hub_download is None:
        return {
            "enabled": True,
            "status": "failed",
            "reason": "snapshot_download and hf_hub_download are unavailable",
            "files": 0,
            "remote_prefix": remote_prefix,
            "local_dir": str(local_dir.resolve()),
        }
    errors = []
    for remote_path in matched:
        try:
            downloaded = hf_hub_download(
                repo_id=DATASET_REPO,
                repo_type="dataset",
                filename=remote_path,
                token=HF_TOKEN,
            )
            rel = Path(remote_path).relative_to(remote_prefix)
            dest = local_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(downloaded, dest)
            copied += 1
        except Exception as exc:
            if len(errors) < 5:
                errors.append(f"{remote_path}: {exc}")
    return {
        "enabled": True,
        "status": "synced" if copied else "failed",
        "method": "hf_hub_download_fallback",
        "files": copied,
        "errors": errors,
        "remote_prefix": remote_prefix,
        "local_dir": str(local_dir.resolve()),
    }


def upload_manifest_cache_folder_once(cache_name, commit_message=None):
    """Commit one manifest cache folder to the Dataset as a single compact folder commit."""
    if api is None:
        raise RuntimeError("huggingface_hub is not installed; cannot commit manifest cache folder.")
    ensure_dataset_exists()
    cache_name = build_manifest_cache_name(cache_name)
    manifest_cache_dir = build_manifest_cache_dir(cache_name)
    if not manifest_cache_dir.exists() or not any(manifest_cache_dir.rglob('*')):
        return f"Manifest cache commit skipped: no files in {manifest_cache_dir}."
    remote_prefix = build_remote_cache_prefix(cache_name)
    raw_cleanup = clear_remote_raw_success_payload_if_migrated(cache_name)
    tmp = tempfile.TemporaryDirectory(prefix=f"compact_manifest_cache_{sanitize_filename(cache_name)}_")
    try:
        filtered_dir = Path(tmp.name) / cache_name
        filtered_dir.mkdir(parents=True, exist_ok=True)
        snapshot_stats = build_filtered_manifest_cache_snapshot(manifest_cache_dir, filtered_dir)
        if not any(filtered_dir.rglob('*')):
            return f"Manifest cache commit skipped: no uploadable compact cache files in {manifest_cache_dir}."
        api.upload_folder(
            folder_path=str(filtered_dir),
            path_in_repo=remote_prefix,
            repo_id=DATASET_REPO,
            repo_type='dataset',
            commit_message=commit_message or f"Commit compact VieNeu TTS cache folder: {cache_name}",
        )
        return (
            f"Committed compact manifest cache folder in one commit: {remote_prefix} "
            f"| uploaded={snapshot_stats.get('copied', 0)} skipped_raw_success={snapshot_stats.get('skipped', 0)} "
            f"| remote_raw_success_deleted={raw_cleanup.get('deleted', 0)}" + (f" | raw_success_cleanup_skipped={raw_cleanup.get('reason')}" if raw_cleanup.get('skipped') else "")
        )
    finally:
        tmp.cleanup()



def list_folder_files_snapshot(src_dir):
    """Freeze the file list at button-click time so active workers cannot extend the commit forever."""
    src = Path(src_dir)
    if not src.exists() or not src.is_dir():
        return []
    try:
        return [item for item in src.rglob("*") if item.is_file()]
    except Exception:
        return []


def copy_folder_contents(src_dir, dst_dir, file_snapshot=None, filter_fn=None):
    """Copy a frozen snapshot of files from one folder into another, preserving relative paths.

    Important for the manual force-cache button: workers may still be rendering and
    writing new cache files while the button is running. We must copy only the files
    that existed when the button was clicked, otherwise the commit can keep chasing
    newly-created segment cache files.
    """
    src = Path(src_dir)
    dst = Path(dst_dir)
    if not src.exists() or not src.is_dir():
        return {"copied": 0, "skipped": 0, "errors": []}
    files = list(file_snapshot) if file_snapshot is not None else list_folder_files_snapshot(src)
    copied = 0
    skipped = 0
    errors = []
    for item in files:
        try:
            item = Path(item)
            if not item.exists() or not item.is_file():
                skipped += 1
                continue
            rel = item.relative_to(src)
            if filter_fn is not None and not filter_fn(rel.as_posix()):
                skipped += 1
                continue
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(item, target)
            copied += 1
        except Exception as exc:
            skipped += 1
            if len(errors) < 8:
                errors.append(f"{item}: {exc}")
    return {"copied": copied, "skipped": skipped, "errors": errors}


def collect_current_manifest_cache_names():
    """Return manifest cache folder names currently present locally or in pending uploads."""
    names = set()
    if CHUNK_CACHE_ROOT.exists():
        for item in CHUNK_CACHE_ROOT.iterdir():
            if item.is_dir() and any(item.rglob("*")):
                names.add(item.name)
    pending_cache_root = PENDING_UPLOAD_ROOT / "chunkcache"
    if pending_cache_root.exists():
        for item in pending_cache_root.iterdir():
            if item.is_dir() and any(item.rglob("*")):
                names.add(item.name)
    return sorted(names)


def parse_manifest_cache_names(manifest_names_text):
    """Parse optional UI text into sanitized manifest cache names."""
    raw = str(manifest_names_text or "").strip()
    if not raw:
        return []
    parts = re.split(r"[\n,;]+", raw)
    names = []
    for part in parts:
        value = part.strip()
        if not value:
            continue
        # Accept either manifest file names, cache names, or chunkcache/<name> paths.
        value = value.replace("\\", "/").strip("/")
        if value.startswith("chunkcache/"):
            value = value.split("/", 1)[1]
        value = Path(value).name if "/" in value else value
        name = build_manifest_cache_name(value)
        if name and name not in names:
            names.append(name)
    return names


def force_commit_current_manifest_caches(manifest_names_text=""):
    """Button handler: force commit current cache folders by manifest.

    Blank input commits every manifest cache folder currently found in:
    - chunkcache/<manifest>
    - pending_hf_upload/chunkcache/<manifest>

    Each manifest is merged into a temporary folder and uploaded with one commit.
    Pending cache files for a manifest are removed only after that manifest commit succeeds.
    Pending final outputs are not touched.
    """
    if api is None:
        return "Force cache commit failed: huggingface_hub is not installed."
    if not HF_TOKEN:
        return "Force cache commit failed: HF_TOKEN is empty."
    try:
        ensure_dataset_exists()
        names = parse_manifest_cache_names(manifest_names_text)
        if not names:
            names = collect_current_manifest_cache_names()
        if not names:
            return "No manifest cache folders found in chunkcache/ or pending_hf_upload/chunkcache/."

        logs = [
            f"Force committing {len(names)} manifest cache folder(s) to {DATASET_REPO}...",
            "Snapshot mode: only cache files that exist at button-click time are copied; new files written by active workers are ignored until the next click.",
        ]
        committed = 0
        skipped = 0
        failed = 0
        pending_cache_root = PENDING_UPLOAD_ROOT / "chunkcache"

        # Freeze folder names and file lists immediately. This prevents active workers
        # from extending this manual commit while they continue receiving/rendering segments.
        manifest_snapshots = []
        for cache_name in names:
            cache_name = build_manifest_cache_name(cache_name)
            local_dir = CHUNK_CACHE_ROOT / cache_name
            pending_dir = pending_cache_root / cache_name
            local_snapshot = list_folder_files_snapshot(local_dir)
            pending_snapshot = list_folder_files_snapshot(pending_dir)
            manifest_snapshots.append((cache_name, local_dir, pending_dir, local_snapshot, pending_snapshot))

        for cache_name, local_dir, pending_dir, local_snapshot, pending_snapshot in manifest_snapshots:
            local_files = len(local_snapshot)
            pending_files = len(pending_snapshot)
            if local_files <= 0 and pending_files <= 0:
                skipped += 1
                logs.append(f"SKIP {cache_name}: no local/pending cache files in snapshot.")
                continue

            tmp = tempfile.TemporaryDirectory(prefix=f"force_cache_commit_{sanitize_filename(cache_name)}_")
            try:
                merged_dir = Path(tmp.name) / cache_name
                merged_dir.mkdir(parents=True, exist_ok=True)
                local_copy = copy_folder_contents(local_dir, merged_dir, file_snapshot=local_snapshot, filter_fn=should_upload_remote_cache_rel)
                pending_copy = copy_folder_contents(pending_dir, merged_dir, file_snapshot=pending_snapshot, filter_fn=should_upload_remote_cache_rel)
                copied_local = int(local_copy.get("copied", 0))
                copied_pending = int(pending_copy.get("copied", 0))
                copy_skipped = int(local_copy.get("skipped", 0)) + int(pending_copy.get("skipped", 0))
                merged_files = sum(1 for p in merged_dir.rglob("*") if p.is_file())
                if merged_files <= 0:
                    skipped += 1
                    logs.append(f"SKIP {cache_name}: snapshot copied no compact cache files.")
                    continue

                remote_prefix = build_remote_cache_prefix(cache_name)
                raw_cleanup = clear_remote_raw_success_payload_if_migrated(cache_name)
                api.upload_folder(
                    folder_path=str(merged_dir),
                    path_in_repo=remote_prefix,
                    repo_id=DATASET_REPO,
                    repo_type="dataset",
                    commit_message=f"Force commit current compact VieNeu TTS cache: {cache_name}",
                )
                committed += 1
                extra = f", skipped_during_snapshot_copy={copy_skipped}" if copy_skipped else ""
                logs.append(
                    f"OK {cache_name}: committed compact snapshot {merged_files} file(s) to {remote_prefix} "
                    f"(local_snapshot={local_files}, pending_snapshot={pending_files}, "
                    f"copied_local={copied_local}, copied_pending={copied_pending}{extra}, remote_raw_success_deleted={raw_cleanup.get('deleted', 0)})."
                )
                copy_errors = (local_copy.get("errors") or []) + (pending_copy.get("errors") or [])
                for err in copy_errors[:8]:
                    logs.append(f"  copy warning: {err}")

                # Cache files staged under pending upload have now been committed directly.
                # Do not delete pending final outputs; only remove this manifest cache folder.
                if pending_dir.exists():
                    shutil.rmtree(pending_dir, ignore_errors=True)
            except Exception as exc:
                failed += 1
                logs.append(f"FAILED {cache_name}: {exc}")
            finally:
                tmp.cleanup()

        # Best effort cleanup of empty pending cache root.
        try:
            if pending_cache_root.exists() and not any(pending_cache_root.iterdir()):
                pending_cache_root.rmdir()
        except Exception:
            pass
        logs.append(f"Done. committed={committed}, skipped={skipped}, failed={failed}.")
        return "\n".join(logs)
    except Exception as exc:
        return f"Force cache commit failed: {exc}"

def upload_chunk_cache_file(local_file, cache_name, remote_name):
    if api is None:
        raise RuntimeError("huggingface_hub is not installed in this environment.")
    ensure_dataset_exists()
    api.upload_file(
        path_or_fileobj=local_file,
        path_in_repo=f"{build_remote_cache_prefix(cache_name)}/{remote_name}",
        repo_id=DATASET_REPO,
        repo_type="dataset",
    )


def stage_file_for_batch_upload(local_file, staging_root, remote_path):
    if not local_file or not os.path.exists(local_file):
        return None
    safe_remote_path = str(remote_path or "").replace("\\", "/").lstrip("/")
    if not safe_remote_path:
        return None
    dest = os.path.join(staging_root, safe_remote_path)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copyfile(local_file, dest)
    return dest


def stage_upload_entries(upload_entries, staging_root):
    staged = []
    for local_file, remote_path in upload_entries or []:
        staged_file = stage_file_for_batch_upload(local_file, staging_root, remote_path)
        if staged_file:
            staged.append(staged_file)
    return staged


def drain_upload_entries(upload_entries):
    """Take a snapshot of staged upload entries and clear the shared list.

    Workers append to this list after each file finishes. The batch loop drains it
    periodically so long jobs can sync every N minutes, then drains it one last
    time at batch end.
    """
    if not upload_entries:
        return []
    entries = list(upload_entries)
    upload_entries.clear()
    return entries


def upload_staged_folder_once(staging_root, commit_message=None):
    if api is None:
        raise RuntimeError("huggingface_hub is not installed in this environment.")
    ensure_dataset_exists()
    if not os.path.isdir(staging_root):
        raise RuntimeError(f"Batch upload staging folder does not exist: {staging_root}")
    has_files = any(Path(staging_root).rglob("*"))
    if not has_files:
        return "No files staged for upload."
    api.upload_folder(
        folder_path=staging_root,
        path_in_repo="",
        repo_id=DATASET_REPO,
        repo_type="dataset",
        commit_message=commit_message or "Batch upload VieNeu TTS outputs",
        ignore_patterns=["_sync_state.json"],
    )
    return f"Uploaded staged folder to dataset {DATASET_REPO} in one commit."


def download_chunk_cache_file(cache_name, remote_name, local_path):
    if hf_hub_download is None:
        return False
    try:
        ensure_dataset_exists()
        downloaded = hf_hub_download(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            filename=f"{build_remote_cache_prefix(cache_name)}/{remote_name}",
            token=HF_TOKEN,
        )
    except Exception:
        return False
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    shutil.copyfile(downloaded, local_path)
    return True


def read_uploaded_text_file(file):
    with open(file.name, "r", encoding="utf-8") as f:
        return f.read()


def read_json_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def input_picker_config(mode):
    if mode == DEFAULT_MANIFEST_MODE:
        return {
            "label": "Upload multi-voice manifest JSON files",
            "file_types": MANIFEST_FILE_TYPES,
            "info": "Nhan moi file .json bat ky, vi du: chapter_01.json, my_segments.json. Khong bat buoc ten audio_segments.json.",
        }
    return {
        "label": "Upload TXT files",
        "file_types": PLAIN_FILE_TYPES,
        "info": "Single-voice mode chi nhan file .txt.",
    }


def update_input_picker(mode):
    cfg = input_picker_config(mode)
    return gr.update(
        label=cfg["label"],
        file_types=cfg["file_types"],
        value=None,
    ), cfg["info"]


def validate_uploaded_file_for_mode(file, mode):
    suffix = Path(getattr(file, "name", "")).suffix.lower()
    if mode == DEFAULT_MANIFEST_MODE:
        if suffix != ".json":
            raise ValueError(
                "Multi-voice mode chi nhan file .json. Ban co the dung bat ky ten nao, "
                "vi du: chapter_01.json hoac filename.json."
            )
        return
    if suffix != ".txt":
        raise ValueError("Single-voice mode chi nhan file .txt.")


def coerce_int(value, default=None):
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def default_watermark_plan():
    return {
        "enabled": False,
        "mode": "inline_interval",
        "text": "",
        "insert_at_intro": False,
        "insert_at_outro": False,
        "target_interval_sec": 900,
    }


def format_vtt_time(t):
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h:02}:{m:02}:{s:06.3f}"


def format_srt_time(t):
    return format_vtt_time(t).replace(".", ",")


def write_subtitle_file(events, subtitle_path, subtitle_format):
    if subtitle_format == "no_script":
        return
    with open(subtitle_path, "w", encoding="utf-8") as f:
        if subtitle_format == "vtt":
            f.write("WEBVTT\n\n")
        for idx, event in enumerate(events, start=1):
            start = event["start"]
            end = event["end"]
            text = event["text"]
            f.write(f"{idx}\n")
            if subtitle_format == "vtt":
                f.write(f"{format_vtt_time(start)} --> {format_vtt_time(end)}\n")
            else:
                f.write(f"{format_srt_time(start)} --> {format_srt_time(end)}\n")
            f.write(f"{text}\n\n")


async def synthesize_audio_once(text, voice, rate, pitch, audio_path, timeout_sec=None, emotion=None):
    if edge_tts is None:
        raise RuntimeError("edge_tts is not installed in this environment.")
    communicate = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate=f"{int(rate):+d}%",
        pitch=f"{int(pitch):+d}Hz",
    )
    events = []
    audio_received = False

    async def _collect_stream():
        nonlocal audio_received
        with open(audio_path, "wb") as audio_file:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_received = True
                    audio_file.write(chunk["data"])
                elif chunk["type"] == "SentenceBoundary":
                    start = chunk["offset"] / 10_000_000
                    duration = chunk["duration"] / 10_000_000
                    end = start + duration
                    text_part = (chunk.get("text") or "").strip()
                    if text_part:
                        events.append({"start": start, "end": end, "text": text_part})

    timeout_sec = int(timeout_sec or SEGMENT_TTS_TIMEOUT_SEC)
    try:
        await asyncio.wait_for(_collect_stream(), timeout=timeout_sec)
    except asyncio.TimeoutError as exc:
        if os.path.exists(audio_path):
            os.remove(audio_path)
        raise TimeoutError(
            f"vieneu_tts timeout after {timeout_sec}s "
            f"(voice={voice}, text_len={len(text)})"
        ) from exc
    if not audio_received:
        raise RuntimeError("No audio was received from edge_tts.")
    return events


def clean_tts_text(text):
    text = str(text or "")
    # Normalize whitespace and strip zero-width/control chars that can make TTS unstable.
    text = text.replace("\ufeff", "").replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
    text = "".join(ch for ch in text if ch in ("\n", "\t") or ord(ch) >= 32)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_text_into_tts_chunks(text, max_chars=SEGMENT_TTS_MAX_CHARS):
    cleaned = clean_tts_text(text)
    if not cleaned:
        return []
    if len(cleaned) <= max_chars:
        return [cleaned]

    sentence_parts = []
    cursor = 0
    for match in re.finditer(r".+?(?:[.!?;:]+|$)", cleaned):
        part = match.group(0).strip()
        if part:
            sentence_parts.append(part)
        cursor = match.end()
    if not sentence_parts or cursor < len(cleaned):
        tail = cleaned[cursor:].strip()
        if tail:
            sentence_parts.append(tail)

    normalized_parts = []
    for part in sentence_parts:
        if len(part) <= max_chars:
            normalized_parts.append(part)
            continue
        words = part.split()
        chunk = ""
        for word in words:
            candidate = f"{chunk} {word}".strip()
            if candidate and len(candidate) <= max_chars:
                chunk = candidate
                continue
            if chunk:
                normalized_parts.append(chunk)
            if len(word) <= max_chars:
                chunk = word
                continue
            for i in range(0, len(word), max_chars):
                normalized_parts.append(word[i : i + max_chars])
            chunk = ""
        if chunk:
            normalized_parts.append(chunk)

    chunks = []
    current = ""
    for part in normalized_parts:
        candidate = f"{current} {part}".strip()
        if candidate and len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        current = part
    if current:
        chunks.append(current)
    return chunks


def offset_events(events, offset_sec):
    return [
        {
            "start": round(offset_sec + event["start"], 3),
            "end": round(offset_sec + event["end"], 3),
            "text": event["text"],
        }
        for event in events
    ]


async def synthesize_audio(text, voice, rate, pitch, audio_path, timeout_sec=None):
    chunks = split_text_into_tts_chunks(text, max_chars=SEGMENT_TTS_MAX_CHARS)
    if not chunks:
        raise ValueError("Text is empty after cleanup.")
    if len(chunks) == 1:
        return await synthesize_audio_once(chunks[0], voice, rate, pitch, audio_path, timeout_sec=timeout_sec)

    audio_dir = os.path.dirname(audio_path) or "."
    stem = Path(audio_path).stem
    suffix = Path(audio_path).suffix or ".mp3"
    part_files = []
    merged_events = []
    cursor = 0.0
    for idx, chunk_text in enumerate(chunks, start=1):
        chunk_path = os.path.join(audio_dir, f"{stem}_chunk_{idx:04d}{suffix}")
        last_error = None
        for attempt in range(1, SEGMENT_TTS_RETRIES + 1):
            try:
                chunk_events = await synthesize_audio_once(chunk_text, voice, rate, pitch, chunk_path, timeout_sec=timeout_sec)
                chunk_events = ensure_event_fallback(chunk_events, chunk_text, rate)
                merged_events.extend(offset_events(chunk_events, cursor))
                event_duration = max(event["end"] for event in chunk_events)
                actual_duration = get_audio_duration_sec(chunk_path)
                chunk_duration = max(float(actual_duration or 0), float(event_duration or 0))
                cursor = round(cursor + chunk_duration, 3)
                part_files.append(chunk_path)
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)
                if attempt < SEGMENT_TTS_RETRIES:
                    await asyncio.sleep(SEGMENT_RETRY_BACKOFF_SEC * attempt)
        if last_error is not None:
            for part_file in part_files:
                if os.path.exists(part_file):
                    os.remove(part_file)
            raise RuntimeError(
                f"Chunk {idx}/{len(chunks)} failed after {SEGMENT_TTS_RETRIES} retries "
                f"(voice={voice}, text_len={len(chunk_text)}): {last_error}"
            ) from last_error
    concat_audio_files(part_files, audio_path)
    for part_file in part_files:
        if os.path.exists(part_file):
            os.remove(part_file)
    return merged_events


def segment_debug_label(segment, profile=None):
    profile = profile or {}
    text = clean_tts_text(segment.get("text", ""))
    excerpt = text[:80] + ("..." if len(text) > 80 else "")
    return (
        f"segment_id={segment.get('segment_id', '?')} "
        f"order={segment.get('order', '?')} "
        f"type={segment.get('segment_type', '?')} "
        f"voice={profile.get('voice') or segment.get('voice') or '?'} "
        f"emotion={profile.get('emotion') or segment.get('emotion') or segment.get('vieneu_emotion') or '?'} "
        f"rate={profile.get('rate_pct', segment.get('rate_pct', '?'))} "
        f"pitch={profile.get('pitch_hz', segment.get('pitch_hz', '?'))} "
        f"text_len={len(text)} "
        f"text={excerpt!r}"
    )


def has_speech_content(text):
    return bool(re.search(r"\w", str(text or ""), flags=re.UNICODE))


def is_punctuation_only_text(text):
    cleaned = clean_tts_text(text)
    return bool(cleaned) and not has_speech_content(cleaned)


def segment_text_len(segment):
    return len(clean_tts_text(segment.get("text", "")))


def segment_word_count(text_or_segment):
    if isinstance(text_or_segment, dict):
        text = text_or_segment.get("text", "")
    else:
        text = text_or_segment
    return len(re.findall(r"\w+", clean_tts_text(text), flags=re.UNICODE))


def has_soft_continuation_ending(text):
    return clean_tts_text(text).endswith((",", ";", ":", "-", "(", "[", "{"))


def has_strong_sentence_ending(text):
    return clean_tts_text(text).endswith((".", "!", "?", "?", '"', "'", "?", "?"))


def is_dialogue_segment(segment):
    return str((segment or {}).get("segment_type") or "").strip().lower() == "dialogue"


def choose_merge_target(segments, idx):
    current = segments[idx]
    prev_seg = segments[idx - 1] if idx > 0 else None
    next_seg = segments[idx + 1] if idx + 1 < len(segments) else None
    text = clean_tts_text(current.get("text", ""))
    if prev_seg is None and next_seg is None:
        return None, None
    if prev_seg is None:
        return idx + 1, "prepend"
    if next_seg is None:
        return idx - 1, "append"

    def score(candidate):
        score_value = 0
        if candidate.get("segment_type") == current.get("segment_type"):
            score_value += 4
        if candidate.get("voice_profile_key") == current.get("voice_profile_key"):
            score_value += 3
        if candidate.get("voice_mode") == current.get("voice_mode"):
            score_value += 2
        score_value += min(segment_text_len(candidate), 24) / 24.0
        return score_value

    opening_chars = set(['"', "'", "(", "[", "{", "<"])
    closing_chars = set(['"', "'", ")", "]", "}", ">", "!", "?", ".", ",", ";", ":"])
    opening_only = text and all(ch in opening_chars for ch in text)
    closing_only = text and all(ch in closing_chars for ch in text)
    prev_is_dialogue = is_dialogue_segment(prev_seg)
    next_is_dialogue = is_dialogue_segment(next_seg)

    if is_dialogue_segment(current):
        if next_is_dialogue and not prev_is_dialogue:
            return idx + 1, "prepend"
        if prev_is_dialogue and not next_is_dialogue:
            return idx - 1, "append"
        if next_is_dialogue and prev_is_dialogue:
            if opening_only or has_soft_continuation_ending(text):
                return idx + 1, "prepend"
            if closing_only:
                return idx - 1, "append"
            next_score = score(next_seg) + (1.5 if not has_strong_sentence_ending(text) else 0.25)
            prev_score = score(prev_seg)
            return (idx + 1, "prepend") if next_score >= prev_score else (idx - 1, "append")

    if opening_only:
        return idx + 1, "prepend"
    if closing_only:
        return idx - 1, "append"
    if has_soft_continuation_ending(text):
        return idx + 1, "prepend"
    if not has_strong_sentence_ending(text) and next_seg is not None:
        next_score = score(next_seg) + 1.25
        prev_score = score(prev_seg)
        if next_score >= prev_score:
            return idx + 1, "prepend"

    prev_score = score(prev_seg)
    next_score = score(next_seg)
    if next_score > prev_score:
        return idx + 1, "prepend"
    return idx - 1, "append"


def merge_segment_into_target(target, segment, direction, reason):
    target_text = clean_tts_text(target.get("text", ""))
    segment_text = clean_tts_text(segment.get("text", ""))
    if direction == "prepend":
        target["text"] = f"{segment_text} {target_text}".strip()
        target["source_start_line"] = min(
            int(segment.get("source_start_line", target.get("source_start_line", 0)) or 0),
            int(target.get("source_start_line", segment.get("source_start_line", 0)) or 0),
        )
    else:
        target["text"] = f"{target_text} {segment_text}".strip()
        target["source_end_line"] = max(
            int(segment.get("source_end_line", target.get("source_end_line", 0)) or 0),
            int(target.get("source_end_line", segment.get("source_end_line", 0)) or 0),
        )
    detection = target.setdefault("detection", {})
    detection_reason = str(detection.get("reason") or "").strip()
    extra = f"merged_{reason}:{segment.get('segment_id') or '?'}"
    detection["reason"] = f"{detection_reason}; {extra}".strip("; ").strip()
    detection["merged_segments"] = int(detection.get("merged_segments", 0) or 0) + 1
    return target


def repair_manifest_segments(
    segments,
    min_chars=MANIFEST_MIN_SEGMENT_CHARS,
    min_words=MANIFEST_MIN_SEGMENT_WORDS,
):
    working = [deepcopy(seg) for seg in (segments or []) if clean_tts_text(seg.get("text", ""))]
    if len(working) <= 1:
        for idx, seg in enumerate(working, start=1):
            seg["order"] = idx
            seg["segment_id"] = seg.get("segment_id") or f"seg_{idx:06d}"
        return working

    changed = True
    while changed:
        changed = False
        idx = 0
        while idx < len(working):
            seg = working[idx]
            seg_text = clean_tts_text(seg.get("text", ""))
            seg_words = segment_word_count(seg_text)
            should_merge = is_punctuation_only_text(seg_text) or (
                seg.get("segment_type") != "seo_tag"
                and (
                    len(seg_text) < int(min_chars or 1)
                    or seg_words < int(min_words or 1)
                )
            )
            if not should_merge:
                idx += 1
                continue
            target_idx, direction = choose_merge_target(working, idx)
            if target_idx is None:
                idx += 1
                continue
            reason = "punct_only" if is_punctuation_only_text(seg_text) else "short"
            working[target_idx] = merge_segment_into_target(working[target_idx], seg, direction, reason)
            del working[idx]
            changed = True
            if target_idx > idx:
                idx = max(0, idx - 1)
            continue

    for idx, seg in enumerate(working, start=1):
        seg["order"] = idx
        seg["segment_id"] = f"seg_{idx:06d}"
    return working


def estimate_duration_from_text(text, rate_pct):
    chars_per_sec = 11.5 + (float(rate_pct) * 0.08)
    return round(max(1.0, len((text or "").strip()) / max(chars_per_sec, 6.0)), 3)


def ensure_event_fallback(events, text, rate_pct):
    if events:
        return events
    duration = estimate_duration_from_text(text, rate_pct)
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if not cleaned:
        cleaned = "..."
    return [{"start": 0.0, "end": duration, "text": cleaned}]


def find_ffmpeg():
    return shutil.which("ffmpeg")


def find_ffprobe():
    return shutil.which("ffprobe")


def get_audio_duration_sec(audio_path):
    """Return actual media duration using ffprobe, or None if unavailable.

    Edge TTS sentence-boundary timestamps can be slightly shorter than the real MP3
    stream because of encoder padding / trailing audio. For subtitle alignment, the
    concat cursor should advance by the real audio duration when possible.
    """
    ffprobe = find_ffprobe()
    if not ffprobe or not audio_path or not os.path.exists(audio_path):
        return None
    try:
        completed = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                audio_path,
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        value = float((completed.stdout or "").strip())
        if value > 0:
            return round(value, 3)
    except Exception:
        return None
    return None


def build_silence_audio(duration_ms, output_path):
    ffmpeg = find_ffmpeg()
    if not ffmpeg or duration_ms <= 0:
        return None
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=r=24000:cl=mono",
        "-t",
        f"{duration_ms / 1000.0:.3f}",
        "-q:a",
        "9",
        "-acodec",
        "libmp3lame",
        output_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_path



def get_or_build_silence_audio(duration_ms, tmpdir=None, silence_cache=None):
    """Return a reusable silence MP3 for a pause duration.

    The old code created one pause file per segment. For long manifests, that can
    mean thousands of ffmpeg calls. This caches by pause duration, for example
    silence_220ms.mp3, then reuses the same file in the concat list.
    """
    duration_ms = int(duration_ms or 0)
    if duration_ms <= 0:
        return None
    if silence_cache is not None and duration_ms in silence_cache and os.path.exists(silence_cache[duration_ms]):
        return silence_cache[duration_ms]
    root = Path(tmpdir) / "_silence_cache" if tmpdir else SILENCE_CACHE_ROOT
    root.mkdir(parents=True, exist_ok=True)
    output_path = root / f"silence_{duration_ms}ms.mp3"
    if not output_path.exists():
        build_silence_audio(duration_ms, str(output_path))
    if not output_path.exists():
        return None
    if silence_cache is not None:
        silence_cache[duration_ms] = str(output_path)
    return str(output_path)

def concat_audio_files(segment_files, final_path):
    ffmpeg = find_ffmpeg()
    if ffmpeg:
        list_path = Path(final_path).with_suffix(".concat.txt")
        with open(list_path, "w", encoding="utf-8") as f:
            for part in segment_files:
                f.write(f"file '{Path(part).resolve().as_posix()}'\n")
        try:
            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(list_path),
                    "-c",
                    "copy",
                    final_path,
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            return {"method": "ffmpeg_concat_demuxer"}
        finally:
            if list_path.exists():
                list_path.unlink()
    with open(final_path, "wb") as out:
        for part in segment_files:
            with open(part, "rb") as src:
                shutil.copyfileobj(src, out)
    return {"method": "binary_append_fallback"}


def detect_audio_active_bounds_sec(audio_path, duration_sec=None):
    """Estimate non-silent bounds of a rendered segment MP3 using ffmpeg silencedetect.

    This does not replace real forced alignment, but it is safer than trusting Edge TTS
    SentenceBoundary offsets blindly. It lets us avoid showing subtitles before the
    segment's actual audible speech when the MP3 has leading encoder/silence padding.
    """
    ffmpeg = find_ffmpeg()
    duration = float(duration_sec or get_audio_duration_sec(audio_path) or 0.0)
    if not ffmpeg or not audio_path or not os.path.exists(audio_path) or duration <= 0:
        return {"speech_start_sec": 0.0, "speech_end_sec": round(duration, 3)}
    try:
        completed = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-nostats",
                "-i",
                audio_path,
                "-af",
                f"silencedetect=n={SILENCE_DETECT_NOISE_DB}dB:d={SILENCE_DETECT_MIN_DURATION_SEC}",
                "-f",
                "null",
                "-",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        log = (completed.stderr or "") + "\n" + (completed.stdout or "")
        silence_starts = [float(x) for x in re.findall(r"silence_start:\s*([0-9.]+)", log)]
        silence_ends = [float(x) for x in re.findall(r"silence_end:\s*([0-9.]+)", log)]

        speech_start = 0.0
        if silence_starts and abs(silence_starts[0]) <= 0.05 and silence_ends:
            speech_start = min(max(silence_ends[0], 0.0), duration)

        speech_end = duration
        if silence_starts and silence_starts[-1] < duration and (not silence_ends or len(silence_starts) > len(silence_ends)):
            speech_end = min(max(silence_starts[-1], speech_start), duration)

        # Avoid over-aggressive silence detection for very short/quiet speech.
        if speech_end - speech_start < 0.25:
            speech_start, speech_end = 0.0, duration
        return {"speech_start_sec": round(speech_start, 3), "speech_end_sec": round(speech_end, 3)}
    except Exception:
        return {"speech_start_sec": 0.0, "speech_end_sec": round(duration, 3)}


def normalize_segment_subtitle_events(events, audio_duration_sec, speech_start_sec=0.0, speech_end_sec=None):
    """Clamp Edge TTS events to the measured segment audio duration.

    Edge TTS SentenceBoundary is still useful for sentence splits, but the segment
    start/end is anchored to measured audio/speech bounds. This prevents cumulative
    drift and reduces subtitles appearing before the actual speech.
    """
    audio_duration = max(0.001, float(audio_duration_sec or 0.001))
    speech_start = min(max(float(speech_start_sec or 0.0), 0.0), audio_duration)
    speech_end = float(speech_end_sec if speech_end_sec is not None else audio_duration)
    speech_end = min(max(speech_end, speech_start + 0.001), audio_duration)

    cleaned_events = []
    for raw in sorted(events or [], key=lambda item: float(item.get("start", 0) or 0)):
        text = str(raw.get("text") or "").strip()
        if not text:
            continue
        raw_start = max(0.0, float(raw.get("start", 0) or 0))
        raw_end = max(raw_start + 0.001, float(raw.get("end", raw_start) or raw_start))
        cleaned_events.append({"start": raw_start, "end": raw_end, "text": text})

    if not cleaned_events:
        return [{"start": speech_start, "end": speech_end, "text": "..."}]

    first_raw_start = cleaned_events[0]["start"]
    last_raw_end = max(event["end"] for event in cleaned_events)
    raw_span = max(0.001, last_raw_end - first_raw_start)
    target_span = max(0.001, speech_end - speech_start)

    normalized = []
    for idx, event in enumerate(cleaned_events):
        # Preserve Edge's relative sentence positions, but map them into real measured speech/audio bounds.
        start = speech_start + ((event["start"] - first_raw_start) / raw_span) * target_span
        end = speech_start + ((event["end"] - first_raw_start) / raw_span) * target_span
        start = min(max(start, speech_start), speech_end)
        end = min(max(end, start + 0.08), speech_end)
        if idx > 0:
            # Prevent overlaps caused by noisy Edge boundaries.
            prev_end = normalized[-1]["end"]
            if start < prev_end:
                start = min(prev_end, speech_end)
                end = min(max(end, start + 0.08), speech_end)
        normalized.append({"start": start, "end": end, "text": event["text"]})

    if SUBTITLE_EXTEND_LAST_EVENT_TO_AUDIO_END and normalized:
        normalized[-1]["end"] = speech_end
    return normalized


def build_global_subtitles(segment_events):
    events = []
    cursor = 0.0
    subtitle_delay_sec = max(0.0, float(SUBTITLE_GLOBAL_DELAY_MS or 0) / 1000.0)
    for item in segment_events:
        audio_duration_sec = float(item.get("audio_duration_sec", 0) or 0)
        pause_after_ms = int(item.get("pause_after_ms", 0) or 0)
        normalized_events = normalize_segment_subtitle_events(
            item.get("events") or [],
            audio_duration_sec=audio_duration_sec,
            speech_start_sec=float(item.get("speech_start_sec", 0.0) or 0.0),
            speech_end_sec=float(item.get("speech_end_sec", audio_duration_sec) or audio_duration_sec),
        )
        for event in normalized_events:
            start = cursor + event["start"] + subtitle_delay_sec
            end = cursor + event["end"] + subtitle_delay_sec
            # Never let a segment's subtitles bleed into the configured pause or the next segment.
            end = min(end, cursor + audio_duration_sec)
            if end <= start:
                end = min(cursor + audio_duration_sec, start + 0.08)
            events.append(
                {
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "text": event["text"],
                }
            )
        cursor = round(cursor + audio_duration_sec + pause_after_ms / 1000.0, 3)
    return events


def write_render_report(report_path, data):
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_manifest(path):
    data = read_json_file(path)
    return normalize_manifest(data, source_name=os.path.basename(path))


def load_manifest_from_text(manifest_text):
    data = json.loads(manifest_text)
    return normalize_manifest(data, source_name="preview_segments.txt")


def normalize_manifest(data, source_name=None):
    if isinstance(data, list):
        data = {
            "schema_version": SCHEMA_VERSION,
            "source_file": source_name or "preview_segments.txt",
            "source_final_path": "preview/manual_input",
            "language": "vi-VN",
            "audio_strategy": dict(DEFAULT_MANIFEST_STRATEGY),
            "watermark_plan": default_watermark_plan(),
            "segments": data,
        }
    if not isinstance(data, dict):
        return data
    manifest = dict(data)
    manifest["schema_version"] = manifest.get("schema_version") or SCHEMA_VERSION
    manifest["source_file"] = manifest.get("source_file") or source_name or "preview_segments.txt"
    manifest["source_final_path"] = manifest.get("source_final_path") or "preview/manual_input"
    manifest["language"] = manifest.get("language") or "vi-VN"
    strategy = dict(DEFAULT_MANIFEST_STRATEGY)
    strategy.update(manifest.get("audio_strategy") or {})
    manifest["audio_strategy"] = strategy
    watermark_plan = default_watermark_plan()
    watermark_plan.update(manifest.get("watermark_plan") or {})
    manifest["watermark_plan"] = watermark_plan
    normalized_segments = []
    for idx, raw_segment in enumerate(manifest.get("segments") or [], start=1):
        if not isinstance(raw_segment, dict):
            continue
        segment = dict(raw_segment)
        if not str(segment.get("segment_id", "")).strip():
            segment["segment_id"] = f"seg_{idx:06d}"
        segment["order"] = coerce_int(segment.get("order"), idx)
        for field in ("pause_after_ms", "rate_pct", "pitch_hz", "source_start_line", "source_end_line"):
            field_value = coerce_int(segment.get(field))
            if field_value is not None:
                segment[field] = field_value
        for field in ("speaker_source", "performed_voice_persona", "detection"):
            value = segment.get(field)
            if value is not None and not isinstance(value, dict):
                segment[field] = {}
        normalized_segments.append(segment)
    # Keep segment list exactly as the manifest provides it. Do NOT merge short segments here;
    # rescue grouping is optional and only operates after a segment has failed normal rendering.
    manifest["segments"] = normalized_segments
    return manifest


def validate_manifest(data):
    errors = []
    if not isinstance(data, dict):
        errors.append("Manifest must be an object.")
        return errors
    if data.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"Unsupported schema_version: {data.get('schema_version')}")
    if not isinstance(data.get("segments"), list) or not data["segments"]:
        errors.append("Manifest must contain a non-empty segments list.")
        return errors
    seen_segment_ids = set()
    seen_orders = set()
    for idx, seg in enumerate(data["segments"], start=1):
        segment_id = str(seg.get("segment_id", "")).strip()
        order = seg.get("order")
        if seg.get("segment_type") not in SEGMENT_TYPES:
            errors.append(f"segments[{idx}].segment_type invalid")
        if not str(seg.get("text", "")).strip():
            errors.append(f"segments[{idx}].text empty")
        if seg.get("voice_mode", "natural") not in VOICE_MODES:
            errors.append(f"segments[{idx}].voice_mode invalid")
        if not segment_id:
            errors.append(f"segments[{idx}].segment_id missing")
        elif segment_id in seen_segment_ids:
            errors.append(f"segments[{idx}].segment_id duplicate")
        else:
            seen_segment_ids.add(segment_id)
        if not isinstance(order, int):
            errors.append(f"segments[{idx}].order invalid")
        elif order in seen_orders:
            errors.append(f"segments[{idx}].order duplicate")
        else:
            seen_orders.add(order)
        pause_after_ms = seg.get("pause_after_ms", 0)
        if not isinstance(pause_after_ms, int) or pause_after_ms < 0:
            errors.append(f"segments[{idx}].pause_after_ms invalid")
        rate_pct = seg.get("rate_pct")
        if rate_pct is not None and not isinstance(rate_pct, int):
            errors.append(f"segments[{idx}].rate_pct invalid")
        pitch_hz = seg.get("pitch_hz")
        if pitch_hz is not None and not isinstance(pitch_hz, int):
            errors.append(f"segments[{idx}].pitch_hz invalid")
        for field in ("speaker_source", "performed_voice_persona"):
            obj = seg.get(field) or {}
            if obj.get("gender", "unknown") not in GENDERS:
                errors.append(f"segments[{idx}].{field}.gender invalid")
            if obj.get("age_tone", "unknown") not in AGE_TONES:
                errors.append(f"segments[{idx}].{field}.age_tone invalid")
            if obj.get("role_rank", "unknown") not in ROLE_RANKS:
                errors.append(f"segments[{idx}].{field}.role_rank invalid")
    if not errors:
        orders_sorted = sorted(seen_orders)
        expected_orders = list(range(1, len(data["segments"]) + 1))
        if orders_sorted != expected_orders:
            errors.append("segments.order must be unique contiguous integers starting at 1")
    return errors


def normalize_voice_name(value, default_voice="vi-VN-HoaiMyNeural"):
    raw = str(value or "").strip()
    if not raw:
        return default_voice
    return VOICE_ALIASES.get(raw) or VOICE_ALIASES.get(raw.lower()) or raw


def resolve_segment_tts_settings(segment, defaults):
    defaults = defaults or {}
    default_voice = (
        defaults.get("default_voice")
        or defaults.get("narrator_default_voice")
        or defaults.get("dialogue_default_voice")
        or "vi-VN-HoaiMyNeural"
    )
    voice = normalize_voice_name(segment.get("voice"), default_voice=default_voice)
    return {
        "voice": voice,
        "rate_pct": int(segment.get("rate_pct", defaults.get("default_rate_pct", 10))),
        "pitch_hz": int(segment.get("pitch_hz", defaults.get("default_pitch_hz", -8))),
        "pause_after_ms": int(segment.get("pause_after_ms", defaults.get("default_pause_after_ms", 220))),
    }


# Backward-compatible function name used by old debug/report code.
def resolve_voice_profile(segment, defaults):
    return resolve_segment_tts_settings(segment, defaults)


def segment_setting_key(segment, defaults=None):
    profile = resolve_segment_tts_settings(segment, defaults or DEFAULT_MANIFEST_STRATEGY)
    return (profile["voice"], int(profile["rate_pct"]), int(profile["pitch_hz"]), profile.get("emotion", get_vieneu_emotion()))


def is_short_segment(segment):
    text = clean_tts_text(segment.get("text", ""))
    if not text:
        return False
    return len(text) <= SHORT_SEGMENT_MAX_CHARS or segment_word_count(text) <= SHORT_SEGMENT_MAX_WORDS


def build_segment_subtitle_events_from_json_text(edge_events, json_text, audio_duration_sec, rate_pct=10):
    """Use Edge timing only as a hint; never use Edge text for SRT/VTT."""
    cleaned_text = clean_tts_text(json_text)
    duration = float(audio_duration_sec or 0)
    if duration <= 0:
        duration = estimate_duration_from_text(cleaned_text, rate_pct)
    if edge_events:
        start = min(float(event.get("start", 0) or 0) for event in edge_events)
        end = max(float(event.get("end", 0) or 0) for event in edge_events)
        start = max(0.0, min(start, duration))
        end = min(max(end, start + 0.08), duration)
    else:
        start, end = 0.0, duration
    return [{"start": round(start, 3), "end": round(end, 3), "text": cleaned_text}]


def classify_tts_failure(exc, segment=None):
    message = str(exc or "")
    lower = message.lower()
    text = clean_tts_text((segment or {}).get("text", ""))
    if "timeout" in lower or "timed out" in lower or "asyncio.timeouterror" in lower:
        return "timeout", "timeout"
    if "no audio" in lower or "empty" in lower or "0 bytes" in lower or is_short_segment(segment or {}):
        return "short_or_no_audio", "no_audio_or_short_text"
    if "text is empty" in lower:
        return "other", "empty_text"
    return "other", "other"


def make_segment_cache_key(segment, profile):
    payload = {
        "segment_id": segment.get("segment_id"),
        "text_hash": hashlib.sha256(segment.get("text", "").encode("utf-8")).hexdigest(),
        "voice": profile["voice"],
        "rate_pct": profile["rate_pct"],
        "pitch_hz": profile["pitch_hz"],
        "pause_after_ms": int(profile.get("pause_after_ms", segment.get("pause_after_ms", 0)) or 0),
        "emotion": profile.get("emotion", get_vieneu_emotion()),
    }
    packed = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(packed).hexdigest()


def build_manifest_cache_name(source_file):
    return sanitize_filename(source_file)


def build_manifest_cache_dir(source_file):
    cache_dir = CHUNK_CACHE_ROOT / build_manifest_cache_name(source_file)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def normalize_timeline_mode(value):
    value = str(value or DEFAULT_TIMELINE_MODE).strip().lower()
    if value not in TIMELINE_MODES:
        return DEFAULT_TIMELINE_MODE
    return value


def build_cache_index_path(manifest_cache_dir):
    return Path(manifest_cache_dir) / CACHE_INDEX_FILENAME


def build_manifest_cache_index(cache_name, manifest, manifest_cache_dir, defaults=None):
    defaults = defaults or DEFAULT_MANIFEST_STRATEGY
    cache_dir = Path(manifest_cache_dir)
    items = {}
    hits = 0
    missing = 0
    failed = 0
    for segment in sorted(manifest.get("segments") or [], key=lambda item: int(item.get("order", 0) or 0)):
        seg_id = str(segment.get("segment_id") or "")
        profile = resolve_segment_tts_settings(segment, defaults)
        cache_key = make_segment_cache_key(segment, profile)
        paths = get_segment_cache_paths(cache_dir, segment)
        failed_path = get_segment_failed_path(cache_dir, segment)
        status = "missing"
        meta_data = {}
        if paths["audio"].exists() and paths["meta"].exists():
            try:
                with open(paths["meta"], "r", encoding="utf-8") as f:
                    meta_data = json.load(f)
                if meta_data.get("cache_key") == cache_key:
                    status = str(meta_data.get("status") or "success")
                    hits += 1
                else:
                    status = "stale"
                    missing += 1
            except Exception:
                status = "bad_meta"
                missing += 1
        elif failed_path.exists():
            status = "failed"
            failed += 1
        else:
            missing += 1
        items[seg_id] = {
            "order": int(segment.get("order", 0) or 0),
            "status": status,
            "audio": paths["audio"].name,
            "meta": paths["meta"].name,
            "failed": failed_path.name,
            "cache_key": cache_key,
            "voice": profile.get("voice"),
            "rate_pct": int(profile.get("rate_pct", 0) or 0),
            "pitch_hz": int(profile.get("pitch_hz", 0) or 0),
            "audio_duration_sec": meta_data.get("audio_duration_sec"),
            "rescue_method": meta_data.get("rescue_method"),
        }
    index = {
        "schema": "edge_tts_cache_index.v1",
        "cache_name": build_manifest_cache_name(cache_name),
        "remote_cache_prefix": build_remote_cache_prefix(cache_name),
        "source_file": manifest.get("source_file"),
        "created_at": time.time(),
        "segments_total": len(manifest.get("segments") or []),
        "cache_ready": hits,
        "cache_missing_or_stale": missing,
        "failed": failed,
        "items": items,
    }
    return index


def write_manifest_cache_index(cache_name, manifest, manifest_cache_dir, defaults=None):
    index = build_manifest_cache_index(cache_name, manifest, manifest_cache_dir, defaults=defaults)
    index_path = build_cache_index_path(manifest_cache_dir)
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    return index


def summarize_cache_index(index):
    if not isinstance(index, dict):
        return {"segments_total": 0, "cache_ready": 0, "cache_missing_or_stale": 0, "failed": 0}
    return {
        "segments_total": int(index.get("segments_total", 0) or 0),
        "cache_ready": int(index.get("cache_ready", 0) or 0),
        "cache_missing_or_stale": int(index.get("cache_missing_or_stale", 0) or 0),
        "failed": int(index.get("failed", 0) or 0),
        "index_file": CACHE_INDEX_FILENAME,
    }


def build_segment_cache_stem(segment):
    order = int(segment.get("order", 0) or 0)
    segment_id = sanitize_filename(str(segment.get("segment_id", "")).strip() or f"seg_{order:06d}")
    return f"segment_{order:06d}__{segment_id}"


def get_segment_cache_paths(manifest_cache_dir, segment):
    stem = build_segment_cache_stem(segment)
    return {
        "audio": manifest_cache_dir / f"{stem}.mp3",
        "meta": manifest_cache_dir / f"{stem}.json",
    }


def get_segment_failed_path(manifest_cache_dir, segment):
    return manifest_cache_dir / f"{build_segment_cache_stem(segment)}.failed.json"



def clear_manifest_local_cache_state(manifest_filename):
    """Clear local cache folders for this manifest before a new run.

    This prevents stale files when the user reruns a manifest with the same
    filename but slightly different text. Remote Dataset cache is not touched
    here; force-rerender remote clearing is handled separately.
    """
    cache_name = build_manifest_cache_name(manifest_filename)
    removed = []
    cache_dir = CHUNK_CACHE_ROOT / cache_name
    if remove_path_safely(cache_dir):
        removed.append(str(cache_dir))
    pending_cache_dir = PENDING_UPLOAD_ROOT / build_remote_cache_prefix(cache_name)
    if remove_path_safely(pending_cache_dir):
        removed.append(str(pending_cache_dir))
    return removed


def _cache_file_base_name(path):
    name = Path(path).name
    if name == CACHE_INDEX_FILENAME:
        return None
    if name.endswith(".failed.json"):
        return name[: -len(".failed.json")]
    if name.endswith(".json"):
        return name[: -len(".json")]
    if name.endswith(".mp3"):
        return name[: -len(".mp3")]
    return None


def prune_manifest_cache_to_current_manifest(manifest_cache_dir, ordered_segments, defaults):
    """Remove stale local cache files that do not match the current manifest.

    Folder-level repo refresh is fast, but a rerun can use the same filename with
    slightly changed text. Segment filenames may be identical while cache keys are
    different. This cleanup keeps only cache entries whose meta cache_key matches
    the current segment text/voice/rate/pitch, and deletes old files before manual
    snapshot commits can pick them up.
    """
    cache_dir = Path(manifest_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    expected_by_stem = {}
    for segment in ordered_segments or []:
        profile = resolve_segment_tts_settings(segment, defaults)
        expected_by_stem[build_segment_cache_stem(segment)] = {
            "segment": segment,
            "profile": profile,
            "cache_key": make_segment_cache_key(segment, profile),
        }

    removed_files = []
    stale_segments = 0
    orphan_files = 0
    incomplete_pairs = 0
    failed_removed = 0

    def _remove(path, reason):
        nonlocal orphan_files, incomplete_pairs, failed_removed
        try:
            path = Path(path)
            if path.exists() and path.is_file():
                path.unlink(missing_ok=True)
                removed_files.append({"file": path.name, "reason": reason})
                if reason == "orphan":
                    orphan_files += 1
                elif reason == "incomplete_pair":
                    incomplete_pairs += 1
                elif reason == "stale_failed":
                    failed_removed += 1
                return True
        except Exception:
            pass
        return False

    # Remove orphan files from older manifests / renamed segments.
    for path in list(cache_dir.iterdir()):
        if not path.is_file():
            continue
        if path.name == CACHE_INDEX_FILENAME:
            path.unlink(missing_ok=True)
            removed_files.append({"file": path.name, "reason": "rebuild_index"})
            continue
        base = _cache_file_base_name(path)
        if base and base not in expected_by_stem:
            _remove(path, "orphan")

    # Remove stale cache pairs for segments whose text/settings changed.
    for stem, info in expected_by_stem.items():
        segment = info["segment"]
        expected_key = info["cache_key"]
        paths = get_segment_cache_paths(cache_dir, segment)
        failed_path = get_segment_failed_path(cache_dir, segment)

        audio_exists = paths["audio"].exists()
        meta_exists = paths["meta"].exists()
        if audio_exists and meta_exists:
            try:
                with open(paths["meta"], "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if meta.get("cache_key") != expected_key:
                    stale_segments += 1
                    _remove(paths["audio"], "stale_cache_key")
                    _remove(paths["meta"], "stale_cache_key")
                    if failed_path.exists():
                        _remove(failed_path, "stale_failed")
            except Exception:
                stale_segments += 1
                _remove(paths["audio"], "bad_meta")
                _remove(paths["meta"], "bad_meta")
        elif audio_exists != meta_exists:
            _remove(paths["audio"], "incomplete_pair")
            _remove(paths["meta"], "incomplete_pair")

        # Failed markers are also tied to the exact current text/settings.
        if failed_path.exists():
            try:
                with open(failed_path, "r", encoding="utf-8") as f:
                    failed_meta = json.load(f)
                profile = info["profile"]
                same_failed = (
                    clean_tts_text(failed_meta.get("text", "")) == clean_tts_text(segment.get("text", ""))
                    and str(failed_meta.get("voice")) == str(profile.get("voice"))
                    and int(failed_meta.get("rate_pct", 0) or 0) == int(profile.get("rate_pct", 0) or 0)
                    and int(failed_meta.get("pitch_hz", 0) or 0) == int(profile.get("pitch_hz", 0) or 0)
                )
                if not same_failed:
                    _remove(failed_path, "stale_failed")
            except Exception:
                _remove(failed_path, "stale_failed")

    return {
        "enabled": True,
        "removed_files": len(removed_files),
        "stale_segments": stale_segments,
        "orphan_files": orphan_files,
        "incomplete_pairs": incomplete_pairs,
        "failed_removed": failed_removed,
        "sample_removed": removed_files[:20],
    }


def load_failed_segment(manifest_cache_dir, segment, cache_name=None):
    failed_path = get_segment_failed_path(manifest_cache_dir, segment)
    if not failed_path.exists() and cache_name:
        download_chunk_cache_file(cache_name, failed_path.name, str(failed_path))
    if not failed_path.exists():
        return None
    try:
        with open(failed_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None



def atomic_write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + f".tmp.{os.getpid()}.{time.time_ns()}")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        try:
            f.flush()
            os.fsync(f.fileno())
        except Exception:
            pass
    os.replace(tmp_path, path)
    return str(path)


def atomic_copy_file(src, dst):
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst.with_name(dst.name + f".tmp.{os.getpid()}.{time.time_ns()}")
    shutil.copyfile(str(src), str(tmp_path))
    os.replace(tmp_path, dst)
    return str(dst)


def save_failed_segment(manifest_cache_dir, segment, profile, exc, failure_stage="normal_render"):
    failed_path = get_segment_failed_path(manifest_cache_dir, segment)
    failure_class, error_type = classify_tts_failure(exc, segment)
    data = {
        "segment_id": segment.get("segment_id"),
        "order": int(segment.get("order", 0) or 0),
        "status": "failed",
        "failure_stage": failure_stage,
        "failure_class": failure_class,
        "error_type": error_type,
        "error_message": str(exc),
        "text": clean_tts_text(segment.get("text", "")),
        "voice": profile.get("voice"),
        "rate_pct": int(profile.get("rate_pct", 0) or 0),
        "pitch_hz": int(profile.get("pitch_hz", 0) or 0),
        "failed_at": time.time(),
    }
    atomic_write_json(failed_path, data)
    return data


def clear_failed_segment(manifest_cache_dir, segment):
    failed_path = get_segment_failed_path(manifest_cache_dir, segment)
    if failed_path.exists():
        failed_path.unlink(missing_ok=True)



def load_cached_segment(manifest_cache_dir, segment, cache_key, cache_name=None, allow_remote_download=True):
    paths = get_segment_cache_paths(manifest_cache_dir, segment)
    if allow_remote_download and (not paths["audio"].exists() or not paths["meta"].exists()) and cache_name:
        download_chunk_cache_file(cache_name, paths["audio"].name, str(paths["audio"]))
        download_chunk_cache_file(cache_name, paths["meta"].name, str(paths["meta"]))
    if not paths["audio"].exists() or not paths["meta"].exists():
        return None
    with open(paths["meta"], "r", encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("cache_key") != cache_key or not meta.get("events"):
        return None
    actual_duration = get_audio_duration_sec(str(paths["audio"]))
    meta_duration = float(meta.get("audio_duration_sec", 0) or 0)
    events_duration = max((float(event.get("end", 0) or 0) for event in meta.get("events") or []), default=0.0)
    duration = max(float(actual_duration or 0), meta_duration, events_duration)
    return {
        "audio_path": str(paths["audio"]),
        "events": meta["events"],
        "audio_duration_sec": round(duration, 3),
    }


def save_cached_segment(manifest_cache_dir, segment, cache_key, rendered, cache_name=None, upload_remote=False):
    paths = get_segment_cache_paths(manifest_cache_dir, segment)
    # Atomic cache write: copy to .tmp first, then os.replace. Snapshot/commit will never pick up half-written files.
    atomic_copy_file(rendered["audio_path"], paths["audio"])
    meta_data = {
        "cache_key": cache_key,
        "segment_id": segment.get("segment_id"),
        "order": int(segment.get("order", 0) or 0),
        "voice": rendered["profile"]["voice"],
        "events": rendered["events"],
        "audio_duration_sec": rendered["audio_duration_sec"],
        "status": rendered.get("status", "success"),
        "rescue_method": rendered.get("rescue_method"),
        "split_method": rendered.get("split_method"),
        "atomic_cache_write": True,
        "saved_at": time.time(),
    }
    atomic_write_json(paths["meta"], meta_data)
    clear_failed_segment(manifest_cache_dir, segment)
    if upload_remote and cache_name:
        upload_chunk_cache_file(str(paths["audio"]), cache_name, paths["audio"].name)
        upload_chunk_cache_file(str(paths["meta"]), cache_name, paths["meta"].name)


async def render_segment(segment, output_path, defaults, manifest_cache_dir=None, cache_name=None, upload_remote_cache=False, tts_timeout_sec=None, force_rerender=False, repo_cache_only=False, tts_semaphore=None):
    profile = resolve_segment_tts_settings(segment, defaults)
    cleaned_text = clean_tts_text(segment.get("text", ""))
    if not cleaned_text:
        raise ValueError(f"Segment text is empty after cleanup: {segment_debug_label(segment, profile)}")
    cache_key = make_segment_cache_key(segment, profile)
    cached = None if force_rerender else (load_cached_segment(manifest_cache_dir, segment, cache_key, cache_name=cache_name, allow_remote_download=not repo_cache_only) if manifest_cache_dir else None)
    if cached:
        try:
            if os.path.exists(output_path):
                os.remove(output_path)
            os.link(cached["audio_path"], output_path)
        except Exception:
            shutil.copyfile(cached["audio_path"], output_path)
        return {
            "cache_hit": True,
            "audio_path": output_path,
            "events": cached["events"],
            "audio_duration_sec": cached["audio_duration_sec"],
            "profile": profile,
        }
    # Use the same UI-configured TTS timeout for every segment, including very short text.
    # Short segments should not silently use a different timeout because it makes retry
    # behavior confusing and inconsistent with the slider.
    effective_timeout_sec = tts_timeout_sec
    try:
        if tts_semaphore is not None:
            async with tts_semaphore:
                raw_edge_events = await synthesize_audio(
                    text=cleaned_text,
                    voice=profile["voice"],
                    rate=profile["rate_pct"],
                    pitch=profile["pitch_hz"],
                    audio_path=output_path,
                    timeout_sec=effective_timeout_sec,
                    emotion=profile.get("emotion"),
                )
        else:
            raw_edge_events = await synthesize_audio(
                text=cleaned_text,
                voice=profile["voice"],
                rate=profile["rate_pct"],
                pitch=profile["pitch_hz"],
                audio_path=output_path,
                timeout_sec=effective_timeout_sec,
                emotion=profile.get("emotion"),
            )
    except Exception as exc:
        if tts_semaphore is not None and hasattr(tts_semaphore, "note_failure"):
            tts_semaphore.note_failure(str(exc)[:120])
        raise RuntimeError(
            "TTS render failed: "
            f"{segment_debug_label(segment, profile)}"
            + f" | timeout_sec={effective_timeout_sec}"
            + f" | last_error={exc}"
        ) from exc
    if tts_semaphore is not None and hasattr(tts_semaphore, "note_success"):
        tts_semaphore.note_success()
    raw_edge_events = ensure_event_fallback(raw_edge_events, cleaned_text, profile["rate_pct"])
    event_duration_sec = max(event["end"] for event in raw_edge_events)
    actual_duration_sec = get_audio_duration_sec(output_path)
    duration_sec = round(max(float(actual_duration_sec or 0), float(event_duration_sec or 0)), 3)
    events = build_segment_subtitle_events_from_json_text(raw_edge_events, cleaned_text, duration_sec, profile["rate_pct"])
    rendered = {
        "cache_hit": False,
        "audio_path": output_path,
        "events": events,
        "audio_duration_sec": duration_sec,
        "profile": profile,
    }
    if manifest_cache_dir:
        save_cached_segment(
            manifest_cache_dir,
            segment,
            cache_key,
            rendered,
            cache_name=cache_name,
            upload_remote=upload_remote_cache,
        )
    return rendered


def build_plain_output_names(original_name):
    base = sanitize_filename(original_name)
    if base.endswith("-cv"):
        base = base[:-3] + "-audio"
    else:
        base = f"{base}_audio"
    return base, f"{base}.mp3"


async def generate_plain_single_voice(text, voice, rate, pitch, audio_path, tts_timeout_sec=None):
    events = await synthesize_audio(text, voice, rate, pitch, audio_path, timeout_sec=tts_timeout_sec)
    return ensure_event_fallback(events, text, rate)


async def process_plain_text_file(file, voice, rate, pitch, subtitle_format, upload=True, work_dir=None, tts_timeout_sec=None):
    text = read_uploaded_text_file(file)
    original_name = os.path.basename(file.name)
    story_prefix = extract_story_prefix(original_name)
    base, audio_name = build_plain_output_names(original_name)
    subtitle_name = None if subtitle_format == "no_script" else f"{base}.{subtitle_format}"
    report_name = f"{base}_render_report.json"
    cleanup_dir = None
    tmpdir = work_dir
    if tmpdir is None:
        cleanup_dir = tempfile.TemporaryDirectory()
        tmpdir = cleanup_dir.name
    else:
        os.makedirs(tmpdir, exist_ok=True)
    try:
        local_audio = os.path.join(tmpdir, audio_name)
        events = await generate_plain_single_voice(text, voice, rate, pitch, local_audio, tts_timeout_sec=tts_timeout_sec)
        if subtitle_name:
            local_subtitle = os.path.join(tmpdir, subtitle_name)
            write_subtitle_file(events, local_subtitle, subtitle_format)
        else:
            local_subtitle = None
        local_report = os.path.join(tmpdir, report_name)
        report = {
            "mode": DEFAULT_PLAIN_MODE,
            "source_file": original_name,
            "story_prefix": story_prefix,
            "output_subdir": build_output_subdir(original_name),
            "audio_name": audio_name,
            "subtitle_name": subtitle_name,
            "segments_total": 1,
            "subtitle_events_total": len(events),
            "voice": voice,
            "rate_pct": int(rate),
            "pitch_hz": int(pitch),
            "uploaded": bool(upload),
        }
        write_render_report(local_report, report)
        output_files = [local_audio]
        upload_entries = [(local_audio, f"outputs/{story_prefix}/{audio_name}")]
        if local_subtitle:
            output_files.append(local_subtitle)
            upload_entries.append((local_subtitle, f"outputs/{story_prefix}/{subtitle_name}"))
        output_files.append(local_report)
        upload_entries.append((local_report, f"outputs/{story_prefix}/{report_name}"))
        prefix = f"Rendered locally for upload staging ({story_prefix})" if upload else f"Rendered locally for download ({story_prefix})"
        return {
            "message": f"{prefix}: {audio_name}" + (
                f", {subtitle_name}, {report_name}" if subtitle_name else f", {report_name}"
            ),
            "files": output_files,
            "upload_entries": upload_entries,
        }
    finally:
        if cleanup_dir is not None:
            cleanup_dir.cleanup()

def remove_path_safely(path):
    try:
        path = Path(path)
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
            return True
        if path.is_file():
            path.unlink(missing_ok=True)
            return True
    except Exception:
        return False
    return False


def clear_manifest_rerender_state(manifest_filename):
    """Delete local cache and pending upload files for one manifest so it renders from zero."""
    cache_name = build_manifest_cache_name(manifest_filename)
    removed = []

    cache_dir = CHUNK_CACHE_ROOT / cache_name
    if remove_path_safely(cache_dir):
        removed.append(str(cache_dir))

    pending_cache_dir = PENDING_UPLOAD_ROOT / build_remote_cache_prefix(cache_name)
    if remove_path_safely(pending_cache_dir):
        removed.append(str(pending_cache_dir))

    # Remove pending final outputs for this source file. Existing committed final outputs on HF are not deleted here;
    # the new successful commit will overwrite files with the same remote path. Remote chunkcache is cleared
    # once per batch in batch_tts() when Force rerender + Upload are enabled.
    story_prefix = extract_story_prefix(manifest_filename)
    base = sanitize_filename(manifest_filename) + "_audio"
    pending_output_dir = PENDING_UPLOAD_ROOT / "outputs" / story_prefix
    for suffix in (".mp3", ".srt", ".vtt", "_render_report.json"):
        candidate = pending_output_dir / f"{base}{suffix}"
        if remove_path_safely(candidate):
            removed.append(str(candidate))

    return removed



def get_silence_intervals_sec(audio_path):
    ffmpeg = find_ffmpeg()
    if not ffmpeg or not audio_path or not os.path.exists(audio_path):
        return []
    try:
        completed = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-nostats",
                "-i",
                audio_path,
                "-af",
                f"silencedetect=n={SILENCE_DETECT_NOISE_DB}dB:d={SILENCE_DETECT_MIN_DURATION_SEC}",
                "-f",
                "null",
                "-",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        log = (completed.stderr or "") + "\n" + (completed.stdout or "")
        starts = [float(x) for x in re.findall(r"silence_start:\s*([0-9.]+)", log)]
        ends = [float(x) for x in re.findall(r"silence_end:\s*([0-9.]+)", log)]
        duration = float(get_audio_duration_sec(audio_path) or 0)
        intervals = []
        for idx, st in enumerate(starts):
            en = ends[idx] if idx < len(ends) else duration
            if en > st:
                intervals.append((st, en))
        return intervals
    except Exception:
        return []


def snap_boundary_to_nearest_silence(boundary, silence_intervals, window=RESCUE_BOUNDARY_SNAP_WINDOW_SEC):
    best = None
    best_dist = None
    for st, en in silence_intervals or []:
        mid = (st + en) / 2.0
        dist = abs(mid - boundary)
        if dist <= window and (best_dist is None or dist < best_dist):
            best = mid
            best_dist = dist
    return float(best if best is not None else boundary)


def weighted_split_bounds(segments, duration_sec):
    weights = []
    for seg in segments:
        text = clean_tts_text(seg.get("text", ""))
        # Short Vietnamese particles need a minimum weight or they get cut too aggressively.
        weights.append(max(4, len(text)))
    total = float(sum(weights) or len(segments) or 1)
    cursor = 0.0
    bounds = []
    for idx, weight in enumerate(weights):
        start = cursor
        if idx == len(weights) - 1:
            end = float(duration_sec)
        else:
            end = cursor + float(duration_sec) * (weight / total)
        bounds.append((start, end))
        cursor = end
    return bounds


def compute_rescue_split_bounds(raw_edge_events, segments, audio_path, audio_duration_sec):
    duration = float(audio_duration_sec or get_audio_duration_sec(audio_path) or 0)
    if duration <= 0:
        duration = estimate_duration_from_text(" ".join(clean_tts_text(s.get("text", "")) for s in segments), 10)
    edge_events = sorted(raw_edge_events or [], key=lambda item: float(item.get("start", 0) or 0))
    if len(edge_events) >= len(segments):
        selected = edge_events[: len(segments)]
        bounds = []
        for item in selected:
            st = max(0.0, float(item.get("start", 0) or 0))
            en = min(duration, max(st + 0.08, float(item.get("end", st) or st)))
            bounds.append((st, en))
    else:
        bounds = weighted_split_bounds(segments, duration)

    silences = get_silence_intervals_sec(audio_path)
    refined = []
    for idx, (st, en) in enumerate(bounds):
        if idx > 0:
            st = snap_boundary_to_nearest_silence(st, silences)
        if idx < len(bounds) - 1:
            en = snap_boundary_to_nearest_silence(en, silences)
        st = max(0.0, st - RESCUE_CUT_PAD_BEFORE_SEC)
        en = min(duration, en + RESCUE_CUT_PAD_AFTER_SEC)
        if refined and st < refined[-1][1]:
            st = refined[-1][1]
        if en <= st:
            en = min(duration, st + 0.12)
        refined.append((round(st, 3), round(en, 3)))
    return refined


def cut_audio_clip(input_path, output_path, start_sec, end_sec):
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to cut rescue audio clips.")
    duration = max(0.08, float(end_sec) - float(start_sec))
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-ss",
            f"{float(start_sec):.3f}",
            "-i",
            input_path,
            "-t",
            f"{duration:.3f}",
            "-vn",
            "-acodec",
            "libmp3lame",
            "-q:a",
            "4",
            output_path,
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return output_path


async def render_rescue_group(group_segments, profile, tmpdir, group_label, tts_timeout_sec=None, tts_semaphore=None):
    # Blank lines encourage Edge TTS to leave a tiny pause between very short segments.
    group_text = "\n\n".join(clean_tts_text(seg.get("text", "")) for seg in group_segments if clean_tts_text(seg.get("text", "")))
    group_audio = os.path.join(tmpdir, f"{group_label}.mp3")
    if tts_semaphore is not None:
        async with tts_semaphore:
            raw_events = await synthesize_audio_once(
                group_text,
                profile["voice"],
                profile["rate_pct"],
                profile["pitch_hz"],
                group_audio,
                timeout_sec=tts_timeout_sec,
                emotion=profile.get("emotion"),
            )
    else:
        raw_events = await synthesize_audio_once(
            group_text,
            profile["voice"],
            profile["rate_pct"],
            profile["pitch_hz"],
            group_audio,
            timeout_sec=tts_timeout_sec,
            emotion=profile.get("emotion"),
        )
    raw_events = ensure_event_fallback(raw_events, group_text, profile["rate_pct"])
    group_duration = get_audio_duration_sec(group_audio)
    if not group_duration:
        group_duration = max((float(event.get("end", 0) or 0) for event in raw_events), default=0.0)
    bounds = compute_rescue_split_bounds(raw_events, group_segments, group_audio, group_duration)
    return group_audio, bounds


async def rescue_failed_segments(
    candidates,
    ordered_segments,
    rendered_by_segment_id,
    defaults,
    manifest_cache_dir,
    tmpdir,
    cache_name=None,
    tts_timeout_sec=None,
    tts_semaphore=None,
):
    rescued = []
    rescue_failed = []
    if not candidates:
        return rescued, rescue_failed

    by_id = {str(seg.get("segment_id")): seg for seg in ordered_segments}
    index_by_id = {str(seg.get("segment_id")): idx for idx, seg in enumerate(ordered_segments)}
    remaining = {str(seg.get("segment_id")): seg for seg in candidates}

    async def save_rescued_clip(seg, input_audio, bounds, group_segments, method, group_label):
        seg_id = str(seg.get("segment_id"))
        order = int(seg.get("order", index_by_id.get(seg_id, 0) + 1) or 0)
        output_path = os.path.join(tmpdir, f"segment_{order:06d}_rescued.mp3")
        cut_audio_clip(input_audio, output_path, bounds[0], bounds[1])
        duration = float(get_audio_duration_sec(output_path) or max(0.08, bounds[1] - bounds[0]))
        profile = resolve_segment_tts_settings(seg, defaults)
        rendered = {
            "cache_hit": False,
            "audio_path": output_path,
            "events": [{"start": 0.0, "end": round(duration, 3), "text": clean_tts_text(seg.get("text", ""))}],
            "audio_duration_sec": round(duration, 3),
            "profile": profile,
            "status": "rescued",
            "rescue_method": method,
            "split_method": "edge_boundary_hint_plus_silence_snap_plus_weighted_fallback",
        }
        cache_key = make_segment_cache_key(seg, profile)
        save_cached_segment(manifest_cache_dir, seg, cache_key, rendered, cache_name=cache_name, upload_remote=False)
        rendered_by_segment_id[seg_id] = rendered
        rescued.append(
            {
                "segment_id": seg_id,
                "order": order,
                "text": clean_tts_text(seg.get("text", "")),
                "voice": profile["voice"],
                "rate_pct": profile["rate_pct"],
                "pitch_hz": profile["pitch_hz"],
                "rescue_method": method,
                "split_method": rendered["split_method"],
                "source_group_segments": [s.get("segment_id") for s in group_segments],
                "group_label": group_label,
            }
        )

    # Stage 1: neighbor rescue for adjacent same voice/rate/pitch.
    for seg_id in list(remaining.keys()):
        seg = remaining.get(seg_id)
        if not seg:
            continue
        idx = index_by_id.get(seg_id)
        if idx is None:
            continue
        candidates_neighbors = []
        for nidx in (idx - 1, idx + 1):
            if 0 <= nidx < len(ordered_segments):
                neighbor = ordered_segments[nidx]
                if segment_setting_key(neighbor, defaults) == segment_setting_key(seg, defaults):
                    candidates_neighbors.append(neighbor)
        if not candidates_neighbors:
            continue
        neighbor = candidates_neighbors[0]
        group_segments = sorted([seg, neighbor], key=lambda item: int(item.get("order", 0) or 0))
        profile = resolve_segment_tts_settings(seg, defaults)
        group_label = f"rescue_neighbor_{sanitize_filename(seg_id)}"
        try:
            group_audio, bounds = await render_rescue_group(group_segments, profile, tmpdir, group_label, tts_timeout_sec=tts_timeout_sec, tts_semaphore=tts_semaphore)
            seg_pos = [str(s.get("segment_id")) for s in group_segments].index(seg_id)
            await save_rescued_clip(seg, group_audio, bounds[seg_pos], group_segments, "neighbor_group", group_label)
            remaining.pop(seg_id, None)
        except Exception as exc:
            rescue_failed.append({"segment_id": seg_id, "order": seg.get("order"), "rescue_method": "neighbor_group", "error": str(exc)})

    # Stage 2: carrier rescue for remaining repeated short/no-audio failures with same setting.
    groups = {}
    for seg in remaining.values():
        groups.setdefault(segment_setting_key(seg, defaults), []).append(seg)
    for key, segs in groups.items():
        segs = sorted(segs, key=lambda item: int(item.get("order", 0) or 0))
        chunk = []
        chunk_chars = 0
        chunk_index = 0
        async def flush_chunk(chunk_to_render, chunk_idx):
            if not chunk_to_render:
                return
            profile = resolve_segment_tts_settings(chunk_to_render[0], defaults)
            group_label = f"rescue_carrier_{sanitize_filename(profile['voice'])}_{chunk_idx:04d}"
            try:
                group_audio, bounds = await render_rescue_group(chunk_to_render, profile, tmpdir, group_label, tts_timeout_sec=tts_timeout_sec, tts_semaphore=tts_semaphore)
                for local_idx, seg in enumerate(chunk_to_render):
                    await save_rescued_clip(seg, group_audio, bounds[local_idx], chunk_to_render, "carrier_group", group_label)
                    remaining.pop(str(seg.get("segment_id")), None)
            except Exception as exc:
                for seg in chunk_to_render:
                    rescue_failed.append({"segment_id": seg.get("segment_id"), "order": seg.get("order"), "rescue_method": "carrier_group", "error": str(exc)})

        for seg in segs:
            text_len = len(clean_tts_text(seg.get("text", "")))
            if chunk and (len(chunk) >= RESCUE_CARRIER_GROUP_MAX_SEGMENTS or chunk_chars + text_len > RESCUE_CARRIER_GROUP_MAX_CHARS):
                chunk_index += 1
                await flush_chunk(chunk, chunk_index)
                chunk = []
                chunk_chars = 0
            chunk.append(seg)
            chunk_chars += text_len
        if chunk:
            chunk_index += 1
            await flush_chunk(chunk, chunk_index)

    return rescued, rescue_failed


# ==============================================================================
# Split cache layout v2: success cache and failed cache are stored separately.
#
# New remote/local structure:
#   chunkcache/<manifest>/success/segment_*.mp3
#   chunkcache/<manifest>/success/segment_*.json
#   chunkcache/<manifest>/failed/segment_*.failed.json
#   chunkcache/<manifest>/_cache_index.json
#
# The loaders remain backward-compatible with the old flat layout so old repo cache
# can still be reused. New writes always go to success/ or failed/.
# ==============================================================================

def get_segment_success_dir(manifest_cache_dir):
    path = Path(manifest_cache_dir) / SUCCESS_CACHE_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_segment_failed_dir(manifest_cache_dir):
    path = Path(manifest_cache_dir) / FAILED_CACHE_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_segment_cache_paths(manifest_cache_dir, segment):
    stem = build_segment_cache_stem(segment)
    success_dir = get_segment_success_dir(manifest_cache_dir)
    return {
        "audio": success_dir / f"{stem}.mp3",
        "meta": success_dir / f"{stem}.json",
    }


def get_legacy_segment_cache_paths(manifest_cache_dir, segment):
    stem = build_segment_cache_stem(segment)
    cache_dir = Path(manifest_cache_dir)
    return {
        "audio": cache_dir / f"{stem}.mp3",
        "meta": cache_dir / f"{stem}.json",
    }


def get_all_segment_cache_path_pairs(manifest_cache_dir, segment):
    return [get_segment_cache_paths(manifest_cache_dir, segment), get_legacy_segment_cache_paths(manifest_cache_dir, segment)]


def get_segment_failed_path(manifest_cache_dir, segment):
    return get_segment_failed_dir(manifest_cache_dir) / f"{build_segment_cache_stem(segment)}.failed.json"


def get_legacy_segment_failed_path(manifest_cache_dir, segment):
    return Path(manifest_cache_dir) / f"{build_segment_cache_stem(segment)}.failed.json"


def get_all_segment_failed_paths(manifest_cache_dir, segment):
    return [get_segment_failed_path(manifest_cache_dir, segment), get_legacy_segment_failed_path(manifest_cache_dir, segment)]


def _rel_cache_path(base_dir, path):
    try:
        return Path(path).relative_to(Path(base_dir)).as_posix()
    except Exception:
        return Path(path).name


def _remote_cache_path_matches_scope(remote_path, remote_prefix, cache_scope):
    remote_prefix = str(remote_prefix).strip("/")
    if not str(remote_path).startswith(remote_prefix + "/"):
        return False
    rel = str(remote_path)[len(remote_prefix) + 1:]
    if not rel:
        return False
    if rel == CACHE_INDEX_FILENAME:
        return True
    if cache_scope == CACHE_SCOPE_FAILED_ONLY:
        return rel.startswith(FAILED_CACHE_DIRNAME + "/") or ("/" not in rel and rel.endswith(".failed.json"))
    if cache_scope == CACHE_SCOPE_SUCCESS_ONLY:
        if rel.startswith(SUCCESS_CACHE_DIRNAME + "/") or rel.startswith(FINAL_PARTS_DIRNAME + "/"):
            return True
        # Backward-compatible legacy flat success cache. Do not include legacy failed markers.
        return "/" not in rel and (rel.endswith(".mp3") or (rel.endswith(".json") and not rel.endswith(".failed.json") and rel != CACHE_INDEX_FILENAME))
    return True


def sync_remote_cache_folder_to_local(cache_name, clear_local_first=True, cache_scope=CACHE_SCOPE_ALL):
    """Refresh cache from repo with scope control.

    cache_scope="failed_only" downloads only failed markers plus index. This is used at
    the start of a normal run so failed segments can be retried before downloading
    thousands of success audio/meta files. Only after failed markers are resolved do
    we download cache_scope="success_only" for final build/cache hits.
    """
    cache_name = build_manifest_cache_name(cache_name)
    cache_scope = str(cache_scope or CACHE_SCOPE_ALL)
    if cache_scope not in {CACHE_SCOPE_ALL, CACHE_SCOPE_FAILED_ONLY, CACHE_SCOPE_SUCCESS_ONLY}:
        cache_scope = CACHE_SCOPE_ALL
    remote_prefix = build_remote_cache_prefix(cache_name).strip("/")
    local_dir = build_manifest_cache_dir(cache_name)
    if clear_local_first and local_dir.exists():
        shutil.rmtree(local_dir, ignore_errors=True)
    local_dir.mkdir(parents=True, exist_ok=True)
    # Ensure split folders exist so the user can inspect local structure even before writes.
    get_segment_success_dir(local_dir)
    get_segment_failed_dir(local_dir)

    if api is None:
        return {"enabled": True, "status": "skipped", "reason": "huggingface_hub is unavailable", "files": 0, "remote_prefix": remote_prefix, "local_dir": str(local_dir.resolve()), "cache_scope": cache_scope}
    ensure_dataset_exists()
    try:
        repo_files = api.list_repo_files(repo_id=DATASET_REPO, repo_type="dataset", token=HF_TOKEN)
    except Exception as exc:
        return {"enabled": True, "status": "failed", "reason": f"list_repo_files failed: {exc}", "files": 0, "remote_prefix": remote_prefix, "local_dir": str(local_dir.resolve()), "cache_scope": cache_scope}

    matched = [path for path in repo_files if _remote_cache_path_matches_scope(path, remote_prefix, cache_scope)]
    if not matched:
        return {"enabled": True, "status": "empty", "reason": "no remote cache files matched scope", "files": 0, "remote_prefix": remote_prefix, "local_dir": str(local_dir.resolve()), "cache_scope": cache_scope}

    copied = 0
    matched_set = set(matched)
    if snapshot_download is not None:
        try:
            if cache_scope == CACHE_SCOPE_FAILED_ONLY:
                allow_patterns = [remote_prefix + f"/{FAILED_CACHE_DIRNAME}/**", remote_prefix + "/*.failed.json", remote_prefix + f"/{CACHE_INDEX_FILENAME}"]
            elif cache_scope == CACHE_SCOPE_SUCCESS_ONLY:
                allow_patterns = [remote_prefix + f"/{SUCCESS_CACHE_DIRNAME}/**", remote_prefix + f"/{FINAL_PARTS_DIRNAME}/**", remote_prefix + "/*.mp3", remote_prefix + "/*.json", remote_prefix + f"/{CACHE_INDEX_FILENAME}"]
            else:
                allow_patterns = [remote_prefix + "/**", remote_prefix + "/*"]
            snapshot_dir = snapshot_download(repo_id=DATASET_REPO, repo_type="dataset", allow_patterns=allow_patterns, token=HF_TOKEN)
            source_dir = Path(snapshot_dir) / remote_prefix
            if source_dir.exists():
                for src in source_dir.rglob("*"):
                    if not src.is_file():
                        continue
                    rel = src.relative_to(source_dir).as_posix()
                    remote_path = f"{remote_prefix}/{rel}"
                    if remote_path not in matched_set:
                        continue
                    dest = local_dir / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src, dest)
                    copied += 1
                return {"enabled": True, "status": "synced", "method": "snapshot_download_folder", "files": copied, "remote_prefix": remote_prefix, "local_dir": str(local_dir.resolve()), "cache_scope": cache_scope}
        except Exception:
            copied = 0

    if hf_hub_download is None:
        return {"enabled": True, "status": "failed", "reason": "snapshot_download and hf_hub_download are unavailable", "files": 0, "remote_prefix": remote_prefix, "local_dir": str(local_dir.resolve()), "cache_scope": cache_scope}
    errors = []
    for remote_path in matched:
        try:
            downloaded = hf_hub_download(repo_id=DATASET_REPO, repo_type="dataset", filename=remote_path, token=HF_TOKEN)
            rel = Path(remote_path).relative_to(remote_prefix)
            dest = local_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(downloaded, dest)
            copied += 1
        except Exception as exc:
            if len(errors) < 5:
                errors.append(f"{remote_path}: {exc}")
    return {"enabled": True, "status": "synced" if copied else "failed", "method": "hf_hub_download_fallback", "files": copied, "errors": errors, "remote_prefix": remote_prefix, "local_dir": str(local_dir.resolve()), "cache_scope": cache_scope}


def build_manifest_cache_index(cache_name, manifest, manifest_cache_dir, defaults=None):
    defaults = defaults or DEFAULT_MANIFEST_STRATEGY
    cache_dir = Path(manifest_cache_dir)
    items = {}
    hits = 0
    missing = 0
    failed = 0
    stale = 0
    bad_meta = 0
    legacy_success = 0
    legacy_failed = 0
    for segment in sorted(manifest.get("segments") or [], key=lambda item: int(item.get("order", 0) or 0)):
        seg_id = str(segment.get("segment_id") or "")
        profile = resolve_segment_tts_settings(segment, defaults)
        cache_key = make_segment_cache_key(segment, profile)
        primary_paths = get_segment_cache_paths(cache_dir, segment)
        legacy_paths = get_legacy_segment_cache_paths(cache_dir, segment)
        failed_path = get_segment_failed_path(cache_dir, segment)
        legacy_failed_path = get_legacy_segment_failed_path(cache_dir, segment)
        paths = primary_paths if (primary_paths["audio"].exists() or primary_paths["meta"].exists()) else legacy_paths
        status = "missing"
        meta_data = {}
        layout = "split_success" if paths == primary_paths else "legacy_flat"
        if paths["audio"].exists() and paths["meta"].exists():
            try:
                with open(paths["meta"], "r", encoding="utf-8") as f:
                    meta_data = json.load(f)
                if meta_data.get("cache_key") == cache_key:
                    status = str(meta_data.get("status") or "success")
                    hits += 1
                    if layout == "legacy_flat":
                        legacy_success += 1
                else:
                    status = "stale"
                    stale += 1
                    missing += 1
            except Exception:
                status = "bad_meta"
                bad_meta += 1
                missing += 1
        elif failed_path.exists() or legacy_failed_path.exists():
            status = "failed"
            failed += 1
            if legacy_failed_path.exists() and not failed_path.exists():
                legacy_failed += 1
        else:
            missing += 1
        items[seg_id] = {
            "order": int(segment.get("order", 0) or 0),
            "status": status,
            "layout": layout if status not in {"failed", "missing"} else ("split_failed" if failed_path.exists() else ("legacy_failed" if legacy_failed_path.exists() else "none")),
            "audio": _rel_cache_path(cache_dir, paths["audio"]),
            "meta": _rel_cache_path(cache_dir, paths["meta"]),
            "failed": _rel_cache_path(cache_dir, failed_path if failed_path.exists() or not legacy_failed_path.exists() else legacy_failed_path),
            "cache_key": cache_key,
            "voice": profile.get("voice"),
            "rate_pct": int(profile.get("rate_pct", 0) or 0),
            "pitch_hz": int(profile.get("pitch_hz", 0) or 0),
            "audio_duration_sec": meta_data.get("audio_duration_sec"),
            "rescue_method": meta_data.get("rescue_method"),
        }
    index = {
        "schema": "vieneu_tts_cache_index.v2.split_success_failed",
        "cache_name": build_manifest_cache_name(cache_name),
        "remote_cache_prefix": build_remote_cache_prefix(cache_name),
        "source_file": manifest.get("source_file"),
        "created_at": time.time(),
        "cache_layout": {"success": SUCCESS_CACHE_DIRNAME, "failed": FAILED_CACHE_DIRNAME},
        "segments_total": len(manifest.get("segments") or []),
        "cache_ready": hits,
        "cache_missing_or_stale": missing,
        "failed": failed,
        "stale": stale,
        "bad_meta": bad_meta,
        "legacy_success": legacy_success,
        "legacy_failed": legacy_failed,
        "items": items,
    }
    return index


def write_manifest_cache_index(cache_name, manifest, manifest_cache_dir, defaults=None):
    index = build_manifest_cache_index(cache_name, manifest, manifest_cache_dir, defaults=defaults)
    index_path = build_cache_index_path(manifest_cache_dir)
    atomic_write_json(index_path, index)
    return index


def prune_manifest_cache_to_current_manifest(manifest_cache_dir, ordered_segments, defaults):
    cache_dir = Path(manifest_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    get_segment_success_dir(cache_dir)
    get_segment_failed_dir(cache_dir)
    expected_by_stem = {}
    for segment in ordered_segments or []:
        profile = resolve_segment_tts_settings(segment, defaults)
        expected_by_stem[build_segment_cache_stem(segment)] = {"segment": segment, "profile": profile, "cache_key": make_segment_cache_key(segment, profile)}

    removed_files = []
    stale_segments = 0
    orphan_files = 0
    incomplete_pairs = 0
    failed_removed = 0

    def _remove(path, reason):
        nonlocal orphan_files, incomplete_pairs, failed_removed
        try:
            path = Path(path)
            if path.exists() and path.is_file():
                rel = _rel_cache_path(cache_dir, path)
                path.unlink(missing_ok=True)
                removed_files.append({"file": rel, "reason": reason})
                if reason == "orphan": orphan_files += 1
                elif reason == "incomplete_pair": incomplete_pairs += 1
                elif reason == "stale_failed": failed_removed += 1
                return True
        except Exception:
            pass
        return False

    for path in list(cache_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name == CACHE_INDEX_FILENAME:
            path.unlink(missing_ok=True)
            removed_files.append({"file": _rel_cache_path(cache_dir, path), "reason": "rebuild_index"})
            continue
        if ".tmp." in path.name:
            _remove(path, "orphan")
            continue
        base = _cache_file_base_name(path)
        if base and base not in expected_by_stem:
            _remove(path, "orphan")

    for stem, info in expected_by_stem.items():
        segment = info["segment"]
        expected_key = info["cache_key"]
        failed_paths = get_all_segment_failed_paths(cache_dir, segment)
        for paths in get_all_segment_cache_path_pairs(cache_dir, segment):
            audio_exists = paths["audio"].exists()
            meta_exists = paths["meta"].exists()
            if audio_exists and meta_exists:
                try:
                    with open(paths["meta"], "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    if meta.get("cache_key") != expected_key:
                        stale_segments += 1
                        _remove(paths["audio"], "stale_cache_key")
                        _remove(paths["meta"], "stale_cache_key")
                        for failed_path in failed_paths:
                            if failed_path.exists():
                                _remove(failed_path, "stale_failed")
                except Exception:
                    stale_segments += 1
                    _remove(paths["audio"], "bad_meta")
                    _remove(paths["meta"], "bad_meta")
            elif audio_exists != meta_exists:
                _remove(paths["audio"], "incomplete_pair")
                _remove(paths["meta"], "incomplete_pair")

        for failed_path in failed_paths:
            if failed_path.exists():
                try:
                    with open(failed_path, "r", encoding="utf-8") as f:
                        failed_meta = json.load(f)
                    profile = info["profile"]
                    same_failed = (
                        clean_tts_text(failed_meta.get("text", "")) == clean_tts_text(segment.get("text", ""))
                        and str(failed_meta.get("voice")) == str(profile.get("voice"))
                        and int(failed_meta.get("rate_pct", 0) or 0) == int(profile.get("rate_pct", 0) or 0)
                        and int(failed_meta.get("pitch_hz", 0) or 0) == int(profile.get("pitch_hz", 0) or 0)
                    )
                    if not same_failed:
                        _remove(failed_path, "stale_failed")
                except Exception:
                    _remove(failed_path, "stale_failed")
    return {"enabled": True, "removed_files": len(removed_files), "stale_segments": stale_segments, "orphan_files": orphan_files, "incomplete_pairs": incomplete_pairs, "failed_removed": failed_removed, "sample_removed": removed_files[:20]}


def load_failed_segment(manifest_cache_dir, segment, cache_name=None):
    cache_dir = Path(manifest_cache_dir)
    paths = get_all_segment_failed_paths(cache_dir, segment)
    if cache_name:
        for failed_path in paths:
            if not failed_path.exists():
                try:
                    rel = _rel_cache_path(cache_dir, failed_path)
                    download_chunk_cache_file(cache_name, rel, str(failed_path))
                except Exception:
                    pass
    for failed_path in paths:
        if not failed_path.exists():
            continue
        try:
            with open(failed_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            continue
    return None


def save_failed_segment(manifest_cache_dir, segment, profile, exc, failure_stage="normal_render"):
    cache_dir = Path(manifest_cache_dir)
    failed_path = get_segment_failed_path(cache_dir, segment)
    failure_class, error_type = classify_tts_failure(exc, segment)
    data = {
        "segment_id": segment.get("segment_id"),
        "order": int(segment.get("order", 0) or 0),
        "status": "failed",
        "failure_stage": failure_stage,
        "failure_class": failure_class,
        "error_type": error_type,
        "error_message": str(exc),
        "text": clean_tts_text(segment.get("text", "")),
        "voice": profile.get("voice"),
        "rate_pct": int(profile.get("rate_pct", 0) or 0),
        "pitch_hz": int(profile.get("pitch_hz", 0) or 0),
        "cache_layout": "split_failed",
        "failed_at": time.time(),
    }
    # Do not leave stale success cache for a segment that failed with current text/settings.
    for paths in get_all_segment_cache_path_pairs(cache_dir, segment):
        for key in ("audio", "meta"):
            try:
                if paths[key].exists():
                    paths[key].unlink(missing_ok=True)
            except Exception:
                pass
    atomic_write_json(failed_path, data)
    return data


def clear_failed_segment(manifest_cache_dir, segment):
    for failed_path in get_all_segment_failed_paths(manifest_cache_dir, segment):
        try:
            if failed_path.exists():
                failed_path.unlink(missing_ok=True)
        except Exception:
            pass


def load_cached_segment(manifest_cache_dir, segment, cache_key, cache_name=None, allow_remote_download=True):
    cache_dir = Path(manifest_cache_dir)
    for paths in get_all_segment_cache_path_pairs(cache_dir, segment):
        if allow_remote_download and (not paths["audio"].exists() or not paths["meta"].exists()) and cache_name:
            try:
                download_chunk_cache_file(cache_name, _rel_cache_path(cache_dir, paths["audio"]), str(paths["audio"]))
                download_chunk_cache_file(cache_name, _rel_cache_path(cache_dir, paths["meta"]), str(paths["meta"]))
            except Exception:
                pass
        if not paths["audio"].exists() or not paths["meta"].exists():
            continue
        try:
            with open(paths["meta"], "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue
        if meta.get("cache_key") != cache_key or not meta.get("events"):
            continue
        actual_duration = get_audio_duration_sec(str(paths["audio"]))
        meta_duration = float(meta.get("audio_duration_sec", 0) or 0)
        events_duration = max((float(event.get("end", 0) or 0) for event in meta.get("events") or []), default=0.0)
        duration = max(float(actual_duration or 0), meta_duration, events_duration)
        return {"audio_path": str(paths["audio"]), "events": meta["events"], "audio_duration_sec": round(duration, 3)}
    return None


def save_cached_segment(manifest_cache_dir, segment, cache_key, rendered, cache_name=None, upload_remote=False):
    cache_dir = Path(manifest_cache_dir)
    paths = get_segment_cache_paths(cache_dir, segment)
    atomic_copy_file(rendered["audio_path"], paths["audio"])
    meta_data = {
        "cache_key": cache_key,
        "segment_id": segment.get("segment_id"),
        "order": int(segment.get("order", 0) or 0),
        "voice": rendered["profile"]["voice"],
        "events": rendered["events"],
        "audio_duration_sec": rendered["audio_duration_sec"],
        "status": rendered.get("status", "success"),
        "rescue_method": rendered.get("rescue_method"),
        "split_method": rendered.get("split_method"),
        "cache_layout": "split_success",
        "atomic_cache_write": True,
        "saved_at": time.time(),
    }
    atomic_write_json(paths["meta"], meta_data)
    clear_failed_segment(cache_dir, segment)
    if upload_remote and cache_name:
        upload_chunk_cache_file(str(paths["audio"]), cache_name, _rel_cache_path(cache_dir, paths["audio"]))
        upload_chunk_cache_file(str(paths["meta"]), cache_name, _rel_cache_path(cache_dir, paths["meta"]))



# ==============================================================================
# Final part cache: prebuild contiguous success ranges.
#
# If a long manifest has a few failed segments, all already-successful contiguous
# ranges can be concatenated into reusable MP3 parts immediately. Later, after the
# failed gaps are fixed, the final build can stitch large prebuilt parts plus the
# repaired gap parts instead of re-concatenating thousands of tiny segment files.
# Each part includes the configured pause_after_ms after every segment in the range,
# matching the original final concat behavior.
# ==============================================================================

def get_final_parts_dir(manifest_cache_dir):
    path = Path(manifest_cache_dir) / FINAL_PARTS_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def segment_part_signature(segment, rendered, defaults):
    profile = rendered.get("profile") or resolve_segment_tts_settings(segment, defaults)
    return {
        "segment_id": str(segment.get("segment_id")),
        "order": int(segment.get("order", 0) or 0),
        "cache_key": make_segment_cache_key(segment, profile),
        "audio_duration_sec": round(float(rendered.get("audio_duration_sec", 0) or 0), 3),
        "pause_after_ms": int(profile.get("pause_after_ms", segment.get("pause_after_ms", 0)) or 0),
        "voice": profile.get("voice"),
        "rate_pct": int(profile.get("rate_pct", 0) or 0),
        "pitch_hz": int(profile.get("pitch_hz", 0) or 0),
        "emotion": profile.get("emotion", get_vieneu_emotion()),
    }


def build_final_part_stem(range_segments, signatures):
    first_order = int(range_segments[0].get("order", 0) or 0)
    last_order = int(range_segments[-1].get("order", 0) or 0)
    payload = json.dumps(signatures, sort_keys=True, ensure_ascii=False).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:16]
    return f"part_{first_order:06d}_{last_order:06d}__{digest}"


def load_valid_final_part(manifest_cache_dir, stem, signatures):
    parts_dir = get_final_parts_dir(manifest_cache_dir)
    audio_path = parts_dir / f"{stem}.mp3"
    meta_path = parts_dir / f"{stem}.json"
    if not audio_path.exists() or not meta_path.exists():
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("signatures") != signatures:
            return None
        duration = float(meta.get("audio_duration_sec", 0) or 0)
        if duration <= 0:
            duration = float(get_audio_duration_sec(str(audio_path)) or 0)
        if duration <= 0:
            return None
        meta["audio_path"] = str(audio_path)
        meta["audio_duration_sec"] = round(duration, 3)
        meta["cache_hit"] = True
        return meta
    except Exception:
        return None


def collect_success_ranges_for_parts(ordered_segments, rendered_by_segment_id, min_segments=FINAL_PART_MIN_SEGMENTS):
    ranges = []
    current = []
    for segment in ordered_segments:
        seg_id = str(segment.get("segment_id"))
        if seg_id in rendered_by_segment_id:
            current.append(segment)
            continue
        if len(current) >= int(min_segments or 2):
            ranges.append(current)
        current = []
    if len(current) >= int(min_segments or 2):
        ranges.append(current)
    return ranges


def build_segment_events_for_range(range_segments, rendered_by_segment_id, timeline_mode):
    segment_events = []
    for segment in range_segments:
        seg_id = str(segment.get("segment_id"))
        rendered = rendered_by_segment_id[seg_id]
        profile = rendered["profile"]
        part_audio = rendered["audio_path"]
        audio_duration = float(rendered["audio_duration_sec"])
        if timeline_mode == TIMELINE_ACCURATE:
            speech_bounds = detect_audio_active_bounds_sec(part_audio, duration_sec=audio_duration)
        else:
            speech_bounds = {"speech_start_sec": 0.0, "speech_end_sec": audio_duration}
        segment_events.append({
            "segment_id": seg_id,
            "events": rendered["events"],
            "audio_duration_sec": audio_duration,
            "pause_after_ms": int(profile.get("pause_after_ms", segment.get("pause_after_ms", 0))),
            "speech_start_sec": speech_bounds.get("speech_start_sec", 0.0),
            "speech_end_sec": speech_bounds.get("speech_end_sec", audio_duration),
        })
    return segment_events


def build_or_load_success_part(range_segments, rendered_by_segment_id, defaults, manifest_cache_dir, tmpdir, timeline_mode, silence_cache=None):
    signatures = [segment_part_signature(seg, rendered_by_segment_id[str(seg.get("segment_id"))], defaults) for seg in range_segments]
    stem = build_final_part_stem(range_segments, signatures)
    cached = load_valid_final_part(manifest_cache_dir, stem, signatures)
    if cached:
        return cached

    silence_cache = silence_cache if silence_cache is not None else {}
    concat_inputs = []
    for segment in range_segments:
        seg_id = str(segment.get("segment_id"))
        rendered = rendered_by_segment_id[seg_id]
        part_audio = rendered["audio_path"]
        if not os.path.exists(part_audio):
            raise RuntimeError(f"Cannot build success part because segment audio is missing: {seg_id} -> {part_audio}")
        concat_inputs.append(part_audio)
        profile = rendered["profile"]
        pause_ms = int(profile.get("pause_after_ms", segment.get("pause_after_ms", 0)) or 0)
        if pause_ms > 0:
            silence_path = get_or_build_silence_audio(pause_ms, tmpdir=tmpdir, silence_cache=silence_cache)
            if not silence_path:
                raise RuntimeError(f"Unable to create reusable silence audio in success part after segment: {seg_id}")
            concat_inputs.append(silence_path)

    parts_dir = get_final_parts_dir(manifest_cache_dir)
    tmp_audio = os.path.join(tmpdir, f"{stem}.building.mp3")
    concat_meta = concat_audio_files(concat_inputs, tmp_audio)
    duration = float(get_audio_duration_sec(tmp_audio) or 0)
    events = build_segment_events_for_range(range_segments, rendered_by_segment_id, timeline_mode)
    meta = {
        "schema": "vieneu_tts_final_part.v1",
        "part_stem": stem,
        "start_order": int(range_segments[0].get("order", 0) or 0),
        "end_order": int(range_segments[-1].get("order", 0) or 0),
        "segments_count": len(range_segments),
        "segment_ids": [str(seg.get("segment_id")) for seg in range_segments],
        "signatures": signatures,
        "audio_duration_sec": round(duration, 3),
        "segment_events": events,
        "concat_method": concat_meta.get("method"),
        "includes_pause_after_each_segment": True,
        "timeline_mode": timeline_mode,
        "built_at": time.time(),
        "cache_hit": False,
    }
    audio_path = parts_dir / f"{stem}.mp3"
    meta_path = parts_dir / f"{stem}.json"
    atomic_copy_file(tmp_audio, audio_path)
    atomic_write_json(meta_path, meta)
    try:
        os.remove(tmp_audio)
    except Exception:
        pass
    meta["audio_path"] = str(audio_path)
    return meta


def prebuild_contiguous_success_parts(ordered_segments, rendered_by_segment_id, defaults, manifest_cache_dir, tmpdir, timeline_mode, progress_callback=None, total=None, failed_count=0, phase_label="part_cache"):
    ranges = collect_success_ranges_for_parts(ordered_segments, rendered_by_segment_id, min_segments=FINAL_PART_MIN_SEGMENTS)
    built_parts = []
    if progress_callback:
        preview = ", ".join(f"{int(r[0].get('order',0)):06d}-{int(r[-1].get('order',0)):06d}" for r in ranges[:8])
        progress_callback(
            total=total or len(ordered_segments),
            done=len(rendered_by_segment_id),
            ok=len(rendered_by_segment_id),
            failed=failed_count,
            status="RUNNING",
            phase=phase_label,
            detail=f"prebuilding contiguous success parts ranges={len(ranges)}" + (f" [{preview}]" if preview else ""),
        )
    silence_cache = {}
    for idx, range_segments in enumerate(ranges, start=1):
        start_order = int(range_segments[0].get("order", 0) or 0)
        end_order = int(range_segments[-1].get("order", 0) or 0)
        if progress_callback:
            progress_callback(
                total=total or len(ordered_segments),
                done=len(rendered_by_segment_id),
                ok=len(rendered_by_segment_id),
                failed=failed_count,
                status="RUNNING",
                phase=phase_label,
                detail=f"part {idx}/{len(ranges)} building/reusing success range {start_order}-{end_order} segments={len(range_segments)}",
            )
        part_meta = build_or_load_success_part(range_segments, rendered_by_segment_id, defaults, manifest_cache_dir, tmpdir, timeline_mode, silence_cache=silence_cache)
        built_parts.append(part_meta)
    return built_parts


def build_final_from_success_parts_or_segments(ordered_segments, rendered_by_segment_id, defaults, manifest_cache_dir, tmpdir, timeline_mode, progress_callback=None, total=None, failed_count=0):
    """Build final concat inputs using large prebuilt success parts where possible.

    When complete, all segments are success. This prefers cached contiguous success parts
    (usually one whole-manifest part) and falls back to individual segment audio only for
    ranges that are too small for part caching. The pause_after_ms policy is preserved.
    """
    part_ranges = collect_success_ranges_for_parts(ordered_segments, rendered_by_segment_id, min_segments=FINAL_PART_MIN_SEGMENTS)
    range_by_start = {int(r[0].get("order", 0) or 0): r for r in part_ranges}
    concat_inputs = []
    segment_events = []
    part_cache_meta = []
    silence_cache = {}
    idx = 0
    while idx < len(ordered_segments):
        segment = ordered_segments[idx]
        order = int(segment.get("order", idx + 1) or (idx + 1))
        if order in range_by_start:
            range_segments = range_by_start[order]
            if progress_callback:
                progress_callback(total=total or len(ordered_segments), done=len(rendered_by_segment_id), ok=len(rendered_by_segment_id), failed=failed_count, status="RUNNING", phase="final_part", detail=f"using/building success part {int(range_segments[0].get('order',0))}-{int(range_segments[-1].get('order',0))} segments={len(range_segments)}")
            part_meta = build_or_load_success_part(range_segments, rendered_by_segment_id, defaults, manifest_cache_dir, tmpdir, timeline_mode, silence_cache=silence_cache)
            concat_inputs.append(part_meta["audio_path"])
            segment_events.extend(part_meta.get("segment_events") or [])
            part_cache_meta.append({k: part_meta.get(k) for k in ("part_stem", "start_order", "end_order", "segments_count", "audio_duration_sec", "cache_hit")})
            idx += len(range_segments)
            continue

        # Fallback for single success segment, preserving pause.
        seg_id = str(segment.get("segment_id"))
        rendered = rendered_by_segment_id[seg_id]
        part_audio = rendered["audio_path"]
        if not os.path.exists(part_audio):
            raise RuntimeError(f"Rendered audio disappeared before concat: {seg_id} -> {part_audio}")
        concat_inputs.append(part_audio)
        segment_events.extend(build_segment_events_for_range([segment], rendered_by_segment_id, timeline_mode))
        profile = rendered["profile"]
        pause_ms = int(profile.get("pause_after_ms", segment.get("pause_after_ms", 0)) or 0)
        if pause_ms > 0:
            silence_path = get_or_build_silence_audio(pause_ms, tmpdir=tmpdir, silence_cache=silence_cache)
            if not silence_path:
                raise RuntimeError(f"Unable to create reusable silence audio after segment: {seg_id}")
            concat_inputs.append(silence_path)
        idx += 1
    return concat_inputs, segment_events, part_cache_meta


def load_compact_final_part_metas(manifest_cache_dir, ordered_segments=None):
    """Load reusable final_parts from cache, validating audio+meta and segment order.

    These parts are the compact remote cache replacement for thousands of raw
    success/*.mp3 files. They can be downloaded and stitched directly without
    loading/rendering each successful segment again.
    """
    parts_dir = get_final_parts_dir(manifest_cache_dir)
    if not parts_dir.exists():
        return []
    expected_by_order = {}
    if ordered_segments:
        for seg in ordered_segments:
            expected_by_order[int(seg.get('order', 0) or 0)] = str(seg.get('segment_id'))
    parts = []
    for meta_path in sorted(parts_dir.glob('*.json')):
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            audio_path = meta_path.with_suffix('.mp3')
            if not audio_path.exists():
                continue
            start_order = int(meta.get('start_order', 0) or 0)
            end_order = int(meta.get('end_order', 0) or 0)
            segment_ids = [str(x) for x in (meta.get('segment_ids') or [])]
            if start_order <= 0 or end_order < start_order or not segment_ids:
                continue
            if expected_by_order:
                expected_ids = [expected_by_order.get(order) for order in range(start_order, end_order + 1)]
                if expected_ids != segment_ids:
                    continue
            duration = float(meta.get('audio_duration_sec', 0) or 0)
            if duration <= 0:
                duration = float(get_audio_duration_sec(str(audio_path)) or 0)
            if duration <= 0:
                continue
            meta['audio_path'] = str(audio_path)
            meta['audio_duration_sec'] = round(duration, 3)
            meta['cache_hit'] = True
            meta['source'] = 'compact_final_parts'
            parts.append(meta)
        except Exception:
            continue
    # Prefer longer parts if duplicates start at the same order.
    parts.sort(key=lambda item: (int(item.get('start_order', 0) or 0), -int(item.get('end_order', 0) or 0)))
    return parts


def compact_part_covered_segment_ids(manifest_cache_dir, ordered_segments):
    ids = set()
    for part in load_compact_final_part_metas(manifest_cache_dir, ordered_segments=ordered_segments):
        ids.update(str(x) for x in (part.get('segment_ids') or []))
    return ids


def build_final_from_compact_parts_or_segments(ordered_segments, rendered_by_segment_id, defaults, manifest_cache_dir, tmpdir, timeline_mode, progress_callback=None, total=None, failed_count=0):
    """Build final using compact final_parts plus newly rendered gap segments.

    This is the clean remote-cache path:
    - final_parts/ covers already-success contiguous ranges
    - rendered_by_segment_id covers freshly retried failed gaps
    - raw success/ segment files are not required from the repo
    """
    compact_parts = load_compact_final_part_metas(manifest_cache_dir, ordered_segments=ordered_segments)
    parts_by_start = {}
    for part in compact_parts:
        start_order = int(part.get('start_order', 0) or 0)
        current = parts_by_start.get(start_order)
        if current is None or int(part.get('end_order', 0) or 0) > int(current.get('end_order', 0) or 0):
            parts_by_start[start_order] = part

    concat_inputs = []
    segment_events = []
    part_cache_meta = []
    silence_cache = {}
    missing = []
    idx = 0
    while idx < len(ordered_segments):
        segment = ordered_segments[idx]
        order = int(segment.get('order', idx + 1) or (idx + 1))
        part = parts_by_start.get(order)
        if part:
            end_order = int(part.get('end_order', order) or order)
            segment_ids = [str(x) for x in (part.get('segment_ids') or [])]
            expected_ids = [str(ordered_segments[j].get('segment_id')) for j in range(idx, min(len(ordered_segments), idx + len(segment_ids)))]
            if segment_ids == expected_ids:
                concat_inputs.append(part['audio_path'])
                segment_events.extend(part.get('segment_events') or [])
                part_cache_meta.append({
                    'part_stem': part.get('part_stem'),
                    'start_order': part.get('start_order'),
                    'end_order': part.get('end_order'),
                    'segments_count': part.get('segments_count'),
                    'audio_duration_sec': part.get('audio_duration_sec'),
                    'cache_hit': True,
                    'source': 'compact_final_parts',
                })
                if progress_callback:
                    progress_callback(total=total or len(ordered_segments), done=order, ok=len(rendered_by_segment_id), failed=failed_count, status='RUNNING', phase='final_part', detail=f"using compact final_part {part.get('start_order')}-{part.get('end_order')} segments={part.get('segments_count')}")
                idx += len(segment_ids)
                continue

        seg_id = str(segment.get('segment_id'))
        rendered = rendered_by_segment_id.get(seg_id)
        if not rendered:
            missing.append(seg_id)
            idx += 1
            continue
        part_audio = rendered['audio_path']
        if not os.path.exists(part_audio):
            missing.append(seg_id)
            idx += 1
            continue
        concat_inputs.append(part_audio)
        segment_events.extend(build_segment_events_for_range([segment], rendered_by_segment_id, timeline_mode))
        profile = rendered['profile']
        pause_ms = int(profile.get('pause_after_ms', segment.get('pause_after_ms', 0)) or 0)
        if pause_ms > 0:
            silence_path = get_or_build_silence_audio(pause_ms, tmpdir=tmpdir, silence_cache=silence_cache)
            if not silence_path:
                raise RuntimeError(f"Unable to create reusable silence audio after repaired segment: {seg_id}")
            concat_inputs.append(silence_path)
        idx += 1

    if missing:
        raise RuntimeError('compact final build missing segment audio/part for: ' + ', '.join(missing[:12]))
    return concat_inputs, segment_events, part_cache_meta

async def process_manifest_file(file, subtitle_format, upload=True, work_dir=None, tts_timeout_sec=None, force_rerender=False, progress_callback=None, rescue_repeated_short_failures=False, repo_cache_only=True, force_commit_cache_now=False, segment_worker_count=SEGMENT_RENDER_WORKERS, timeline_mode=DEFAULT_TIMELINE_MODE, tts_semaphore=None):
    manifest_name = os.path.basename(file.name)
    if progress_callback:
        progress_callback(total=0, done=0, ok=0, failed=0, status="RUNNING", phase="load", detail=f"loading manifest {manifest_name}")
    manifest = load_manifest(file.name)
    # Always start this manifest from a clean local cache folder. This prevents stale
    # local/pending cache from a previous run with the same filename but changed text.
    if progress_callback:
        progress_callback(status="RUNNING", phase="cache_clean", detail="clearing local/pending cache folder before render")
    removed_local_start = clear_manifest_local_cache_state(manifest_name)
    removed_for_rerender = clear_manifest_rerender_state(manifest_name) if force_rerender else []
    if progress_callback:
        progress_callback(status="RUNNING", phase="cache_clean", detail=f"local cache clean done removed={len(removed_local_start)}" + (f" | force_rerender removed={len(removed_for_rerender)}" if force_rerender else ""))
    segment_count = len(manifest.get("segments") or [])
    voice_counts = {}
    for segment in manifest.get("segments") or []:
        voice_name = str(segment.get("voice") or "").strip() or "(auto)"
        voice_counts[voice_name] = voice_counts.get(voice_name, 0) + 1
    result = await render_manifest_to_outputs(
        manifest=manifest,
        subtitle_format=subtitle_format,
        upload=bool(upload),
        cache_name=manifest_name,
        work_dir=work_dir,
        tts_timeout_sec=tts_timeout_sec,
        force_rerender=force_rerender,
        progress_callback=progress_callback,
        rescue_repeated_short_failures=rescue_repeated_short_failures,
        repo_cache_only=repo_cache_only,
        segment_worker_count=segment_worker_count,
        timeline_mode=timeline_mode,
        tts_semaphore=tts_semaphore,
    )
    complete = bool(result.get("complete", result.get("final_audio") is not None))
    output_files = []
    upload_entries = []

    if complete:
        output_files.append(result["final_audio"])
        upload_entries.append((result["final_audio"], f"outputs/{result['story_prefix']}/{result['audio_name']}"))
        if result.get("local_subtitle"):
            output_files.append(result["local_subtitle"])
            upload_entries.append((result["local_subtitle"], f"outputs/{result['story_prefix']}/{result['subtitle_name']}"))
        output_files.append(result["local_report"])
        upload_entries.append((result["local_report"], f"outputs/{result['story_prefix']}/{result['report_name']}"))
    else:
        # Keep the incomplete report available for download/debug only. Do not stage it as final output.
        output_files.append(result["local_report"])

    # Cache is always preserved, even for incomplete manifests, so next run can reuse successful segments.
    # If force_commit_cache_now is enabled, commit the whole manifest cache folder directly in one commit
    # and do not stage thousands of per-segment cache files into the final batch upload.
    manifest_cache_dir = Path(result["report"].get("segment_cache_dir", ""))
    remote_cache_prefix = result["report"].get("remote_cache_prefix")
    cache_entry_count = 0
    cache_commit_message = ""
    if remote_cache_prefix and manifest_cache_dir.exists():
        cache_files = list(iter_uploadable_cache_files(manifest_cache_dir))
        cache_entry_count = len(cache_files)
        if upload and force_commit_cache_now:
            if progress_callback:
                progress_callback(status="RUNNING", phase="cache_commit", detail=f"committing manifest cache folder now files={cache_entry_count}")
            try:
                cache_commit_message = upload_manifest_cache_folder_once(
                    result["report"].get("cache_name") or manifest_name,
                    commit_message=f"Force commit VieNeu TTS cache folder: {manifest_name}",
                )
                if progress_callback:
                    progress_callback(status="RUNNING", phase="cache_commit", detail=cache_commit_message[:180])
            except Exception as exc:
                cache_commit_message = f"Force cache commit failed: {exc}"
                if progress_callback:
                    progress_callback(status="RUNNING", phase="cache_commit", detail=cache_commit_message[:180])
                for cache_file, rel_cache in cache_files:
                    upload_entries.append((str(cache_file), f"{remote_cache_prefix}/{rel_cache}"))
        else:
            for cache_file, rel_cache in cache_files:
                upload_entries.append((str(cache_file), f"{remote_cache_prefix}/{rel_cache}"))

    prefix = f"Rendered locally for upload staging ({result['story_prefix']})" if upload else f"Rendered locally for download ({result['story_prefix']})"
    if complete:
        message = (
            f"{prefix}: {result['audio_name']}"
            f"{', ' + result['subtitle_name'] if result['subtitle_name'] else ''}, {result['report_name']}"
            f" | complete=yes segments={result['report']['segments_total']} cache_hits={result['report']['cache_hits']}"
            f" | input_segments={segment_count} voices={voice_counts}"
            f" | cache={result['report'].get('remote_cache_prefix')}"
            + f" | local_cache_clean_start removed={len(removed_local_start)}"
            + (f" | force_rerender=on removed={len(removed_for_rerender)}" if force_rerender else "")
            + (f" | {cache_commit_message}" if cache_commit_message else "")
        )
    else:
        failed_preview = ", ".join(
            f"{item.get('segment_id')}@{item.get('order')}"
            for item in (result['report'].get('failed_segments') or [])[:8]
        )
        message = (
            f"Incomplete manifest cached for retry ({result['story_prefix']}): "
            f"ok={result['report']['segments_rendered_ok']}/{result['report']['segments_total']} "
            f"failed={result['report']['segments_failed']} [{failed_preview}] "
            f"| final audio/subtitle NOT built | cache_files_staged={cache_entry_count} "
            f"| report={result['report_name']}"
            + f" | local_cache_clean_start removed={len(removed_local_start)}"
            + (f" | force_rerender=on removed={len(removed_for_rerender)}" if force_rerender else "")
            + (f" | {cache_commit_message}" if cache_commit_message else "")
        )

    return {
        "message": message,
        "files": output_files,
        "upload_entries": upload_entries,
    }



# ==============================================================================
# Segment shared-ref infer modes
# ==============================================================================
# These modes keep the manifest/SRT segment list unchanged, but render adjacent
# segments together in one VieNeu infer call to reduce voice/tone drift. The group
# audio is then cut back into real per-segment MP3 clips, so final concat and SRT
# still operate on the original segment_id/order/text.

SEGMENT_INFER_STRATEGY_NONE = "none"
SEGMENT_INFER_STRATEGY_ADJACENT_NON_DIALOGUE = "adjacent_non_dialogue_shared_ref"
SEGMENT_INFER_STRATEGY_ALL_SEGMENTS_SINGLE_VOICE = "all_segments_single_voice_shared_ref"
SEGMENT_INFER_STRATEGY_NATIVE_VIENEU = "vieneu_native_infer_segments"


def get_segment_infer_strategy(manifest):
    audio_strategy = (manifest or {}).get("audio_strategy") or {}
    raw = str(
        audio_strategy.get("segment_infer_strategy")
        or (manifest or {}).get("segment_infer_strategy")
        or SEGMENT_INFER_STRATEGY_NONE
    ).strip().lower()
    aliases = {
        "": SEGMENT_INFER_STRATEGY_NONE,
        "none": SEGMENT_INFER_STRATEGY_NONE,
        "normal": SEGMENT_INFER_STRATEGY_NONE,
        "per_segment": SEGMENT_INFER_STRATEGY_NONE,
        "segments": SEGMENT_INFER_STRATEGY_NONE,
        "segments_ref_chunks": SEGMENT_INFER_STRATEGY_ADJACENT_NON_DIALOGUE,
        "adjacent_non_dialogue": SEGMENT_INFER_STRATEGY_ADJACENT_NON_DIALOGUE,
        "adjacent_non_dialogue_shared_ref": SEGMENT_INFER_STRATEGY_ADJACENT_NON_DIALOGUE,
        "single_voice": SEGMENT_INFER_STRATEGY_ALL_SEGMENTS_SINGLE_VOICE,
        "segments_single_voice": SEGMENT_INFER_STRATEGY_ALL_SEGMENTS_SINGLE_VOICE,
        "all_segments_single_voice": SEGMENT_INFER_STRATEGY_ALL_SEGMENTS_SINGLE_VOICE,
        "all_segments_single_voice_shared_ref": SEGMENT_INFER_STRATEGY_ALL_SEGMENTS_SINGLE_VOICE,
        "segments_native_infer": SEGMENT_INFER_STRATEGY_NATIVE_VIENEU,
        "segment_native_infer": SEGMENT_INFER_STRATEGY_NATIVE_VIENEU,
        "native_infer_segments": SEGMENT_INFER_STRATEGY_NATIVE_VIENEU,
        "vieneu_native_infer_segments": SEGMENT_INFER_STRATEGY_NATIVE_VIENEU,
    }
    return aliases.get(raw, raw if raw in {
        SEGMENT_INFER_STRATEGY_NONE,
        SEGMENT_INFER_STRATEGY_ADJACENT_NON_DIALOGUE,
        SEGMENT_INFER_STRATEGY_ALL_SEGMENTS_SINGLE_VOICE,
        SEGMENT_INFER_STRATEGY_NATIVE_VIENEU,
    } else SEGMENT_INFER_STRATEGY_NONE)


def build_shared_ref_segment_groups(ordered_segments, defaults, strategy):
    """Return groups of original segments to render together.

    Important: this does NOT change segment order, ids, text, or SRT granularity.
    It only decides which adjacent segments share one synthesize_audio() call.
    """
    groups = []
    current = []

    def flush():
        nonlocal current
        if current:
            groups.append(current)
            current = []

    if strategy == SEGMENT_INFER_STRATEGY_ALL_SEGMENTS_SINGLE_VOICE:
        # One-voice segments mode: all adjacent segments can share one infer context.
        # VieNeu SDK can still split internally; we cut the resulting audio back to
        # per-segment clips for final/SRT.
        all_segments = [seg for seg in ordered_segments if clean_tts_text(seg.get("text", ""))]
        return [all_segments] if all_segments else []

    if strategy == SEGMENT_INFER_STRATEGY_ADJACENT_NON_DIALOGUE:
        for seg in ordered_segments:
            if not clean_tts_text(seg.get("text", "")):
                continue
            if is_dialogue_segment_for_vieneu(seg):
                flush()
                groups.append([seg])
            else:
                current.append(seg)
        flush()
        return groups

    return [[seg] for seg in ordered_segments if clean_tts_text(seg.get("text", ""))]


def _shared_group_text(group_segments):
    # Blank lines encourage a small pause between original segments while keeping
    # the whole group in one reference/context.
    return "\n\n".join(clean_tts_text(seg.get("text", "")) for seg in group_segments if clean_tts_text(seg.get("text", ""))).strip()


async def render_shared_ref_segment_group(
    group_segments,
    defaults,
    tmpdir,
    group_index,
    manifest_cache_dir=None,
    cache_name=None,
    tts_timeout_sec=None,
    force_rerender=False,
    repo_cache_only=False,
    tts_semaphore=None,
):
    """Render one shared-ref group, then cut it into original segment clips.

    Returns: {segment_id: rendered_dict}
    """
    if not group_segments:
        return {}
    first = group_segments[0]
    profile = resolve_segment_tts_settings(first, defaults)
    group_text = _shared_group_text(group_segments)
    if not group_text:
        return {}

    start_order = int(first.get("order", 0) or 0)
    end_order = int(group_segments[-1].get("order", start_order) or start_order)
    group_label = f"sharedref_{start_order:06d}_{end_order:06d}_{group_index:04d}"
    group_audio = os.path.join(tmpdir, f"{group_label}.mp3")

    async def _do_synth():
        return await synthesize_audio(
            text=group_text,
            voice=profile["voice"],
            rate=profile["rate_pct"],
            pitch=profile["pitch_hz"],
            audio_path=group_audio,
            timeout_sec=tts_timeout_sec,
            emotion=profile.get("emotion"),
        )

    try:
        if tts_semaphore is not None:
            async with tts_semaphore:
                raw_events = await _do_synth()
        else:
            raw_events = await _do_synth()
    except Exception as exc:
        raise RuntimeError(
            f"shared-ref group render failed {group_label} segments={len(group_segments)} "
            f"voice={profile.get('voice')} emotion={profile.get('emotion')} | {exc}"
        ) from exc

    raw_events = ensure_event_fallback(raw_events, group_text, profile["rate_pct"])
    group_duration = float(get_audio_duration_sec(group_audio) or 0)
    if group_duration <= 0:
        group_duration = max((float(e.get("end", 0) or 0) for e in raw_events), default=0.0)
    if group_duration <= 0:
        raise RuntimeError(f"shared-ref group produced no valid audio duration: {group_label}")

    # Split group audio back into the original segment clips. This keeps final audio
    # and SRT segment-based. Bounds use real group audio duration, Edge/VieNeu events
    # when available, then silence snapping/weighted fallback.
    bounds = compute_rescue_split_bounds(raw_events, group_segments, group_audio, group_duration)
    rendered_map = {}
    for idx, (segment, bound) in enumerate(zip(group_segments, bounds), start=1):
        seg_id = str(segment.get("segment_id"))
        seg_profile = resolve_segment_tts_settings(segment, defaults)
        # In shared-ref modes all segments in the group are intentionally rendered
        # using the group's reference/profile. Preserve that same profile in cache.
        seg_profile.update({
            "voice": profile.get("voice"),
            "rate_pct": int(profile.get("rate_pct", 0) or 0),
            "pitch_hz": int(profile.get("pitch_hz", 0) or 0),
            "emotion": profile.get("emotion", seg_profile.get("emotion")),
        })
        out_audio = os.path.join(tmpdir, f"segment_{int(segment.get('order', idx) or idx):06d}.mp3")
        st, en = bound
        cut_audio_clip(group_audio, out_audio, st, en)
        seg_duration = float(get_audio_duration_sec(out_audio) or max(0.08, float(en) - float(st)))
        seg_text = clean_tts_text(segment.get("text", ""))
        seg_events = build_segment_subtitle_events_from_json_text([], seg_text, seg_duration, seg_profile.get("rate_pct", 0))
        rendered = {
            "cache_hit": False,
            "audio_path": out_audio,
            "events": seg_events,
            "audio_duration_sec": round(seg_duration, 3),
            "profile": seg_profile,
            "split_method": "shared_ref_group_cut",
            "source_group_segments": [str(s.get("segment_id")) for s in group_segments],
            "source_group_audio_duration_sec": round(group_duration, 3),
        }
        if manifest_cache_dir:
            cache_key = make_segment_cache_key(segment, seg_profile)
            save_cached_segment(
                manifest_cache_dir,
                segment,
                cache_key,
                rendered,
                cache_name=cache_name,
                upload_remote=False,
            )
        rendered_map[seg_id] = rendered
    return rendered_map


def _native_infer_segments_batch_limits():
    try:
        max_chars = int(os.getenv("VIENEU_INFER_SEGMENTS_BATCH_CHARS", "12000") or 12000)
    except Exception:
        max_chars = 12000
    try:
        max_segments = int(os.getenv("VIENEU_INFER_SEGMENTS_BATCH_SEGMENTS", "80") or 80)
    except Exception:
        max_segments = 80
    return max(1200, max_chars), max(1, max_segments)


def build_vieneu_native_segment_batches(ordered_segments):
    max_chars, max_segments = _native_infer_segments_batch_limits()
    batches = []
    current = []
    current_chars = 0
    for seg in ordered_segments or []:
        text = clean_tts_text(seg.get("text", ""))
        if not text:
            continue
        text_len = len(text)
        if current and (len(current) >= max_segments or current_chars + text_len > max_chars):
            batches.append(current)
            current = []
            current_chars = 0
        current.append(seg)
        current_chars += text_len
    if current:
        batches.append(current)
    return batches


async def render_vieneu_native_segment_batch(
    group_segments,
    defaults,
    tmpdir,
    batch_index,
    manifest_cache_dir=None,
    cache_name=None,
    tts_timeout_sec=None,
):
    """Render a native VieNeu infer_segments batch and keep per-segment outputs."""
    if not group_segments:
        return {}

    batch_orders = [int(seg.get("order", 0) or 0) for seg in group_segments]
    batch_label = f"nativeinfer_{min(batch_orders or [0]):06d}_{max(batch_orders or [0]):06d}_{batch_index:04d}"
    narrator_emotion = get_vieneu_narration_emotion() if "get_vieneu_narration_emotion" in globals() else "storytelling"
    dialogue_emotion = get_vieneu_dialogue_emotion() if "get_vieneu_dialogue_emotion" in globals() else "natural"
    base_timeout = int(tts_timeout_sec or SEGMENT_TTS_TIMEOUT_SEC or 60)
    scaled_timeout = base_timeout * max(1, min(4, (len(group_segments) // 10) + 1))

    def _run():
        tts = get_vieneu_engine(emotion=narrator_emotion)
        infer_segments_fn = getattr(tts, "infer_segments", None)
        if not callable(infer_segments_fn):
            raise RuntimeError("Current VieNeu engine does not expose infer_segments().")

        log_vieneu_stability_once()
        infer_kwargs = build_vieneu_infer_kwargs_for_stability(tts) if "build_vieneu_infer_kwargs_for_stability" in globals() else {}
        speed_values = get_vieneu_standard_speed_defaults() if "get_vieneu_standard_speed_defaults" in globals() else {}
        if "max_chars" in speed_values:
            infer_kwargs["max_chars"] = int(speed_values["max_chars"])
        if "apply_watermark" in speed_values:
            infer_kwargs["apply_watermark"] = bool(speed_values["apply_watermark"])

        narrator_voice_key = resolve_vieneu_voice_for_segment({"segment_type": "narration"}, defaults)
        female_voice_key = resolve_vieneu_voice_for_segment({"segment_type": "dialogue", "performed_voice_persona": {"gender": "female"}}, defaults)
        male_voice_key = resolve_vieneu_voice_for_segment({"segment_type": "dialogue", "performed_voice_persona": {"gender": "male"}}, defaults)
        narrator_voice = get_vieneu_voice_data(narrator_voice_key, tts=tts) or narrator_voice_key
        female_voice = get_vieneu_voice_data(female_voice_key, tts=tts) or female_voice_key
        male_voice = get_vieneu_voice_data(male_voice_key, tts=tts) or male_voice_key

        narrator_emotion_tag = vieneu_emotion_tag_for_infer(narrator_emotion) if not vieneu_init_supports_emotion() else None
        dialogue_emotion_tag = vieneu_emotion_tag_for_infer(dialogue_emotion) if not vieneu_init_supports_emotion() else None

        result = infer_segments_fn(
            segments=[dict(seg) for seg in group_segments],
            narrator_voice=narrator_voice,
            female_voice=female_voice,
            male_voice=male_voice,
            narrator_emotion_tag=narrator_emotion_tag,
            dialogue_emotion_tag=dialogue_emotion_tag,
            narrator_stability_mode=os.getenv("VIENEU_INFER_SEGMENTS_NARRATOR_STABILITY_MODE", "locked_safe"),
            dialogue_stability_mode=os.getenv("VIENEU_INFER_SEGMENTS_DIALOGUE_STABILITY_MODE", "stable"),
            group_max_chars=int(os.getenv("VIENEU_INFER_SEGMENTS_GROUP_MAX_CHARS", "1200") or 1200),
            return_metadata=True,
            **infer_kwargs,
        )
        wavs = result.get("wavs") if isinstance(result, dict) else result
        if len(wavs or []) != len(group_segments):
            raise RuntimeError(f"infer_segments returned {len(wavs or [])} wav(s) for {len(group_segments)} segment(s)")

        saved = []
        for idx, (seg, wav) in enumerate(zip(group_segments, wavs), start=1):
            out_audio = os.path.join(tmpdir, f"segment_{int(seg.get('order', idx) or idx):06d}.mp3")
            _vieneu_save_audio(tts, wav, out_audio)
            saved.append({
                "segment_id": str(seg.get("segment_id")),
                "audio_path": out_audio,
                "audio_duration_sec": round(float(get_audio_duration_sec(out_audio) or 0), 3),
            })
        return saved, (result.get("groups") if isinstance(result, dict) else [])

    try:
        saved_items, group_meta = await asyncio.wait_for(asyncio.to_thread(_run), timeout=scaled_timeout)
    except asyncio.TimeoutError as exc:
        raise TimeoutError(
            f"vieneu infer_segments timeout after {scaled_timeout}s batch={batch_label} segments={len(group_segments)}"
        ) from exc

    saved_by_id = {item["segment_id"]: item for item in saved_items}
    rendered_map = {}
    for idx, segment in enumerate(group_segments, start=1):
        seg_id = str(segment.get("segment_id"))
        item = saved_by_id.get(seg_id)
        if not item:
            raise RuntimeError(f"infer_segments batch missing output for {seg_id}")
        seg_profile = resolve_segment_tts_settings(segment, defaults)
        seg_duration = float(item.get("audio_duration_sec", 0) or 0)
        if seg_duration <= 0:
            seg_duration = float(estimate_duration_from_text(clean_tts_text(segment.get("text", "")), seg_profile.get("rate_pct", 0)) or 0.08)
        seg_events = build_segment_subtitle_events_from_json_text([], clean_tts_text(segment.get("text", "")), seg_duration, seg_profile.get("rate_pct", 0))
        rendered = {
            "cache_hit": False,
            "audio_path": item["audio_path"],
            "events": seg_events,
            "audio_duration_sec": round(seg_duration, 3),
            "profile": seg_profile,
            "split_method": "vieneu_native_infer_segments",
            "source_group_segments": [str(s.get("segment_id")) for s in group_segments],
            "native_group_meta": group_meta,
        }
        if manifest_cache_dir:
            cache_key = make_segment_cache_key(segment, seg_profile)
            save_cached_segment(
                manifest_cache_dir,
                segment,
                cache_key,
                rendered,
                cache_name=cache_name,
                upload_remote=False,
            )
        rendered_map[seg_id] = rendered
    return rendered_map


async def render_manifest_to_outputs(
    manifest,
    subtitle_format,
    upload,
    source_name=None,
    work_dir=None,
    cache_name=None,
    tts_timeout_sec=None,
    force_rerender=False,
    progress_callback=None,
    rescue_repeated_short_failures=False,
    rescue_short_segments_now=False,
    repo_cache_only=True,
    segment_worker_count=SEGMENT_RENDER_WORKERS,
    timeline_mode=DEFAULT_TIMELINE_MODE,
    tts_semaphore=None,
    cache_only=False,
    retry_failed_only=False,
):
    manifest = normalize_manifest(manifest, source_name=source_name)
    errors = validate_manifest(manifest)
    if errors:
        raise ValueError("Manifest validation failed: " + "; ".join(errors))
    source_file = source_name or manifest.get("source_file") or "preview_segments.txt"
    story_prefix = extract_story_prefix(source_file)
    cache_name = build_manifest_cache_name(cache_name or source_file)
    timeline_mode = normalize_timeline_mode(timeline_mode)
    repo_cache_sync = {"enabled": bool(repo_cache_only), "status": "disabled", "files": 0, "cache_scope": "disabled"}
    initial_cache_scope = CACHE_SCOPE_ALL if cache_only else CACHE_SCOPE_FAILED_ONLY
    if progress_callback:
        progress_callback(status="RUNNING", phase="cache", detail=(f"checking repo cache scope={initial_cache_scope} for {cache_name}" if repo_cache_only and not force_rerender else "repo cache refresh skipped"))
    if repo_cache_only and not force_rerender:
        repo_cache_sync = sync_remote_cache_folder_to_local(cache_name, clear_local_first=True, cache_scope=initial_cache_scope)
    if progress_callback:
        progress_callback(status="RUNNING", phase="cache", detail=f"repo cache {repo_cache_sync.get('status')} scope={repo_cache_sync.get('cache_scope')} files={repo_cache_sync.get('files', 0)}")
    manifest_cache_dir = build_manifest_cache_dir(cache_name)
    defaults = dict(DEFAULT_MANIFEST_STRATEGY)
    defaults.update(manifest.get("audio_strategy") or {})
    base = sanitize_filename(source_file) + "_audio"
    audio_name = f"{base}.mp3"
    subtitle_name = None if subtitle_format == "no_script" else f"{base}.{subtitle_format}"
    report_name = f"{base}_render_report.json"
    ffmpeg_available = bool(find_ffmpeg())
    if not ffmpeg_available:
        raise RuntimeError(
            "Manifest mode requires ffmpeg for safe audio concatenation and pause insertion. "
            "Install ffmpeg before rendering audio_segments.json."
        )
    cleanup_dir = None
    tmpdir = work_dir
    if tmpdir is None:
        cleanup_dir = tempfile.TemporaryDirectory()
        tmpdir = cleanup_dir.name
    else:
        os.makedirs(tmpdir, exist_ok=True)
    try:
        ordered_segments = sorted(manifest["segments"], key=lambda item: int(item.get("order", 0)))
        segment_infer_strategy = get_segment_infer_strategy(manifest)
        cache_prune_report = prune_manifest_cache_to_current_manifest(manifest_cache_dir, ordered_segments, defaults)
        rendered_by_segment_id = {}
        first_pass_failed = []
        rescue_candidates = []
        timeout_segments = []
        short_or_no_audio_segments = []
        other_failed_segments = []
        cache_hits = 0
        rendered_ok_count = 0
        if progress_callback:
            progress_callback(total=len(ordered_segments), done=0, ok=0, failed=0, status="RUNNING", phase="cache", detail=f"manifest loaded | local stale removed={cache_prune_report.get('removed_files', 0)}")

        segment_worker_count = normalize_worker_count(segment_worker_count, SEGMENT_RENDER_WORKERS)
        segment_worker_count = min(segment_worker_count, max(1, len(ordered_segments)))
        processed_count = 0
        previous_failed_map = {}
        previous_failed_pairs = []
        normal_segment_pairs = []
        for order, segment in enumerate(ordered_segments, start=1):
            seg_id = str(segment.get("segment_id", f"seg_{order:06d}"))
            previous_failure = None if force_rerender else load_failed_segment(manifest_cache_dir, segment, cache_name=cache_name)
            if previous_failure:
                previous_failed_map[seg_id] = previous_failure
                previous_failed_pairs.append((order, segment))
            else:
                normal_segment_pairs.append((order, segment))

        previous_failed_retry_count = len(previous_failed_pairs)
        early_stop_failed_first = False
        compact_covered_ids = set()
        compact_parts_loaded_count = 0
        rescued_segments = []
        rescue_failed_segments_list = []

        async def run_segment_pairs(segment_pairs, pass_name):
            nonlocal cache_hits, rendered_ok_count, processed_count
            segment_pairs = order_segment_pairs_for_vieneu_render(segment_pairs)
            if not segment_pairs:
                return
            render_order_strategy = get_vieneu_render_order_strategy()
            if progress_callback and render_order_strategy != "manifest_order":
                non_dialogue_count = sum(1 for _, seg in segment_pairs if not is_dialogue_segment_for_vieneu(seg))
                dialogue_count = sum(1 for _, seg in segment_pairs if is_dialogue_segment_for_vieneu(seg))
                progress_callback(
                    total=len(ordered_segments),
                    done=processed_count,
                    ok=rendered_ok_count,
                    failed=len(first_pass_failed),
                    status="RUNNING",
                    phase="render_order",
                    detail=f"{pass_name}: render_order={render_order_strategy} non_dialogue_first={non_dialogue_count} dialogue_last={dialogue_count}; final concat keeps original order",
                )
            segment_queue = asyncio.Queue()
            for order, segment in segment_pairs:
                await segment_queue.put((order, segment))

            async def manifest_segment_worker(manifest_worker_id):
                nonlocal cache_hits, rendered_ok_count, processed_count
                while True:
                    try:
                        order, segment = segment_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    seg_id = str(segment.get("segment_id", f"seg_{order:06d}"))
                    previous_failure = previous_failed_map.get(seg_id)
                    profile = resolve_segment_tts_settings(segment, defaults)
                    should_force_preview_rescue = bool(rescue_short_segments_now and is_short_segment(segment))
                    if progress_callback:
                        progress_callback(
                            total=len(ordered_segments),
                            done=processed_count,
                            ok=rendered_ok_count,
                            failed=len(first_pass_failed),
                            status="RUNNING",
                            phase=("failed_retry" if pass_name == "failed_first" else "render"),
                            detail=f"{pass_name} worker {manifest_worker_id}: {order}/{len(ordered_segments)} {seg_id}",
                        )

                    if should_force_preview_rescue:
                        rescue_candidates.append(segment)
                        processed_count += 1
                        segment_queue.task_done()
                        continue

                    part_audio = os.path.join(tmpdir, f"segment_{order:06d}.mp3")
                    try:
                        if cache_only:
                            cache_key = make_segment_cache_key(segment, profile)
                            cached = None if force_rerender else load_cached_segment(
                                manifest_cache_dir,
                                segment,
                                cache_key,
                                cache_name=cache_name,
                                allow_remote_download=not repo_cache_only,
                            )
                            if not cached:
                                raise RuntimeError(f"cache_only missing valid cache for {seg_id}")
                            try:
                                if os.path.exists(part_audio):
                                    os.remove(part_audio)
                                os.link(cached["audio_path"], part_audio)
                            except Exception:
                                shutil.copyfile(cached["audio_path"], part_audio)
                            rendered = {
                                "cache_hit": True,
                                "audio_path": part_audio,
                                "events": cached["events"],
                                "audio_duration_sec": cached["audio_duration_sec"],
                                "profile": profile,
                            }
                        else:
                            rendered = await render_segment(
                                segment,
                                part_audio,
                                defaults,
                                manifest_cache_dir=manifest_cache_dir,
                                cache_name=cache_name,
                                upload_remote_cache=False,
                                tts_timeout_sec=tts_timeout_sec,
                                force_rerender=force_rerender,
                                repo_cache_only=repo_cache_only,
                                tts_semaphore=tts_semaphore,
                            )
                        rendered_by_segment_id[seg_id] = rendered
                        rendered_ok_count += 1
                        cache_hits += 1 if rendered["cache_hit"] else 0
                        clear_failed_segment(manifest_cache_dir, segment)
                        processed_count += 1
                        if progress_callback:
                            progress_callback(
                                total=len(ordered_segments),
                                done=processed_count,
                                ok=rendered_ok_count,
                                failed=len(first_pass_failed),
                                status="RUNNING",
                                phase=("cache_hit" if rendered["cache_hit"] else ("failed_retry" if pass_name == "failed_first" else "render")),
                                detail=("cache hit" if rendered["cache_hit"] else "rendered") + f" {seg_id} by {pass_name} worker {manifest_worker_id}",
                            )
                    except Exception as exc:
                        if _is_engine_init_failure(exc) and vieneu_fail_fast_on_engine_init():
                            if os.path.exists(part_audio):
                                try:
                                    os.remove(part_audio)
                                except Exception:
                                    pass
                            fatal_detail = f"fatal engine init failed while rendering {seg_id}; aborting manifest instead of marking every segment failed: {exc}"
                            if progress_callback:
                                progress_callback(
                                    total=len(ordered_segments),
                                    done=processed_count,
                                    ok=rendered_ok_count,
                                    failed=len(first_pass_failed),
                                    status="FAILED",
                                    phase="engine_init",
                                    detail=fatal_detail,
                                )
                            raise VieneuEngineInitFatalError(fatal_detail) from exc
                        if os.path.exists(part_audio):
                            try:
                                os.remove(part_audio)
                            except Exception:
                                pass
                        failure_class, error_type = classify_tts_failure(exc, segment)
                        save_failed_segment(manifest_cache_dir, segment, profile, exc, failure_stage=("failed_first_retry" if pass_name == "failed_first" else "normal_render"))
                        failed_item = {
                            "order": int(segment.get("order", order) or order),
                            "segment_id": seg_id,
                            "segment_type": segment.get("segment_type"),
                            "text": clean_tts_text(segment.get("text", "")),
                            "text_len": segment_text_len(segment),
                            "voice": profile["voice"],
                            "rate_pct": profile["rate_pct"],
                            "pitch_hz": profile["pitch_hz"],
                            "failure_class": failure_class,
                            "error_type": error_type,
                            "error": str(exc),
                            "debug": segment_debug_label(segment, profile),
                            "failed_in_previous_run": bool(previous_failure),
                            "pass_name": pass_name,
                        }
                        first_pass_failed.append(failed_item)
                        if failure_class == "timeout":
                            timeout_segments.append(failed_item)
                        elif failure_class == "short_or_no_audio":
                            short_or_no_audio_segments.append(failed_item)
                            if rescue_repeated_short_failures and previous_failure and previous_failure.get("failure_class") == "short_or_no_audio":
                                rescue_candidates.append(segment)
                        else:
                            other_failed_segments.append(failed_item)
                        processed_count += 1
                        if progress_callback:
                            progress_callback(
                                total=len(ordered_segments),
                                done=processed_count,
                                ok=rendered_ok_count,
                                failed=len(first_pass_failed),
                                status="RUNNING",
                                phase=("failed_retry" if pass_name == "failed_first" else "render"),
                                detail=f"failed {seg_id} ({failure_class}) by {pass_name} worker {manifest_worker_id} - continuing",
                            )
                    finally:
                        segment_queue.task_done()

            worker_tasks = [
                asyncio.create_task(manifest_segment_worker(i + 1))
                for i in range(min(segment_worker_count, max(1, len(segment_pairs))))
            ]
            await asyncio.gather(*worker_tasks)

        if previous_failed_pairs and progress_callback:
            failed_preview = ", ".join(
                f"{str(seg.get('segment_id', '?'))}@{int(seg.get('order', order) or order)}"
                for order, seg in previous_failed_pairs[:12]
            )
            progress_callback(
                total=len(ordered_segments),
                done=0,
                ok=0,
                failed=0,
                status="RUNNING",
                phase="failed_retry",
                detail=f"retrying previous failed cache segment(s) first: count={len(previous_failed_pairs)} [{failed_preview}]. Final concat will be skipped if any still fail.",
            )
        if previous_failed_pairs:
            await run_segment_pairs(previous_failed_pairs, "failed_first")

            if rescue_candidates:
                if progress_callback:
                    progress_callback(
                        total=len(ordered_segments),
                        done=processed_count,
                        ok=rendered_ok_count,
                        failed=len(first_pass_failed),
                        status="RUNNING",
                        phase="rescue",
                        detail=f"rescue previous failed segment(s) before normal/cache pass: {len(rescue_candidates)}",
                    )
                rescued_now, rescue_failed_now = await rescue_failed_segments(
                    candidates=rescue_candidates,
                    ordered_segments=ordered_segments,
                    rendered_by_segment_id=rendered_by_segment_id,
                    defaults=defaults,
                    manifest_cache_dir=manifest_cache_dir,
                    tmpdir=tmpdir,
                    cache_name=cache_name,
                    tts_timeout_sec=tts_timeout_sec,
                    tts_semaphore=tts_semaphore,
                )
                rescued_segments.extend(rescued_now)
                rescue_failed_segments_list.extend(rescue_failed_now)
                rendered_ok_count = len(rendered_by_segment_id)
                rescue_candidates = []

            previous_failed_ids = {str(segment.get("segment_id")) for _, segment in previous_failed_pairs}
            unresolved_previous_failed_ids = {
                str(item.get("segment_id")) for item in first_pass_failed
                if str(item.get("segment_id")) in previous_failed_ids
            }
            unresolved_previous_failed_ids.update(
                str(item.get("segment_id")) for item in rescue_failed_segments_list
                if str(item.get("segment_id")) in previous_failed_ids
            )
            unresolved_previous_failed_ids = {
                seg_id for seg_id in unresolved_previous_failed_ids
                if seg_id not in rendered_by_segment_id
            }
            if unresolved_previous_failed_ids:
                early_stop_failed_first = True
                if progress_callback:
                    preview = ", ".join(sorted(unresolved_previous_failed_ids)[:8])
                    progress_callback(
                        total=len(ordered_segments),
                        done=processed_count,
                        ok=rendered_ok_count,
                        failed=len(unresolved_previous_failed_ids),
                        status="INCOMPLETE",
                        phase="failed_retry",
                        detail=f"previous failed segment(s) still failed: {preview}. Skipping cache scan/final concat.",
                    )

        if retry_failed_only:
            early_stop_failed_first = True
            if progress_callback:
                progress_callback(
                    total=len(ordered_segments),
                    done=processed_count,
                    ok=rendered_ok_count,
                    failed=len(first_pass_failed),
                    status="INCOMPLETE",
                    phase="failed_retry",
                    detail="retry failed only finished; final concat intentionally skipped",
                )

        if not early_stop_failed_first:
            # Failed-first optimization: only after failed markers are gone do we download success cache.
            # This avoids pulling thousands of success segment files when a known failed segment still fails.
            success_cache_sync = {"enabled": bool(repo_cache_only), "status": "skipped", "files": 0, "cache_scope": CACHE_SCOPE_SUCCESS_ONLY}
            if repo_cache_only and not force_rerender and not cache_only:
                if progress_callback:
                    progress_callback(
                        total=len(ordered_segments), done=processed_count, ok=rendered_ok_count,
                        failed=len(first_pass_failed), status="RUNNING", phase="cache",
                        detail=f"previous failed pass clear; downloading success cache scope={CACHE_SCOPE_SUCCESS_ONLY} before normal/final pass",
                    )
                success_cache_sync = sync_remote_cache_folder_to_local(cache_name, clear_local_first=False, cache_scope=CACHE_SCOPE_SUCCESS_ONLY)
                repo_cache_sync["success_status"] = success_cache_sync.get("status")
                repo_cache_sync["success_files"] = success_cache_sync.get("files", 0)
                repo_cache_sync["initial_failed_files"] = repo_cache_sync.get("files", 0)
                cache_prune_report = prune_manifest_cache_to_current_manifest(manifest_cache_dir, ordered_segments, defaults)

            # Compact remote cache: final_parts/ can cover large success ranges, so we do
            # not need to download/render raw success/*.mp3 again. Only render gaps not
            # covered by compact parts and not already retried successfully.
            compact_covered_ids = compact_part_covered_segment_ids(manifest_cache_dir, ordered_segments)
            compact_parts_loaded_count = len(load_compact_final_part_metas(manifest_cache_dir, ordered_segments=ordered_segments))
            if compact_covered_ids:
                before_count = len(normal_segment_pairs)
                normal_segment_pairs = [
                    (order, segment) for order, segment in normal_segment_pairs
                    if str(segment.get("segment_id")) not in compact_covered_ids
                    and str(segment.get("segment_id")) not in rendered_by_segment_id
                ]
                if progress_callback:
                    progress_callback(
                        total=len(ordered_segments),
                        done=processed_count,
                        ok=rendered_ok_count,
                        failed=len(first_pass_failed),
                        status="RUNNING",
                        phase="compact_cache",
                        detail=f"compact final_parts loaded={compact_parts_loaded_count} covered_segments={len(compact_covered_ids)} skipped_normal_render={before_count - len(normal_segment_pairs)}",
                    )

            if progress_callback:
                progress_callback(
                    total=len(ordered_segments),
                    done=processed_count,
                    ok=rendered_ok_count,
                    failed=len(first_pass_failed),
                    status="RUNNING",
                    phase="render",
                    detail=f"render/build remaining gaps started | previous_failed_retry={previous_failed_retry_count} | repo_cache_failed={repo_cache_sync.get('status')} failed_files={repo_cache_sync.get('files', 0)} | compact_success_cache={repo_cache_sync.get('success_status', 'n/a')} compact_files={repo_cache_sync.get('success_files', 0)} | compact_parts={compact_parts_loaded_count} covered={len(compact_covered_ids)} | stale_removed={cache_prune_report.get('removed_files', 0)} | segment_workers={segment_worker_count}",
                )
            async def run_vieneu_native_segment_pairs(segment_pairs, pass_name):
                nonlocal cache_hits, rendered_ok_count, processed_count
                if not segment_pairs:
                    return
                ordered_for_native = [seg for _, seg in sorted(segment_pairs, key=lambda x: int(x[0] or 0))]
                pending_segments = []
                for seg in ordered_for_native:
                    seg_id = str(seg.get("segment_id"))
                    profile = resolve_segment_tts_settings(seg, defaults)
                    if not force_rerender:
                        cache_key = make_segment_cache_key(seg, profile)
                        cached = load_cached_segment(
                            manifest_cache_dir,
                            seg,
                            cache_key,
                            cache_name=cache_name,
                            allow_remote_download=not repo_cache_only,
                        ) if manifest_cache_dir else None
                    else:
                        cached = None

                    if cached:
                        part_audio = os.path.join(tmpdir, f"segment_{int(seg.get('order', 0) or 0):06d}.mp3")
                        try:
                            if os.path.exists(part_audio):
                                os.remove(part_audio)
                            os.link(cached["audio_path"], part_audio)
                        except Exception:
                            shutil.copyfile(cached["audio_path"], part_audio)
                        rendered_by_segment_id[seg_id] = {
                            "cache_hit": True,
                            "audio_path": part_audio,
                            "events": cached["events"],
                            "audio_duration_sec": cached["audio_duration_sec"],
                            "profile": profile,
                        }
                        rendered_ok_count += 1
                        cache_hits += 1
                        processed_count += 1
                        clear_failed_segment(manifest_cache_dir, seg)
                        continue

                    if cache_only:
                        exc = RuntimeError(f"cache_only missing valid cache for native infer segment {seg.get('segment_id')}")
                        save_failed_segment(manifest_cache_dir, seg, profile, exc, failure_stage="native_infer_cache_only")
                        first_pass_failed.append(seg)
                        processed_count += 1
                        continue

                    pending_segments.append(seg)

                if not pending_segments:
                    return

                batches = build_vieneu_native_segment_batches(pending_segments)
                if progress_callback:
                    progress_callback(
                        total=len(ordered_segments),
                        done=processed_count,
                        ok=rendered_ok_count,
                        failed=len(first_pass_failed),
                        status="RUNNING",
                        phase="native_infer",
                        detail=f"{pass_name}: infer_segments batches={len(batches)} uncached_segments={len(pending_segments)}",
                    )
                for batch_index, batch in enumerate(batches, start=1):
                    batch_orders = [int(seg.get("order", 0) or 0) for seg in batch]
                    batch_preview = f"{min(batch_orders or [0])}-{max(batch_orders or [0])} segments={len(batch)}"
                    if progress_callback:
                        progress_callback(
                            total=len(ordered_segments),
                            done=processed_count,
                            ok=rendered_ok_count,
                            failed=len(first_pass_failed),
                            status="RUNNING",
                            phase="native_infer",
                            detail=f"{pass_name}: render infer_segments batch {batch_index}/{len(batches)} {batch_preview}",
                        )
                    try:
                        if tts_semaphore is not None:
                            async with tts_semaphore:
                                rendered_map = await render_vieneu_native_segment_batch(
                                    batch,
                                    defaults,
                                    tmpdir,
                                    batch_index,
                                    manifest_cache_dir=manifest_cache_dir,
                                    cache_name=cache_name,
                                    tts_timeout_sec=tts_timeout_sec,
                                )
                        else:
                            rendered_map = await render_vieneu_native_segment_batch(
                                batch,
                                defaults,
                                tmpdir,
                                batch_index,
                                manifest_cache_dir=manifest_cache_dir,
                                cache_name=cache_name,
                                tts_timeout_sec=tts_timeout_sec,
                            )
                        for seg in batch:
                            seg_id = str(seg.get("segment_id"))
                            rendered = rendered_map.get(seg_id)
                            if not rendered:
                                raise RuntimeError(f"infer_segments batch did not return split audio for {seg_id}")
                            rendered_by_segment_id[seg_id] = rendered
                            rendered_ok_count += 1
                            processed_count += 1
                            clear_failed_segment(manifest_cache_dir, seg)
                    except Exception as exc:
                        for seg in batch:
                            profile = resolve_segment_tts_settings(seg, defaults)
                            save_failed_segment(manifest_cache_dir, seg, profile, exc, failure_stage="native_infer_segments_render")
                            first_pass_failed.append(seg)
                            failure_class, _ = classify_tts_failure(exc, seg)
                            if failure_class == "timeout":
                                timeout_segments.append(seg)
                            elif failure_class == "short_or_no_audio":
                                short_or_no_audio_segments.append(seg)
                                rescue_candidates.append(seg)
                            else:
                                other_failed_segments.append(seg)
                            processed_count += 1
                        if progress_callback:
                            progress_callback(
                                total=len(ordered_segments),
                                done=processed_count,
                                ok=rendered_ok_count,
                                failed=len(first_pass_failed),
                                status="RUNNING",
                                phase="native_infer",
                                detail=f"failed infer_segments batch {batch_preview}: {str(exc)[:180]}",
                            )
            async def run_shared_ref_pairs(segment_pairs, pass_name):
                nonlocal cache_hits, rendered_ok_count, processed_count
                if not segment_pairs:
                    return
                ordered_for_groups = [seg for _, seg in sorted(segment_pairs, key=lambda x: int(x[0] or 0))]
                groups = build_shared_ref_segment_groups(ordered_for_groups, defaults, segment_infer_strategy)
                if progress_callback:
                    progress_callback(
                        total=len(ordered_segments),
                        done=processed_count,
                        ok=rendered_ok_count,
                        failed=len(first_pass_failed),
                        status="RUNNING",
                        phase="shared_ref",
                        detail=f"{pass_name}: segment_infer_strategy={segment_infer_strategy} groups={len(groups)}; SRT remains original segments",
                    )
                for group_index, group in enumerate(groups, start=1):
                    group_ids = [str(seg.get("segment_id")) for seg in group]
                    group_orders = [int(seg.get("order", 0) or 0) for seg in group]
                    group_preview = f"{min(group_orders or [0])}-{max(group_orders or [0])} segments={len(group)}"
                    if progress_callback:
                        progress_callback(
                            total=len(ordered_segments),
                            done=processed_count,
                            ok=rendered_ok_count,
                            failed=len(first_pass_failed),
                            status="RUNNING",
                            phase="shared_ref",
                            detail=f"{pass_name}: render group {group_index}/{len(groups)} {group_preview} ids={','.join(group_ids[:4])}{'...' if len(group_ids)>4 else ''}",
                        )
                    # Per-segment cache is still valid because final/SRT is still per segment.
                    cached_map = {}
                    cache_missing = False
                    if not force_rerender:
                        for seg in group:
                            seg_id = str(seg.get("segment_id"))
                            profile = resolve_segment_tts_settings(seg, defaults)
                            cache_key = make_segment_cache_key(seg, profile)
                            cached = load_cached_segment(
                                manifest_cache_dir,
                                seg,
                                cache_key,
                                cache_name=cache_name,
                                allow_remote_download=not repo_cache_only,
                            ) if manifest_cache_dir else None
                            if not cached:
                                cache_missing = True
                                break
                            part_audio = os.path.join(tmpdir, f"segment_{int(seg.get('order', 0) or 0):06d}.mp3")
                            try:
                                if os.path.exists(part_audio):
                                    os.remove(part_audio)
                                os.link(cached["audio_path"], part_audio)
                            except Exception:
                                shutil.copyfile(cached["audio_path"], part_audio)
                            cached_map[seg_id] = {
                                "cache_hit": True,
                                "audio_path": part_audio,
                                "events": cached["events"],
                                "audio_duration_sec": cached["audio_duration_sec"],
                                "profile": profile,
                            }
                    else:
                        cache_missing = True

                    if cached_map and not cache_missing:
                        for seg in group:
                            seg_id = str(seg.get("segment_id"))
                            rendered_by_segment_id[seg_id] = cached_map[seg_id]
                            rendered_ok_count += 1
                            cache_hits += 1
                            processed_count += 1
                            clear_failed_segment(manifest_cache_dir, seg)
                        continue

                    if cache_only:
                        for seg in group:
                            profile = resolve_segment_tts_settings(seg, defaults)
                            exc = RuntimeError(f"cache_only missing valid cache for shared-ref segment {seg.get('segment_id')}")
                            save_failed_segment(manifest_cache_dir, seg, profile, exc, failure_stage="shared_ref_cache_only")
                            first_pass_failed.append(seg)
                            processed_count += 1
                        continue

                    try:
                        rendered_map = await render_shared_ref_segment_group(
                            group,
                            defaults,
                            tmpdir,
                            group_index,
                            manifest_cache_dir=manifest_cache_dir,
                            cache_name=cache_name,
                            tts_timeout_sec=tts_timeout_sec,
                            force_rerender=force_rerender,
                            repo_cache_only=repo_cache_only,
                            tts_semaphore=tts_semaphore,
                        )
                        for seg in group:
                            seg_id = str(seg.get("segment_id"))
                            rendered = rendered_map.get(seg_id)
                            if not rendered:
                                raise RuntimeError(f"shared-ref group did not return split audio for {seg_id}")
                            rendered_by_segment_id[seg_id] = rendered
                            rendered_ok_count += 1
                            processed_count += 1
                            clear_failed_segment(manifest_cache_dir, seg)
                    except Exception as exc:
                        for seg in group:
                            profile = resolve_segment_tts_settings(seg, defaults)
                            save_failed_segment(manifest_cache_dir, seg, profile, exc, failure_stage="shared_ref_group_render")
                            first_pass_failed.append(seg)
                            failure_class, _ = classify_tts_failure(exc, seg)
                            if failure_class == "timeout":
                                timeout_segments.append(seg)
                            elif failure_class == "short_or_no_audio":
                                short_or_no_audio_segments.append(seg)
                                rescue_candidates.append(seg)
                            else:
                                other_failed_segments.append(seg)
                            processed_count += 1
                        if progress_callback:
                            progress_callback(
                                total=len(ordered_segments),
                                done=processed_count,
                                ok=rendered_ok_count,
                                failed=len(first_pass_failed),
                                status="RUNNING",
                                phase="shared_ref",
                                detail=f"failed shared-ref group {group_preview}: {str(exc)[:180]}",
                            )

            if segment_infer_strategy == SEGMENT_INFER_STRATEGY_NATIVE_VIENEU:
                await run_vieneu_native_segment_pairs(normal_segment_pairs, "normal")
            elif segment_infer_strategy != SEGMENT_INFER_STRATEGY_NONE:
                await run_shared_ref_pairs(normal_segment_pairs, "normal")
            else:
                await run_segment_pairs(normal_segment_pairs, "normal")

            if rescue_candidates:
                if progress_callback:
                    progress_callback(
                        total=len(ordered_segments),
                        done=len(ordered_segments),
                        ok=rendered_ok_count,
                        failed=len(first_pass_failed),
                        status="RUNNING",
                        phase="rescue",
                        detail=f"rescue grouping {len(rescue_candidates)} segment(s)",
                    )
                rescued_now, rescue_failed_now = await rescue_failed_segments(
                    candidates=rescue_candidates,
                    ordered_segments=ordered_segments,
                    rendered_by_segment_id=rendered_by_segment_id,
                    defaults=defaults,
                    manifest_cache_dir=manifest_cache_dir,
                    tmpdir=tmpdir,
                    cache_name=cache_name,
                    tts_timeout_sec=tts_timeout_sec,
                    tts_semaphore=tts_semaphore,
                )
                rescued_segments.extend(rescued_now)
                rescue_failed_segments_list.extend(rescue_failed_now)
                rendered_ok_count = len(rendered_by_segment_id)

        unresolved_failed_ids = {str(item.get("segment_id")) for item in first_pass_failed}
        unresolved_failed_ids.update(str(item.get("segment_id")) for item in rescue_failed_segments_list)
        unresolved_failed_ids = {seg_id for seg_id in unresolved_failed_ids if seg_id not in rendered_by_segment_id}
        final_failed_segments = [item for item in first_pass_failed if str(item.get("segment_id")) in unresolved_failed_ids]
        # Add rescue failure details without duplicating normal failed entries too much.
        for item in rescue_failed_segments_list:
            if str(item.get("segment_id")) in unresolved_failed_ids:
                matched = False
                for target in final_failed_segments:
                    if str(target.get("segment_id")) == str(item.get("segment_id")):
                        target.setdefault("rescue_errors", []).append(item)
                        matched = True
                        break
                if not matched:
                    seg = next((s for s in ordered_segments if str(s.get("segment_id")) == str(item.get("segment_id"))), {})
                    profile = resolve_segment_tts_settings(seg, defaults) if seg else {}
                    final_failed_segments.append({
                        "order": item.get("order") or seg.get("order"),
                        "segment_id": item.get("segment_id"),
                        "segment_type": seg.get("segment_type"),
                        "text": clean_tts_text(seg.get("text", "")),
                        "text_len": segment_text_len(seg) if seg else 0,
                        "voice": profile.get("voice"),
                        "rate_pct": profile.get("rate_pct"),
                        "pitch_hz": profile.get("pitch_hz"),
                        "failure_class": "short_or_no_audio",
                        "error_type": "rescue_failed",
                        "error": item.get("error"),
                        "rescue_errors": [item],
                    })

        covered_for_final_ids = set(rendered_by_segment_id.keys()) | set(compact_covered_ids)
        complete = len(covered_for_final_ids) == len(ordered_segments) and not final_failed_segments
        local_subtitle = None
        final_audio = None
        concat_meta = {"method": None}
        final_audio_duration_sec = 0.0
        global_events = []
        part_files = []
        segment_events = []

        success_final_parts = []
        compact_cache_report = {"parts": [], "raw_success_deleted": 0, "raw_success_kept": 0}
        if rendered_by_segment_id:
            # Pure-Colab compact policy: build final_parts even when some segments
            # failed, then prune raw success/*.mp3/json. If retry later fixes the
            # failed gaps, final concat can reuse these compact parts immediately.
            compact_cache_report = compact_manifest_success_cache(
                manifest_cache_dir,
                ordered_segments,
                rendered_by_segment_id,
                defaults,
                tmpdir,
                timeline_mode,
                progress_callback=progress_callback,
                total=len(ordered_segments),
                failed_count=len(final_failed_segments),
                phase_label=("part_cache" if complete else "compact_cache"),
            )
            success_final_parts = compact_cache_report.get("parts") or []

        if final_failed_segments and progress_callback:
            preview = ", ".join(f"{item.get('segment_id')}@{item.get('order')}" for item in final_failed_segments[:8])
            progress_callback(
                total=len(ordered_segments),
                done=len(ordered_segments),
                ok=len(rendered_by_segment_id),
                failed=len(final_failed_segments),
                status="INCOMPLETE",
                phase="done",
                detail=(
                    f"render pass finished with failed segment(s): {preview}. "
                    f"Compact final_parts saved; raw success pruned={compact_cache_report.get('raw_success_deleted', 0)}. "
                    "Skipping final concat until failed segments are fixed."
                ),
            )

        final_part_cache_used = []
        if complete:
            if progress_callback:
                progress_callback(
                    total=len(ordered_segments),
                    done=len(ordered_segments),
                    ok=len(rendered_by_segment_id),
                    failed=len(final_failed_segments),
                    status="RUNNING",
                    phase="finalize",
                    detail=f"building final from cached success parts + subtitle timeline={timeline_mode}",
                )
            part_files, segment_events, final_part_cache_used = build_final_from_compact_parts_or_segments(
                ordered_segments,
                rendered_by_segment_id,
                defaults,
                manifest_cache_dir,
                tmpdir,
                timeline_mode,
                progress_callback=progress_callback,
                total=len(ordered_segments),
                failed_count=len(final_failed_segments),
            )

            if progress_callback:
                progress_callback(
                    total=len(ordered_segments),
                    done=len(ordered_segments),
                    ok=len(rendered_by_segment_id),
                    failed=len(final_failed_segments),
                    status="RUNNING",
                    phase="concat",
                    detail=f"ffmpeg concat final using {len(part_files)} input part(s); success_part_cache={len(final_part_cache_used)}",
                )
            final_audio = os.path.join(tmpdir, audio_name)
            concat_meta = concat_audio_files(part_files, final_audio)
            final_audio_duration_sec = get_audio_duration_sec(final_audio)
            global_events = build_global_subtitles(segment_events)
            if subtitle_name:
                local_subtitle = os.path.join(tmpdir, subtitle_name)
                write_subtitle_file(global_events, local_subtitle, subtitle_format)

        if progress_callback:
            progress_callback(
                total=len(ordered_segments),
                done=len(ordered_segments),
                ok=len(rendered_by_segment_id),
                failed=len(final_failed_segments),
                status="RUNNING",
                phase="cache_index",
                detail="writing cache index + render report",
            )
        cache_index = write_manifest_cache_index(cache_name, manifest, manifest_cache_dir, defaults=defaults)
        cache_index_summary = summarize_cache_index(cache_index)

        local_report = os.path.join(tmpdir, report_name if complete else f"{base}_INCOMPLETE_render_report.json")
        report = {
            "mode": DEFAULT_MANIFEST_MODE,
            "schema_version": manifest.get("schema_version"),
            "source_file": source_file,
            "story_prefix": story_prefix,
            "cache_name": cache_name,
            "repo_cache_only": bool(repo_cache_only),
            "repo_cache_sync": repo_cache_sync,
            "cache_prune_report": cache_prune_report,
            "segment_worker_count": int(segment_worker_count),
            "previous_failed_retry_count": int(previous_failed_retry_count),
            "early_stop_failed_first": bool(early_stop_failed_first),
            "cache_only": bool(cache_only),
            "retry_failed_only": bool(retry_failed_only),
            "timeline_mode": timeline_mode,
            "cache_index": cache_index_summary,
            "segment_cache_dir": str(manifest_cache_dir.resolve()),
            "remote_cache_prefix": build_remote_cache_prefix(cache_name),
            "output_subdir": build_output_subdir(source_file),
            "complete": complete,
            "segments_total": len(ordered_segments),
            "segments_rendered_ok": len(rendered_by_segment_id),
            "segments_compact_part_covered": len(compact_covered_ids),
            "compact_parts_loaded": int(compact_parts_loaded_count),
            "segments_final_covered_total": len(set(rendered_by_segment_id.keys()) | set(compact_covered_ids)),
            "segments_failed": len(final_failed_segments),
            "segments_rescued": len(rescued_segments),
            "failed_segments": final_failed_segments,
            "rescued_segments": rescued_segments,
            "timeout_segments": timeout_segments,
            "short_or_no_audio_segments": short_or_no_audio_segments,
            "other_failed_segments": other_failed_segments,
            "cache_hits": cache_hits,
            "cache_misses": max(0, len(rendered_by_segment_id) - cache_hits - len(rescued_segments)),
            "concat_method": concat_meta["method"],
            "ffmpeg_available": ffmpeg_available,
            "silence_supported": True,
            "silence_cache_enabled": True,
            "final_part_cache_enabled": True,
            "prebuilt_success_parts": len(success_final_parts),
            "compact_cache_report": compact_cache_report,
            "final_part_cache_used": final_part_cache_used,
            "subtitle_events_total": len(global_events),
            "subtitle_format": subtitle_format,
            "force_rerender": bool(force_rerender),
            "manual_group_rescue_enabled": bool(rescue_repeated_short_failures),
            "preview_manual_group_short_segments_now": bool(rescue_short_segments_now),
            "timeline_method": ("segment_order_concat_json_text_subtitles_fast_no_silencedetect" if timeline_mode == TIMELINE_FAST else "segment_order_concat_json_text_subtitles_accurate_silencedetect") if complete else "not_built_incomplete_manifest",
            "subtitle_last_event_end_sec": round((global_events[-1]["end"] if global_events else 0.0), 3),
            "final_audio_duration_sec": final_audio_duration_sec,
            "warnings": (["Manifest incomplete: previous failed cache segment(s) still failed, so remaining cache scan/final concat were skipped."] if early_stop_failed_first else ["Manifest incomplete: final audio/subtitle were not built because one or more segments failed."]) if not complete else [],
        }
        write_render_report(local_report, report)
        if progress_callback:
            progress_callback(
                total=len(ordered_segments),
                done=len(ordered_segments),
                ok=len(rendered_by_segment_id),
                failed=len(final_failed_segments),
                status="COMPLETE" if complete else "INCOMPLETE",
                phase="done",
                detail=(f"final audio/subtitle built | cache_index ready={cache_index_summary['cache_ready']}/{cache_index_summary['segments_total']}" if complete else f"cached only; retry failed segments later | cache_index ready={cache_index_summary['cache_ready']}/{cache_index_summary['segments_total']}"),
            )
        return {
            "story_prefix": story_prefix,
            "audio_name": audio_name,
            "subtitle_name": subtitle_name,
            "report_name": report_name if complete else os.path.basename(local_report),
            "final_audio": final_audio,
            "local_subtitle": local_subtitle,
            "local_report": local_report,
            "report": report,
            "complete": complete,
        }
    finally:
        if cleanup_dir is not None:
            cleanup_dir.cleanup()

    
    
async def preview_manifest_text(manifest_text, subtitle_format, force_short_rescue_grouping=False):
    preview_dir = tempfile.mkdtemp(prefix="vieneu_tts_preview_")
    try:
        manifest = load_manifest_from_text(manifest_text)
        result = await render_manifest_to_outputs(
            manifest=manifest,
            subtitle_format=subtitle_format,
            upload=False,
            source_name=manifest.get("source_file") or "preview_segments.txt",
            work_dir=preview_dir,
            rescue_repeated_short_failures=bool(force_short_rescue_grouping),
            rescue_short_segments_now=bool(force_short_rescue_grouping),
            repo_cache_only=False,
            segment_worker_count=1,
        )
        summary = (
            f"Preview ready: {result['audio_name']} | "
            f"segments={result['report']['segments_total']} | "
            f"cache_hits={result['report']['cache_hits']} | "
            f"rescued={result['report'].get('segments_rescued', 0)}"
        )
        return result["final_audio"], summary, preview_dir
    except Exception as exc:
        shutil.rmtree(preview_dir, ignore_errors=True)
        raise gr.Error(str(exc))


def cleanup_preview_dir(preview_dir):
    if isinstance(preview_dir, str) and preview_dir.strip():
        shutil.rmtree(preview_dir, ignore_errors=True)
    return None


async def process_one_file(file, mode, voice, rate, pitch, subtitle_format, upload, output_root, tts_timeout_sec=None, force_rerender=False, progress_callback=None, rescue_repeated_short_failures=False, repo_cache_only=True, force_commit_cache_now=False, segment_worker_count=SEGMENT_RENDER_WORKERS, timeline_mode=DEFAULT_TIMELINE_MODE, tts_semaphore=None):
    validate_uploaded_file_for_mode(file, mode)
    safe_name = sanitize_filename(os.path.basename(file.name))
    work_dir = os.path.join(output_root, safe_name)
    os.makedirs(work_dir, exist_ok=True)
    if mode == DEFAULT_MANIFEST_MODE:
        return await process_manifest_file(file, subtitle_format, upload=upload, work_dir=work_dir, tts_timeout_sec=tts_timeout_sec, force_rerender=force_rerender, progress_callback=progress_callback, rescue_repeated_short_failures=rescue_repeated_short_failures, repo_cache_only=repo_cache_only, force_commit_cache_now=force_commit_cache_now, segment_worker_count=segment_worker_count, timeline_mode=timeline_mode, tts_semaphore=tts_semaphore)
    if progress_callback:
        progress_callback(total=1, done=0, ok=0, failed=0, status="RUNNING", detail="plain text rendering")
    result = await process_plain_text_file(file, voice, rate, pitch, subtitle_format, upload=upload, work_dir=work_dir, tts_timeout_sec=tts_timeout_sec)
    if progress_callback:
        progress_callback(total=1, done=1, ok=1, failed=0, status="COMPLETE", detail="plain text done")
    return result


async def worker(worker_id, queue, logs, generated_files, upload_entries, progress_status, mode, voice, rate, pitch, subtitle_format, upload, output_root, tts_timeout_sec=None, force_rerender=False, rescue_repeated_short_failures=False, repo_cache_only=True, force_commit_cache_now=False, segment_worker_count=SEGMENT_RENDER_WORKERS, timeline_mode=DEFAULT_TIMELINE_MODE, tts_semaphore=None):
    while not queue.empty():
        try:
            file = await queue.get()
        except Exception:
            return
        name = os.path.basename(file.name)
        logs.append(f"Worker {worker_id} handling {name}")
        update_progress_item(progress_status, file.name, status="RUNNING", phase="start", detail=f"worker {worker_id} starting", worker=worker_id)

        def progress_callback(**updates):
            update_progress_item(progress_status, file.name, **updates)

        try:
            result = await process_one_file(file, mode, voice, rate, pitch, subtitle_format, upload, output_root, tts_timeout_sec=tts_timeout_sec, force_rerender=force_rerender, progress_callback=progress_callback, rescue_repeated_short_failures=rescue_repeated_short_failures, repo_cache_only=repo_cache_only, force_commit_cache_now=force_commit_cache_now, segment_worker_count=segment_worker_count, timeline_mode=timeline_mode, tts_semaphore=tts_semaphore)
            if isinstance(result, dict):
                logs.append(result.get("message", "Done."))
                generated_files.extend(result.get("files") or [])
                upload_entries.extend(result.get("upload_entries") or [])
            else:
                logs.append(str(result))
            current_item = progress_status.get(file.name, {}) if progress_status else {}
            final_status = current_item.get("status") or "COMPLETE"
            try:
                if int(current_item.get("failed", 0) or 0) > 0:
                    final_status = "INCOMPLETE"
            except Exception:
                pass
            if final_status == "INCOMPLETE":
                update_progress_item(progress_status, file.name, status="INCOMPLETE", phase="done", detail="cached only; retry failed segments later")
            elif final_status != "FAILED":
                update_progress_item(progress_status, file.name, status="COMPLETE", phase="done", detail="done")
        except Exception as exc:
            logs.append(f"Failed {name}: {exc}")
            current_item = progress_status.get(file.name, {}) if progress_status else {}
            update_progress_item(
                progress_status,
                file.name,
                done=current_item.get("done", 0),
                total=current_item.get("total", 0),
                status="FAILED",
                phase="failed",
                detail=str(exc)[:160],
            )
            if upload and mode == DEFAULT_MANIFEST_MODE:
                partial_entries = build_manifest_cache_upload_entries(name)
                if partial_entries:
                    upload_entries.extend(partial_entries)
                    logs.append(f"Staged partial cache for failed file {name}: {len(partial_entries)} cache files. Final audio/subtitle was NOT staged.")
        queue.task_done()



def normalize_tts_timeout(value):
    timeout = coerce_int(value, SEGMENT_TTS_TIMEOUT_SEC)
    if timeout is None:
        timeout = SEGMENT_TTS_TIMEOUT_SEC
    return max(30, min(int(timeout), 300))


def normalize_sync_interval(value):
    minutes = coerce_int(value, 60)
    if minutes is None:
        minutes = 60
    return max(0, min(int(minutes), 24 * 60))


def load_sync_state():
    try:
        if SYNC_STATE_PATH.exists():
            with open(SYNC_STATE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
    except Exception:
        pass
    return {"last_sync_ts": 0}


def save_sync_state(data):
    PENDING_UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    with open(SYNC_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def pending_upload_file_count():
    if not PENDING_UPLOAD_ROOT.exists():
        return 0
    return sum(1 for p in PENDING_UPLOAD_ROOT.rglob("*") if p.is_file() and p.name != SYNC_STATE_PATH.name)


def should_sync_pending_uploads(sync_interval_minutes):
    file_count = pending_upload_file_count()
    if file_count <= 0:
        return False, "No pending files to sync."
    if int(sync_interval_minutes or 0) <= 0:
        return True, f"Sync interval is 0, syncing {file_count} pending files now."
    state = load_sync_state()
    last_sync_ts = float(state.get("last_sync_ts", 0) or 0)
    elapsed_sec = time.time() - last_sync_ts
    required_sec = int(sync_interval_minutes) * 60
    if last_sync_ts <= 0 or elapsed_sec >= required_sec:
        elapsed_min = int(elapsed_sec // 60) if last_sync_ts > 0 else "never synced"
        return True, f"Sync interval reached ({elapsed_min} min elapsed). Syncing {file_count} pending files now."
    remaining_min = max(1, int((required_sec - elapsed_sec + 59) // 60))
    return False, f"Sync interval not reached. Pending files={file_count}. Next auto sync in about {remaining_min} min."


def upload_pending_uploads_once(commit_message=None):
    file_count = pending_upload_file_count()
    if file_count <= 0:
        return "No pending files to upload."
    message = upload_staged_folder_once(
        str(PENDING_UPLOAD_ROOT),
        commit_message=commit_message or f"Sync pending Edge TTS uploads ({file_count} files)",
    )
    # Only delete pending files after upload_folder succeeds.
    for item in list(PENDING_UPLOAD_ROOT.iterdir()):
        if item.name == SYNC_STATE_PATH.name:
            continue
        if item.is_dir():
            shutil.rmtree(item, ignore_errors=True)
        elif item.is_file():
            item.unlink(missing_ok=True)
    save_sync_state({"last_sync_ts": time.time()})
    return f"{message} Uploaded {file_count} pending files and cleared pending folder."


def manual_sync_pending_uploads():
    try:
        return upload_pending_uploads_once(commit_message="Manual sync pending Edge TTS uploads")
    except Exception as exc:
        return f"Manual sync failed: {exc}"


def build_manifest_cache_upload_entries(manifest_filename):
    cache_name = build_manifest_cache_name(manifest_filename)
    manifest_cache_dir = build_manifest_cache_dir(cache_name)
    remote_cache_prefix = build_remote_cache_prefix(cache_name)
    entries = []
    if manifest_cache_dir.exists():
        for cache_file, rel_cache in iter_uploadable_cache_files(manifest_cache_dir):
            entries.append((str(cache_file), f"{remote_cache_prefix}/{rel_cache}"))
    return entries

def normalize_worker_count(value, default_value):
    worker_count = coerce_int(value, default_value)
    if worker_count is None:
        worker_count = default_value
    return max(1, min(int(worker_count), 16))


PROGRESS_BAR_WIDTH = 24


def make_progress_bar(done, total, width=PROGRESS_BAR_WIDTH):
    try:
        done = max(0, int(done or 0))
        total = max(0, int(total or 0))
    except Exception:
        done, total = 0, 0
    if total <= 0:
        return "░" * width
    done = min(done, total)
    filled = int(round((done / total) * width))
    filled = max(0, min(width, filled))
    return "█" * filled + "░" * (width - filled)


def update_progress_item(progress_status, key, **updates):
    if progress_status is None:
        return
    item = progress_status.setdefault(key, {"display": os.path.basename(str(key)), "done": 0, "total": 0, "ok": 0, "failed": 0, "status": "PENDING", "detail": "waiting", "updated_at": time.time()})
    item.update({k: v for k, v in updates.items() if v is not None})
    item["updated_at"] = time.time()
    try:
        if int(item.get("failed", 0) or 0) > 0 and str(item.get("status")) == "COMPLETE":
            item["status"] = "INCOMPLETE"
    except Exception:
        pass


def render_progress_status(progress_status):
    if not progress_status:
        return ""
    lines = ["Progress:"]
    items = sorted(progress_status.values(), key=lambda item: int(item.get("index", 999999)))
    for item in items:
        display = item.get("display", "file")
        done = int(item.get("done", 0) or 0)
        total = int(item.get("total", 0) or 0)
        ok = int(item.get("ok", 0) or 0)
        failed = int(item.get("failed", 0) or 0)
        status = str(item.get("status", "PENDING"))
        phase = str(item.get("phase", "")).strip()
        detail = str(item.get("detail", "")).strip()
        percent = f"{(done / total * 100):5.1f}%" if total else "  0.0%"
        bar = make_progress_bar(done, total)
        counter = f"{done}/{total}" if total else "0/?"
        suffix = f" | ok={ok} failed={failed}" if total else ""
        phase_part = f" | phase={phase}" if phase else ""
        age_sec = int(time.time() - float(item.get("updated_at", time.time()) or time.time()))
        age_part = f" | last_update={age_sec}s" if age_sec >= 60 and status == "RUNNING" else ""
        detail_part = f" | {detail}" if detail else ""
        lines.append(f"{display}: [{bar}] {percent} {counter} | {status}{suffix}{phase_part}{age_part}{detail_part}")
    return "\n".join(lines)


def render_batch_summary(progress_status):
    if not progress_status:
        return ""
    total = len(progress_status)
    status_counts = {}
    phase_counts = {}
    done_segments = 0
    total_segments = 0
    ok_segments = 0
    failed_segments = 0
    for item in progress_status.values():
        status = str(item.get("status", "PENDING"))
        phase = str(item.get("phase", "waiting") or "waiting")
        status_counts[status] = status_counts.get(status, 0) + 1
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
        done_segments += int(item.get("done", 0) or 0)
        total_segments += int(item.get("total", 0) or 0)
        ok_segments += int(item.get("ok", 0) or 0)
        failed_segments += int(item.get("failed", 0) or 0)
    status_part = ", ".join(f"{k.lower()}={v}" for k, v in sorted(status_counts.items()))
    phase_part = ", ".join(f"{k}={v}" for k, v in sorted(phase_counts.items()))
    segment_part = f"segments={done_segments}/{total_segments} ok={ok_segments} failed={failed_segments}" if total_segments else "segments=0/?"
    return f"Batch summary: files={total} | {status_part} | {segment_part} | phases: {phase_part}"


def build_log_output(logs, progress_status=None):
    base = "\n".join(logs)
    summary_block = render_batch_summary(progress_status)
    progress_block = render_progress_status(progress_status)
    blocks = [base]
    if summary_block:
        blocks.append(summary_block)
    if progress_block:
        blocks.append(progress_block)
    return "\n\n".join(block for block in blocks if block)


async def batch_tts(files, mode, voice_dropdown, rate, pitch, subtitle_format, worker_count_input, timeline_mode_input, upload_outputs, tts_timeout_sec_input, sync_interval_minutes_input, force_rerender_input, rescue_repeated_short_failures_input, repo_cache_only_input, force_commit_cache_now_input):
    if not files:
        yield "No files selected.", []
        return
    voice = voice_dropdown.split(" - ")[0]
    default_workers = MANIFEST_WORKERS if mode == DEFAULT_MANIFEST_MODE else WORKERS
    configured_worker_count = normalize_worker_count(worker_count_input, default_workers)
    worker_count = min(configured_worker_count, len(files))
    upload = bool(upload_outputs)
    tts_timeout_sec = normalize_tts_timeout(tts_timeout_sec_input)
    sync_interval_minutes = normalize_sync_interval(sync_interval_minutes_input)
    force_rerender = bool(force_rerender_input)
    rescue_repeated_short_failures = bool(rescue_repeated_short_failures_input)
    repo_cache_only = bool(repo_cache_only_input)
    force_commit_cache_now = bool(force_commit_cache_now_input)
    # One UI slider is the GLOBAL Edge TTS concurrency cap.
    # File workers may run multiple manifests, and each manifest may have segment workers,
    # but all actual Edge TTS calls share this semaphore, so they never multiply into N x M calls.
    segment_worker_count = configured_worker_count
    global_tts_semaphore = AdaptiveTTSLimiter(configured_worker_count, step=None)
    timeline_mode = normalize_timeline_mode(timeline_mode_input)
    output_root = tempfile.mkdtemp(prefix="vieneu_tts_batch_outputs_")
    queue = asyncio.Queue()
    progress_status = {}
    for idx, file in enumerate(files, start=1):
        await queue.put(file)
        progress_status[file.name] = {
            "index": idx,
            "display": os.path.basename(file.name),
            "done": 0,
            "total": 0,
            "ok": 0,
            "failed": 0,
            "status": "PENDING",
            "phase": "waiting",
            "detail": "waiting",
        }
    logs = [
        f"vieneu adapter version: {getattr(edge_tts, '__version__', 'unavailable')}",
        f"Mode: {mode}",
        f"Subtitle format: {subtitle_format}",
        f"Files received: {len(files)}",
        f"File workers: {worker_count}",
        f"Manifest segment workers per active manifest: {segment_worker_count}",
        f"Global VieNeu TTS concurrency cap: {configured_worker_count} total call(s)",
        f"Timeline mode: {timeline_mode} ({'skip silencedetect for speed' if timeline_mode == TIMELINE_FAST else 'run silencedetect for accuracy'})",
        f"Upload to Hugging Face Dataset: {'yes' if upload else 'no'}",
        f"TTS timeout per segment/chunk: {tts_timeout_sec}s",
        f"Sync interval: {sync_interval_minutes} minute(s) (periodic while running; always final-syncs at batch end)",
        f"Force rerender / clear cache first: {'yes' if force_rerender else 'no'}",
        f"Repo cache only / refresh local cache from Dataset first: {'yes' if repo_cache_only else 'no'}",
        f"Force commit manifest cache folder immediately: {'yes' if force_commit_cache_now else 'no'}",
        f"Manual group/rescue repeated short/no-audio failures: {'yes' if rescue_repeated_short_failures else 'no'}",
        "Failed segment behavior: continue rendering later segments; only complete files build final audio/subtitle",
        "Progress display: one bar per file, updated by segment order",
        f"Subtitle timeline: ffprobe actual audio duration + pause_after_ms",
        f"Local output dir: {output_root}",
    ]
    if mode == DEFAULT_MANIFEST_MODE and worker_count > 1:
        logs.append(
            "Manifest mode is running with parallel file/segment workers. If VieNeu TTS drops responses, lower Worker count."
        )
    if upload and not HF_TOKEN:
        logs.append(
            "Warning: HF_TOKEN is empty. Upload will fail unless your runtime is already authenticated. "
            "Turn off upload to render downloadable files only."
        )
    yield build_log_output(logs, progress_status), []

    if upload and force_rerender and mode == DEFAULT_MANIFEST_MODE:
        remote_cache_prefixes = collect_remote_cache_prefixes_for_files(files, mode)
        logs.append(
            "Force rerender is ON: clearing remote dataset chunkcache before rendering: "
            + (", ".join(remote_cache_prefixes) if remote_cache_prefixes else "no manifest cache prefixes")
        )
        yield build_log_output(logs, progress_status), []
        try:
            clear_msg = clear_remote_chunkcache_prefixes(
                remote_cache_prefixes,
                commit_message=f"Force rerender: clear VieNeu TTS chunkcache for {len(remote_cache_prefixes)} manifest(s)",
            )
            logs.append(clear_msg)
        except Exception as exc:
            logs.append(
                "Remote chunkcache clear failed: "
                f"{exc}. Local force rerender will still continue, but the Dataset may keep old cache until a later clean commit."
            )
        yield build_log_output(logs, progress_status), []

    logs.append("Batch run started: workers are now taking files from queue.")
    yield build_log_output(logs, progress_status), []

    generated_files = []
    batch_upload_entries = []
    workers = [
        asyncio.create_task(
            worker(
                worker_id=i + 1,
                queue=queue,
                logs=logs,
                generated_files=generated_files,
                upload_entries=batch_upload_entries,
                progress_status=progress_status,
                mode=mode,
                voice=voice,
                rate=rate,
                pitch=pitch,
                subtitle_format=subtitle_format,
                upload=upload,
                output_root=output_root,
                tts_timeout_sec=tts_timeout_sec,
                force_rerender=force_rerender,
                rescue_repeated_short_failures=rescue_repeated_short_failures,
                repo_cache_only=repo_cache_only,
                force_commit_cache_now=force_commit_cache_now,
                segment_worker_count=segment_worker_count,
                timeline_mode=timeline_mode,
                tts_semaphore=global_tts_semaphore,
            )
        )
        for i in range(worker_count)
    ]
    last_periodic_sync_ts = time.time()
    while not all(task.done() for task in workers):
        if upload and sync_interval_minutes > 0:
            elapsed_sec = time.time() - last_periodic_sync_ts
            if elapsed_sec >= sync_interval_minutes * 60:
                entries_to_stage = drain_upload_entries(batch_upload_entries)
                if entries_to_stage:
                    staged_files = stage_upload_entries(entries_to_stage, str(PENDING_UPLOAD_ROOT))
                    logs.append(f"Periodic sync: moved {len(staged_files)} files into pending upload folder: {PENDING_UPLOAD_ROOT.resolve()}")
                if pending_upload_file_count() > 0:
                    try:
                        upload_message = upload_pending_uploads_once(
                            commit_message=f"Periodic VieNeu TTS sync every {sync_interval_minutes} minute(s)"
                        )
                        logs.append(upload_message)
                    except Exception as exc:
                        logs.append(f"Periodic pending sync failed: {exc}. Files remain in {PENDING_UPLOAD_ROOT.resolve()} for next sync.")
                last_periodic_sync_ts = time.time()
        # Watchdog log: if a file stays RUNNING without UI progress for a while,
        # tell the user what phase it is stuck in instead of looking frozen.
        now = time.time()
        for item in progress_status.values():
            if str(item.get("status")) != "RUNNING":
                continue
            age_sec = int(now - float(item.get("updated_at", now) or now))
            if age_sec >= STALE_PROGRESS_LOG_SEC:
                marker = (
                    f"Still running: {item.get('display')} phase={item.get('phase','?')} "
                    f"done={item.get('done',0)}/{item.get('total',0)} ok={item.get('ok',0)} failed={item.get('failed',0)} "
                    f"last_update={age_sec}s detail={item.get('detail','')}"
                )
                if marker not in logs[-20:]:
                    logs.append(marker)
        yield build_log_output(logs, progress_status), list(generated_files)
        await asyncio.sleep(2)
    await asyncio.gather(*workers)
    logs.append("All file workers finished. Starting final upload/sync phase if needed.")
    yield build_log_output(logs, progress_status), list(generated_files)
    if upload:
        entries_to_stage = drain_upload_entries(batch_upload_entries)
        if entries_to_stage:
            staged_files = stage_upload_entries(entries_to_stage, str(PENDING_UPLOAD_ROOT))
            logs.append(f"Final sync: moved {len(staged_files)} files into pending upload folder: {PENDING_UPLOAD_ROOT.resolve()}")
            yield build_log_output(logs, progress_status), list(generated_files)
        if pending_upload_file_count() > 0:
            try:
                upload_message = upload_pending_uploads_once(
                    commit_message=f"Final batch sync: {len(generated_files)} Edge TTS output files"
                )
                logs.append(upload_message)
            except Exception as exc:
                logs.append(f"Final pending sync failed: {exc}. Files remain in {PENDING_UPLOAD_ROOT.resolve()} for next sync or manual sync.")
        else:
            logs.append("Upload requested, but no pending output/cache files need to be committed.")
    final_complete = sum(1 for item in progress_status.values() if item.get("status") == "COMPLETE")
    final_incomplete = sum(1 for item in progress_status.values() if item.get("status") == "INCOMPLETE")
    final_failed = sum(1 for item in progress_status.values() if item.get("status") == "FAILED")
    logs.append(f"ALL DONE: complete={final_complete}, incomplete={final_incomplete}, failed={final_failed}, generated_files={len(generated_files)}")
    logs.append("Batch processing finished.")
    yield build_log_output(logs, progress_status), list(generated_files)


# ==============================================================================
# Operational buttons: health check, retry failed only, build final from cache only
# ==============================================================================

def _manifest_file_display_name(file):
    return os.path.basename(getattr(file, "name", "") or "manifest.json")


def check_cache_health_for_files(files, repo_cache_only=True):
    if not files:
        return "No manifest files selected."
    lines = []
    for file in files:
        try:
            manifest_name = _manifest_file_display_name(file)
            manifest = load_manifest(file.name)
            cache_name = build_manifest_cache_name(manifest_name)
            if repo_cache_only:
                sync_info = sync_remote_cache_folder_to_local(cache_name, clear_local_first=True)
            else:
                sync_info = {"status": "local_only", "files": 0}
            manifest_cache_dir = build_manifest_cache_dir(cache_name)
            defaults = dict(DEFAULT_MANIFEST_STRATEGY)
            defaults.update(manifest.get("audio_strategy") or {})
            ordered_segments = sorted(manifest.get("segments") or [], key=lambda item: int(item.get("order", 0) or 0))
            prune_report = prune_manifest_cache_to_current_manifest(manifest_cache_dir, ordered_segments, defaults)
            index = write_manifest_cache_index(cache_name, manifest, manifest_cache_dir, defaults=defaults)
            summary = summarize_cache_index(index)
            ready = summary.get("cache_ready", 0)
            total = summary.get("segments_total", len(ordered_segments))
            failed = summary.get("failed", 0)
            stale = summary.get("stale", 0)
            missing = summary.get("missing", 0)
            bad_meta = summary.get("bad_meta", 0)
            lines.append(
                f"{manifest_name}: ready={ready}/{total} failed={failed} stale={stale} missing={missing} bad_meta={bad_meta} "
                f"| repo_cache={sync_info.get('status')} files={sync_info.get('files', 0)} | pruned={prune_report.get('removed_files', 0)} "
                f"| ready_for_final={'yes' if ready == total and total > 0 else 'no'}"
            )
        except Exception as exc:
            lines.append(f"{_manifest_file_display_name(file)}: health check failed: {exc}")
    return "\n".join(lines)


async def retry_failed_segments_only(files, worker_count_input, upload_outputs, tts_timeout_sec_input, rescue_repeated_short_failures_input, repo_cache_only_input, force_commit_cache_now_input):
    if not files:
        return "No manifest files selected.", []
    worker_count = normalize_worker_count(worker_count_input, MANIFEST_WORKERS)
    limiter = AdaptiveTTSLimiter(worker_count, step=None)
    upload = bool(upload_outputs)
    tts_timeout_sec = normalize_tts_timeout(tts_timeout_sec_input)
    repo_cache_only = bool(repo_cache_only_input)
    force_commit_cache_now = bool(force_commit_cache_now_input)
    rescue_repeated_short_failures = bool(rescue_repeated_short_failures_input)
    root = tempfile.mkdtemp(prefix="vieneu_tts_retry_failed_only_")
    generated_files = []
    lines = [f"Retry failed only started: files={len(files)} worker_cap={worker_count} auto_downgrade=off"]
    for file in files:
        manifest_name = _manifest_file_display_name(file)
        work_dir = os.path.join(root, sanitize_filename(manifest_name))
        os.makedirs(work_dir, exist_ok=True)
        try:
            manifest = load_manifest(file.name)
            result = await render_manifest_to_outputs(
                manifest=manifest,
                subtitle_format="no_script",
                upload=False,
                cache_name=manifest_name,
                work_dir=work_dir,
                tts_timeout_sec=tts_timeout_sec,
                force_rerender=False,
                rescue_repeated_short_failures=rescue_repeated_short_failures,
                repo_cache_only=repo_cache_only,
                segment_worker_count=worker_count,
                timeline_mode=TIMELINE_FAST,
                tts_semaphore=limiter,
                retry_failed_only=True,
            )
            generated_files.append(result["local_report"])
            report = result.get("report") or {}
            lines.append(
                f"{manifest_name}: retried_previous_failed={report.get('previous_failed_retry_count', 0)} "
                f"ok_now={report.get('segments_rendered_ok', 0)} failed_now={report.get('segments_failed', 0)} report={os.path.basename(result['local_report'])}"
            )
            if upload and force_commit_cache_now:
                msg = upload_manifest_cache_folder_once(manifest_name, commit_message=f"Retry failed only cache commit: {manifest_name}")
                lines.append(f"{manifest_name}: {msg}")
        except Exception as exc:
            lines.append(f"{manifest_name}: retry failed only error: {exc}")
    return "\n".join(lines), generated_files


async def build_final_from_cache_only(files, subtitle_format, worker_count_input, timeline_mode_input, upload_outputs, repo_cache_only_input):
    if not files:
        return "No manifest files selected.", []
    worker_count = normalize_worker_count(worker_count_input, MANIFEST_WORKERS)
    timeline_mode = normalize_timeline_mode(timeline_mode_input)
    upload = bool(upload_outputs)
    repo_cache_only = bool(repo_cache_only_input)
    root = tempfile.mkdtemp(prefix="vieneu_tts_build_final_cache_only_")
    generated_files = []
    upload_entries = []
    lines = [f"Build final from cache only started: files={len(files)} timeline={timeline_mode}"]
    for file in files:
        manifest_name = _manifest_file_display_name(file)
        work_dir = os.path.join(root, sanitize_filename(manifest_name))
        os.makedirs(work_dir, exist_ok=True)
        try:
            manifest = load_manifest(file.name)
            result = await render_manifest_to_outputs(
                manifest=manifest,
                subtitle_format=subtitle_format,
                upload=False,
                cache_name=manifest_name,
                work_dir=work_dir,
                tts_timeout_sec=SEGMENT_TTS_TIMEOUT_SEC,
                force_rerender=False,
                rescue_repeated_short_failures=False,
                repo_cache_only=repo_cache_only,
                segment_worker_count=worker_count,
                timeline_mode=timeline_mode,
                tts_semaphore=None,
                cache_only=True,
            )
            report = result.get("report") or {}
            generated_files.append(result["local_report"])
            if result.get("complete"):
                generated_files.append(result["final_audio"])
                if result.get("local_subtitle"):
                    generated_files.append(result["local_subtitle"])
                lines.append(f"{manifest_name}: COMPLETE from cache | audio={result['audio_name']} cache_hits={report.get('cache_hits', 0)}")
                if upload:
                    upload_entries.append((result["final_audio"], f"outputs/{result['story_prefix']}/{result['audio_name']}"))
                    if result.get("local_subtitle"):
                        upload_entries.append((result["local_subtitle"], f"outputs/{result['story_prefix']}/{result['subtitle_name']}"))
                    upload_entries.append((result["local_report"], f"outputs/{result['story_prefix']}/{result['report_name']}"))
            else:
                lines.append(f"{manifest_name}: NOT READY from cache | ok={report.get('segments_rendered_ok', 0)}/{report.get('segments_total', 0)} failed={report.get('segments_failed', 0)}")
        except Exception as exc:
            lines.append(f"{manifest_name}: build final from cache error: {exc}")
    if upload and upload_entries:
        staged = stage_upload_entries(upload_entries, str(PENDING_UPLOAD_ROOT))
        lines.append(f"Staged {len(staged)} final output file(s).")
        try:
            lines.append(upload_pending_uploads_once(commit_message=f"Build final from cache only ({len(upload_entries)} files)"))
        except Exception as exc:
            lines.append(f"Upload final outputs failed: {exc}. Files remain pending.")
    return "\n".join(lines), generated_files


async def create_demo():
    if gr is None:
        raise RuntimeError("gradio is not installed in this environment.")
    voices = await get_voices()
    default_voice = next(
        (k for k, v in voices.items() if "vi-VN-HoaiMy" in v),
        list(voices.keys())[0],
    )
    with gr.Blocks(title="VieNeu TTS Video Prep", css=PREVIEW_CSS) as demo:
        preview_manifest_example = json.dumps(
            {
                "schema_version": "audio_segments.v2",
                "source_file": "preview_segments.txt",
                "source_final_path": "preview/manual_input",
                "language": "vi-VN",
                "audio_strategy": dict(DEFAULT_MANIFEST_STRATEGY),
                "watermark_plan": default_watermark_plan(),
                "segments": [
                    {
                        "segment_id": "seg_000001",
                        "order": 1,
                        "segment_type": "narration",
                        "text": "Đây là đoạn dẫn truyện để nghe thử.",
                        "source_start_line": 1,
                        "source_end_line": 1,
                        "speaker_source": {"character_hint": "narrator", "gender": "female", "age_tone": "adult", "role_rank": "neutral"},
                        "performed_voice_persona": {"character_hint": "narrator", "gender": "female", "age_tone": "adult", "role_rank": "neutral"},
                        "voice_mode": "natural",
                        "voice_profile_key": "narrator_female_main",
                        "voice": "hoaimy",
                        "rate_pct": 10,
                        "pitch_hz": -8,
                        "pause_after_ms": 220,
                    },
                    {
                        "segment_id": "seg_000002",
                        "order": 2,
                        "segment_type": "dialogue",
                        "text": "Hắn cất giọng nữ yêu điệu, chào bảo bối.",
                        "source_start_line": 2,
                        "source_end_line": 2,
                        "speaker_source": {"character_hint": "han", "gender": "male", "age_tone": "adult", "role_rank": "unknown"},
                        "performed_voice_persona": {"character_hint": "han", "gender": "female", "age_tone": "young_adult", "role_rank": "unknown"},
                        "voice_mode": "imitated",
                        "voice_profile_key": "dialogue_female_young",
                        "voice": "hoaimy",
                        "rate_pct": 10,
                        "pitch_hz": -8,
                        "pause_after_ms": 220,
                    },
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        gr.Markdown(
            "# VieNeu TTS Generator\n"
            "Plain TXT single-voice mode stays backward-compatible. "
            "Manifest mode reads any `.json` manifest file, renders each segment, rebuilds subtitles, and can either download locally or upload grouped outputs."
        )
        input_cfg = input_picker_config(DEFAULT_PLAIN_MODE)
        files = gr.File(
            file_count="multiple",
            file_types=input_cfg["file_types"],
            label=input_cfg["label"],
        )
        file_picker_hint = gr.Markdown(input_cfg["info"])
        mode = gr.Dropdown(
            choices=[DEFAULT_PLAIN_MODE, DEFAULT_MANIFEST_MODE],
            value=DEFAULT_PLAIN_MODE,
            label="Runtime mode",
        )
        voice_dropdown = gr.Dropdown(
            choices=list(voices.keys()),
            value=default_voice,
            label="Single-voice fallback / plain TXT voice",
        )
        with gr.Row():
            rate = gr.Slider(minimum=-50, maximum=50, value=10, step=1, label="Speed (%)")
            pitch = gr.Slider(minimum=-20, maximum=20, value=-8, step=1, label="Pitch (Hz)")
        with gr.Row():
            subtitle_format = gr.Dropdown(
                choices=["srt", "vtt", "no_script"],
                value="srt",
                label="Subtitle format",
            )
            worker_count = gr.Slider(
                minimum=1,
                maximum=16,
                value=1,
                step=1,
                label="Worker count",
                info="VieNeuTTS High Quality v2 chạy model local trên GPU. Cho phép 1-16 worker; với T4 nên test tăng dần 1→2→4 trước khi dùng cao hơn.",
            )
            timeline_mode = gr.Dropdown(
                choices=[TIMELINE_FAST, TIMELINE_ACCURATE],
                value=DEFAULT_TIMELINE_MODE,
                label="Timeline mode",
                info="fast = bo qua silencedetect tung segment de nhanh hon. accurate = can subtitle ky hon nhung cham hon.",
            )
            upload_outputs = gr.Checkbox(
                value=True,
                label="Upload to Hugging Face Dataset",
                info="Bat len neu da set dung HF_TOKEN va DATASET_REPO. Tat de tai file truc tiep tren UI.",
            )
            force_rerender = gr.Checkbox(
                value=False,
                label="Force rerender from zero / clear cache first",
                info="Mac dinh TAT. Chi bat khi muon xoa cache cua manifest hien tai va tao lai tu dau. File fail se khong stage final output.",
            )
            rescue_repeated_short_failures = gr.Checkbox(
                value=False,
                label="Manual group/rescue repeated short/no-audio failures",
                info="Mac dinh KHONG tu gom. Chi khi checkbox nay duoc bat, segment ngan/no-audio da fail tu run truoc va retry lan nay van fail thi moi gom same voice/rate/pitch de cuu.",
            )
            repo_cache_only = gr.Checkbox(
                value=True,
                label="Repo cache only / refresh cache from Dataset first",
                info="Bat len de coi Hugging Face repo la source of truth: clear local cache manifest roi tai folder cache tu repo truoc khi render.",
            )
            force_commit_cache_now = gr.Checkbox(
                value=True,
                label="Force commit manifest cache folder immediately",
                info="Bat len de moi manifest commit chunkcache/<manifest> len repo bang 1 commit rieng, tranh upload tung segment.",
            )
        with gr.Row():
            tts_timeout_sec = gr.Slider(
                minimum=30,
                maximum=300,
                value=180,
                step=10,
                label="TTS timeout / segment chunk (seconds)",
                info="Neu hay timeout thi tang 90-180s, dong thoi giam Worker count.",
            )
            sync_interval_minutes = gr.Slider(
                minimum=0,
                maximum=120,
                value=60,
                step=10,
                label="HF sync interval minutes",
                info="0 = chi commit cuoi batch. 30/60 = sync dinh ky khi batch chay lau, va luon sync lan cuoi khi batch xong.",
            )
        mode.change(
            fn=update_input_picker,
            inputs=[mode],
            outputs=[files, file_picker_hint],
        )
        btn = gr.Button("Convert", variant="primary")
        logs = gr.Textbox(lines=20, label="Logs", elem_id="batch_logs")
        generated_files = gr.File(
            label="Generated files / Download outputs",
            file_count="multiple",
        )
        sync_btn = gr.Button("Sync pending uploads now", variant="secondary")
        sync_status = gr.Textbox(lines=4, label="Manual sync status")
        sync_btn.click(
            fn=manual_sync_pending_uploads,
            inputs=[],
            outputs=sync_status,
        )
        gr.Markdown(
            "### Force commit current manifest cache\n"
            "De trong o duoi de commit tat ca cache manifest hien co trong `chunkcache/` va `pending_hf_upload/chunkcache/`. "
            "Hoac nhap ten manifest/cache, moi dong mot ten, vi du `batch_001.json` hoac `batch_001`."
        )
        force_cache_names = gr.Textbox(
            lines=3,
            label="Manifest cache names to force commit (optional)",
            placeholder="De trong = commit tat ca manifest cache hien co",
        )
        force_cache_btn = gr.Button("Force commit manifest cache now", variant="secondary")
        force_cache_status = gr.Textbox(lines=10, label="Force cache commit status")
        force_cache_btn.click(
            fn=force_commit_current_manifest_caches,
            inputs=[force_cache_names],
            outputs=force_cache_status,
        )
        btn.click(
            fn=batch_tts,
            inputs=[files, mode, voice_dropdown, rate, pitch, subtitle_format, worker_count, timeline_mode, upload_outputs, tts_timeout_sec, sync_interval_minutes, force_rerender, rescue_repeated_short_failures, repo_cache_only, force_commit_cache_now],
            outputs=[logs, generated_files],
        )

        gr.Markdown(
            "### Cache operations\n"
            "Dung cac nut nay khi da co cache: check cache, retry rieng segment failed, hoac build final tu cache ma khong goi Edge TTS."
        )
        with gr.Row():
            health_btn = gr.Button("Check cache health", variant="secondary")
            retry_failed_btn = gr.Button("Retry failed segments only", variant="secondary")
            build_cache_btn = gr.Button("Build final from cache only", variant="secondary")
        ops_status = gr.Textbox(lines=12, label="Cache operation status")
        ops_files = gr.File(label="Cache operation generated files", file_count="multiple")
        health_btn.click(
            fn=check_cache_health_for_files,
            inputs=[files, repo_cache_only],
            outputs=ops_status,
        )
        retry_failed_btn.click(
            fn=retry_failed_segments_only,
            inputs=[files, worker_count, upload_outputs, tts_timeout_sec, rescue_repeated_short_failures, repo_cache_only, force_commit_cache_now],
            outputs=[ops_status, ops_files],
        )
        build_cache_btn.click(
            fn=build_final_from_cache_only,
            inputs=[files, subtitle_format, worker_count, timeline_mode, upload_outputs, repo_cache_only],
            outputs=[ops_status, ops_files],
        )

        gr.Markdown(
            "## Segment Preview Sandbox\n"
            "Paste a full `audio_segments.v2` manifest or just a `segments` array, then render locally to listen before real-environment testing."
        )
        preview_manifest_input = gr.Code(
            value=preview_manifest_example,
            language="json",
            lines=18,
            label="Preview manifest / segments JSON",
            elem_id="preview_manifest_box",
        )
        with gr.Row():
            preview_subtitle_format = gr.Dropdown(
                choices=["srt", "vtt", "no_script"],
                value="no_script",
                label="Preview subtitle format",
            )
            preview_force_rescue = gr.Checkbox(
                value=False,
                label="Preview: manually group short segments now",
                info="Bat len thi preview moi gom/tach segment ngan ngay lap tuc. Tat = render tung segment rieng, khong tu gom.",
            )
            preview_btn = gr.Button("Render Preview Audio", variant="secondary")
        preview_audio = gr.Audio(
            label="Preview audio",
            type="filepath",
        )
        preview_status = gr.Textbox(
            lines=3,
            label="Preview status",
        )
        preview_dir_state = gr.State(value=None)
        preview_btn.click(
            fn=cleanup_preview_dir,
            inputs=[preview_dir_state],
            outputs=[preview_dir_state],
            queue=False,
        ).then(
            fn=preview_manifest_text,
            inputs=[preview_manifest_input, preview_subtitle_format, preview_force_rescue],
            outputs=[preview_audio, preview_status, preview_dir_state],
        )
    return demo


# ==============================================================================
# VieNeuTTS adapter overrides
# ==============================================================================
# Overrides only engine/cache transport pieces. Main manifest logic remains from the
# updated base: fixed worker cap, same timeout for short segments, compact cache
# (_cache_index/failed/final_parts only), failed-first retry, optional rescue
# grouping, final-parts cache, subtitles from JSON text, watchdog logs, and Gradio UI.

VIENEU_SAMPLE_RATE = 24000
VIENEU_DEFAULT_VOICE = "doan"
# VieNeu SDK supports emotion at engine init: "natural" or "storytelling".
# For long-form stories, "storytelling" is usually more stable and less tiring
# than letting prosody swing segment-by-segment.
VIENEU_DEFAULT_EMOTION = "storytelling"
VIENEU_ALLOWED_EMOTIONS = {"natural", "storytelling"}
VIENEU_PRESET_HINTS = {
    "doan": ["đoan", "doan", "nữ miền nam", "nu mien nam", "nam nu", "doan_nam_nu"],
    "vinh": ["vĩnh", "vinh", "nam miền nam", "nam mien nam", "vinh_nam_nam"],
}
VIENEU_VOICE_ALIASES = {
    "doan": "doan", "đoan": "doan", "doan_nam_nu": "doan", "doan nam nu": "doan",
    "nu_mien_nam": "doan", "nữ miền nam": "doan", "nu mien nam": "doan",
    "hoaimy": "doan", "hoai my": "doan", "hoài my": "doan",
    "vi-vn-hoaimyneural": "doan", "vi-VN-HoaiMyNeural": "doan",
    "female": "doan", "female_south": "doan", "narrator_female": "doan", "dialogue_female": "doan",
    "vinh": "vinh", "vĩnh": "vinh", "vinh_nam_nam": "vinh", "vinh nam nam": "vinh",
    "nam_mien_nam": "vinh", "nam miền nam": "vinh", "nam mien nam": "vinh",
    "namminh": "vinh", "nam minh": "vinh", "vi-vn-namminhneural": "vinh", "vi-VN-NamMinhNeural": "vinh",
    "male": "vinh", "male_south": "vinh", "dialogue_male": "vinh",
}
_VIENEU_TTS_SINGLETON = None
_VIENEU_TTS_ENGINES_BY_EMOTION = {}
_VIENEU_ACTIVE_SWITCH_ENGINE = None
_VIENEU_ACTIVE_SWITCH_EMOTION = None
_VIENEU_ENGINE_INIT_LOCK = threading.Lock()
_VIENEU_THREAD_LOCAL = threading.local()
_VIENEU_PRESET_VOICE_CACHE = {}
_VIENEU_PRESET_LIST_CACHE = None
_VIENEU_ENGINE_INIT_FATAL_ERROR = None


class VieneuEngineInitFatalError(RuntimeError):
    """Raised when VieNeu engine initialization fails and fail-fast is enabled."""


def vieneu_fail_fast_on_engine_init():
    return os.getenv("VIENEU_FAIL_FAST_ON_ENGINE_INIT", "1").strip().lower() in {"1", "true", "yes", "on"}


def vieneu_preload_shared_engine_enabled():
    return os.getenv("VIENEU_PRELOAD_SHARED_ENGINE", "1").strip().lower() in {"1", "true", "yes", "on"}


def _set_vieneu_engine_init_fatal_error(exc):
    global _VIENEU_ENGINE_INIT_FATAL_ERROR
    _VIENEU_ENGINE_INIT_FATAL_ERROR = exc


def _raise_if_vieneu_engine_init_fatal_error():
    if _VIENEU_ENGINE_INIT_FATAL_ERROR is not None and vieneu_fail_fast_on_engine_init():
        raise VieneuEngineInitFatalError(str(_VIENEU_ENGINE_INIT_FATAL_ERROR)) from _VIENEU_ENGINE_INIT_FATAL_ERROR


def _is_engine_init_failure(exc):
    current = exc
    while current is not None:
        if isinstance(current, VieneuEngineInitFatalError):
            return True
        msg = str(current)
        if "VieNeuTTS init failed" in msg or "FastVieNeuTTS" in msg or "LMDeploy" in msg or "lmdeploy" in msg:
            return True
        current = getattr(current, "__cause__", None) or getattr(current, "__context__", None)
    return False


def _format_engine_init_fatal_message(slot_label, kwargs, exc):
    safe_kwargs = {k: v for k, v in kwargs.items() if k not in {"hf_token", "token"}}
    return (
        f"FATAL VieNeuTTS engine init failed at slot={slot_label}. "
        f"Audio generation stopped before rendering remaining segments. "
        f"Init kwargs={safe_kwargs}. Original error: {exc}"
    )


def _vieneu_import_error_message():
    return (
        "VieNeuTTS SDK is not installed. In Colab run:\n"
        "!apt-get update -y && apt-get install -y espeak-ng ffmpeg\n"
        "!pip install -U vieneu gradio huggingface_hub soundfile librosa\n"
        "Then restart runtime if needed."
    )


def normalize_vieneu_emotion(value, default=VIENEU_DEFAULT_EMOTION):
    raw = str(value or default or VIENEU_DEFAULT_EMOTION).strip().lower()
    if raw in {"story", "story_telling", "story-telling", "ke_chuyen", "kể chuyện", "ke chuyen"}:
        return "storytelling"
    if raw in {"natural", "tu_nhien", "tự nhiên", "tu nhien", "emotion", "emotional", "cam_xuc", "cảm xúc", "cam xuc"}:
        return "natural"
    if raw in VIENEU_ALLOWED_EMOTIONS:
        return raw
    return default


def get_vieneu_emotion(default=VIENEU_DEFAULT_EMOTION):
    return normalize_vieneu_emotion(os.getenv("VIENEU_EMOTION", default), default=default)


def get_vieneu_narration_emotion():
    return normalize_vieneu_emotion(os.getenv("VIENEU_NARRATION_EMOTION", "storytelling"), default="storytelling")


def get_vieneu_dialogue_emotion():
    return normalize_vieneu_emotion(os.getenv("VIENEU_DIALOGUE_EMOTION", "natural"), default="natural")


def get_vieneu_backend_mode():
    """Resolve backend independently from device.

    standard = PyTorch/GGUF backend. Can run CPU or GPU and supports emotion= at init.
    fast     = LMDeploy backend. GPU only; emotion is passed at infer-time via emotion_tag.
    turbo    = lightweight CPU/GGUF backend; emotion is passed at infer-time when supported.
    """
    raw = os.getenv("VIENEU_BACKEND_MODE", "auto").strip().lower()
    aliases = {
        "gpu_standard": "standard",
        "cpu_standard": "standard",
        "standard_gpu": "standard",
        "standard_cpu": "standard",
        "lmdeploy": "fast",
        "gpu_lmdeploy": "fast",
        "fast_gpu": "fast",
        "cpu_turbo": "turbo",
        "turbo_cpu": "turbo",
        "lite": "turbo",
    }
    raw = aliases.get(raw, raw)
    if raw in {"standard", "fast", "turbo"}:
        return raw

    device_mode = os.getenv("VIENEU_DEVICE_MODE", "gpu").strip().lower()
    quality_mode = os.getenv("VIENEU_QUALITY_MODE", "full_quality_v2").strip().lower()
    if quality_mode in {"fast", "fast_0_3b", "0.3b", "lmdeploy"}:
        return "fast" if device_mode in {"gpu", "cuda"} else "turbo"
    if quality_mode in {"turbo", "lite"}:
        return "turbo"
    # Default used by the notebook presets now: full-quality GPU/CPU should use Standard.
    return "standard"


def vieneu_init_supports_emotion():
    """Return True only for VieNeu backends that accept emotion= in __init__."""
    return get_vieneu_backend_mode() == "standard"


def vieneu_emotion_tag_for_infer(emotion=None):
    """Map logical emotion to VieNeu infer(..., emotion_tag=...).

    standard.py maps natural to <|emotion_0|> and storytelling to no explicit tag.
    FastVieNeuTTS reads emotion_tag from infer kwargs.
    """
    emo = normalize_vieneu_emotion(emotion or get_vieneu_emotion())
    if emo == "natural":
        return "<|emotion_0|>"
    return None


def _vieneu_env_float(name, default=None):
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _vieneu_env_int(name, default=None):
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _vieneu_env_bool(name, default=None):
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


_VIENEU_FAST_LMDEPLOY_PATCHED = False


def apply_vieneu_fast_lmdeploy_patch_once():
    """Patch VieNeu Fast/LMDeploy defaults for T4/Kaggle stability.

    Upstream FastVieNeuTTS uses TurbomindEngineConfig(dtype='bfloat16') and does
    not set session_len/max_context_token_num. On T4 this can fail during
    Turbomind warm-up with logs like:
      total sequence length (4096 + 1) exceeds session_len (4096)
      Warm-up for 6144/8192/8320 tokens failed with status 6

    This patch keeps the official Fast backend but makes LMDeploy explicit:
      - dtype float16 by default for T4
      - session/context length configurable, default 12288
      - generation max_new_tokens configurable
    """
    global _VIENEU_FAST_LMDEPLOY_PATCHED
    if _VIENEU_FAST_LMDEPLOY_PATCHED:
        return
    if get_vieneu_backend_mode() != "fast":
        return
    try:
        import inspect
        from vieneu.fast import FastVieNeuTTS
        from vieneu.utils import _compile_codec_with_triton
    except Exception as exc:
        print(f"⚠️ Fast LMDeploy patch skipped before import: {exc}", flush=True)
        return

    original = getattr(FastVieNeuTTS, "_load_backbone_lmdeploy", None)
    if original is None:
        print("⚠️ Fast LMDeploy patch skipped: method not found", flush=True)
        return

    def _patched_load_backbone_lmdeploy(self, repo, memory_util, tp, enable_prefix_caching, quant_policy, hf_token=None):
        logger_obj = getattr(__import__('logging'), 'getLogger')('Vieneu.Fast')
        logger_obj.info(f"Loading backbone with patched LMDeploy from: {repo}")
        if hf_token:
            import os as _os
            _os.environ["HF_TOKEN"] = hf_token
        try:
            from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
        except ImportError as e:
            raise ImportError("Failed to import `lmdeploy`. Install with: pip install vieneu[gpu]") from e

        dtype = os.getenv("VIENEU_LMDEPLOY_DTYPE", "float16").strip() or "float16"
        session_len = _vieneu_env_int("VIENEU_LMDEPLOY_SESSION_LEN", 12288)
        max_prefill = _vieneu_env_int("VIENEU_LMDEPLOY_MAX_PREFILL_TOKEN_NUM", 4096)

        cfg_kwargs = {
            "cache_max_entry_count": memory_util,
            "tp": tp,
            "enable_prefix_caching": enable_prefix_caching,
            "dtype": dtype,
            "quant_policy": quant_policy,
        }
        try:
            sig = inspect.signature(TurbomindEngineConfig)
            params = set(sig.parameters.keys())
        except Exception:
            params = set()
        # LMDeploy versions differ. Pass only supported names.
        if session_len and "session_len" in params:
            cfg_kwargs["session_len"] = int(session_len)
        if session_len and "max_context_token_num" in params:
            cfg_kwargs["max_context_token_num"] = int(session_len)
        if max_prefill and "max_prefill_token_num" in params:
            cfg_kwargs["max_prefill_token_num"] = int(max_prefill)

        print("🛠️ Fast LMDeploy patched config:", cfg_kwargs, flush=True)
        backend_config = TurbomindEngineConfig(**cfg_kwargs)
        self.backbone = pipeline(repo, backend_config=backend_config)

        stability = get_vieneu_stability_defaults()
        max_new_tokens = _vieneu_env_int("VIENEU_LMDEPLOY_MAX_NEW_TOKENS", 1024)
        min_new_tokens = _vieneu_env_int("VIENEU_LMDEPLOY_MIN_NEW_TOKENS", 24)
        self.gen_config = GenerationConfig(
            top_p=float(stability.get("top_p", 0.80)),
            top_k=int(stability.get("top_k", 20)),
            temperature=float(stability.get("temperature", 0.55)),
            max_new_tokens=int(max_new_tokens),
            repetition_penalty=float(stability.get("repetition_penalty", 1.10)),
            do_sample=bool(stability.get("do_sample", True)),
            min_new_tokens=int(min_new_tokens),
        )

    FastVieNeuTTS._load_backbone_lmdeploy = _patched_load_backbone_lmdeploy
    _VIENEU_FAST_LMDEPLOY_PATCHED = True
    print("✅ Fast LMDeploy patch installed (dtype/session_len/generation defaults).", flush=True)


def get_vieneu_voice_stability_mode():
    return os.getenv("VIENEU_VOICE_STABILITY_MODE", "stable").strip().lower()



def get_vieneu_render_input_mode_for_stability():
    """Normalize render mode for mode-specific stability overrides.

    Supported modes:
      - segments
      - tag_long_text
      - txt_long_text

    This only controls generation sampling overrides. It does not change
    render/split behavior.
    """
    raw = (
        os.getenv("VIENEU_RENDER_INPUT_MODE")
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
        "long_text": "txt_long_text",
        "longtext": "txt_long_text",
    }
    return aliases.get(raw, raw if raw in {"segments", "tag_long_text", "txt_long_text"} else "segments")


def get_vieneu_stability_defaults():
    """Return stable generation defaults to reduce per-segment tone drift.

    VieNeu Fast uses sampling by default (temperature=1.0, top_k=50, top_p=0.95).
    That can make short segments jump between high/low tone. These presets reduce
    randomness while keeping enough variation for Vietnamese speech.
    """
    mode = get_vieneu_voice_stability_mode()
    presets = {
        # locked_safe = khuyến nghị: ổn định mạnh nhưng vẫn bật sampling nhẹ.
        # An toàn hơn locked_max vì tránh vài đoạn không ra speech token / im lặng.
        "locked_safe": {"temperature": 0.28, "top_k": 8, "top_p": 0.70, "do_sample": True, "repetition_penalty": 1.06},
        # locked_max = khóa mạnh nhất: do_sample=False, top_k=1.
        # Có thể gây vài đoạn im lặng/no audio, chỉ dùng để test ngắn.
        "locked_max": {"temperature": 0.25, "top_k": 1, "top_p": 0.60, "do_sample": False, "repetition_penalty": 1.05},
        "locked": {"temperature": 0.30, "top_k": 12, "top_p": 0.72, "do_sample": True, "repetition_penalty": 1.08},
        "stable": {"temperature": 0.55, "top_k": 20, "top_p": 0.80, "do_sample": True, "repetition_penalty": 1.10},
        "balanced": {"temperature": 0.70, "top_k": 35, "top_p": 0.88, "do_sample": True, "repetition_penalty": 1.15},
        "creative": {"temperature": 1.00, "top_k": 50, "top_p": 0.95, "do_sample": True, "repetition_penalty": 1.20},
        "off": {},
        "disabled": {},
        "custom": {},
    }
    values = dict(presets.get(mode, presets["locked_safe"]))

    # Env overrides win. This lets notebook users tune without patching code.
    temp = _vieneu_env_float("VIENEU_INFER_TEMPERATURE", None)
    top_k = _vieneu_env_int("VIENEU_INFER_TOP_K", None)
    top_p = _vieneu_env_float("VIENEU_INFER_TOP_P", None)
    rep = _vieneu_env_float("VIENEU_INFER_REPETITION_PENALTY", None)
    do_sample = _vieneu_env_bool("VIENEU_INFER_DO_SAMPLE", None)

    if temp is not None:
        values["temperature"] = temp
    if top_k is not None:
        values["top_k"] = top_k
    if top_p is not None:
        values["top_p"] = top_p
    if rep is not None:
        values["repetition_penalty"] = rep
    if do_sample is not None:
        values["do_sample"] = do_sample

    # Optional mode-specific overrides. These apply to all three user-facing
    # render options without changing render logic:
    #   segments      -> VIENEU_SEGMENT_TEMPERATURE / TOP_K / TOP_P
    #   tag_long_text -> VIENEU_TAG_TEMPERATURE / TOP_K / TOP_P
    #   txt_long_text -> VIENEU_LONG_TEXT_TEMPERATURE / TOP_K / TOP_P
    render_mode = get_vieneu_render_input_mode_for_stability()
    prefix_map = {
        "segments": "VIENEU_SEGMENT",
        "tag_long_text": "VIENEU_TAG",
        "txt_long_text": "VIENEU_LONG_TEXT",
    }
    mode_prefix = prefix_map.get(render_mode)
    if mode_prefix:
        mode_temp = _vieneu_env_float(f"{mode_prefix}_TEMPERATURE", None)
        mode_top_k = _vieneu_env_int(f"{mode_prefix}_TOP_K", None)
        mode_top_p = _vieneu_env_float(f"{mode_prefix}_TOP_P", None)
        if mode_temp is not None:
            values["temperature"] = mode_temp
        if mode_top_k is not None:
            values["top_k"] = mode_top_k
        if mode_top_p is not None:
            values["top_p"] = mode_top_p
    return values


def apply_vieneu_generation_stability(tts):
    """Apply generation config values that infer() does not expose directly.

    FastVieNeuTTS exposes temperature/top_k through infer(), but top_p/do_sample/
    repetition_penalty live on tts.gen_config. Standard backends may also expose
    gen_config, so this is intentionally best-effort.
    """
    values = get_vieneu_stability_defaults()
    gen_config = getattr(tts, "gen_config", None)
    if gen_config is not None:
        for key in ("top_p", "do_sample", "repetition_penalty"):
            if key in values:
                try:
                    setattr(gen_config, key, values[key])
                except Exception:
                    pass
    return values


def build_vieneu_infer_kwargs_for_stability(tts):
    values = apply_vieneu_generation_stability(tts)
    infer_kwargs = {}
    # Both Fast and Standard infer signatures commonly accept temperature/top_k.
    if "temperature" in values:
        infer_kwargs["temperature"] = values["temperature"]
    if "top_k" in values:
        infer_kwargs["top_k"] = values["top_k"]
    return infer_kwargs



def get_vieneu_standard_speed_defaults():
    """Return generation limits for Standard/Torch backend speed.

    Standard VieNeuTTS uses generate(max_length=self.max_context, min_new_tokens=50)
    in the upstream source. That is stable but slow for many short story segments.
    This patch keeps output quality reasonable while reducing extra token generation.
    """
    mode = os.getenv("VIENEU_STANDARD_SPEED_MODE", "balanced").strip().lower()
    presets = {
        "off": {},
        "quality": {"min_new_tokens": 50, "max_new_tokens": 1024},
        "balanced": {"min_new_tokens": 32, "max_new_tokens": 768},
        "fast": {"min_new_tokens": 18, "max_new_tokens": 512},
        "ultrafast": {"min_new_tokens": 12, "max_new_tokens": 384},
        "custom": {},
    }
    values = dict(presets.get(mode, presets["balanced"]))
    raw_min = os.getenv("VIENEU_STANDARD_MIN_NEW_TOKENS", "").strip()
    raw_max = os.getenv("VIENEU_STANDARD_MAX_NEW_TOKENS", "").strip()
    raw_ctx = os.getenv("VIENEU_STANDARD_MAX_CONTEXT", "").strip()
    raw_watermark = os.getenv("VIENEU_APPLY_WATERMARK", "").strip().lower()
    raw_infer_max_chars = os.getenv("VIENEU_INFER_MAX_CHARS", "").strip()
    try:
        if raw_min:
            values["min_new_tokens"] = int(raw_min)
        if raw_max:
            values["max_new_tokens"] = int(raw_max)
        if raw_ctx:
            values["max_context"] = int(raw_ctx)
        if raw_infer_max_chars:
            values["max_chars"] = int(raw_infer_max_chars)
    except Exception:
        pass
    if raw_watermark in {"0", "false", "no", "off"}:
        values["apply_watermark"] = False
    elif raw_watermark in {"1", "true", "yes", "on"}:
        values["apply_watermark"] = True
    return values


def apply_vieneu_standard_speed_patch(tts):
    """Monkey-patch Standard Torch inference to use max_new_tokens/min_new_tokens.

    This does not affect Fast/LMDeploy. It only patches non-quantized Standard
    engines that expose _infer_torch. The goal is faster short-segment rendering.
    """
    values = get_vieneu_standard_speed_defaults()
    if not values or not hasattr(tts, "_infer_torch"):
        return values
    if getattr(tts, "_vieneu_speed_patch_applied", False):
        return values
    if getattr(tts, "_is_quantized_model", False):
        # GGUF path uses llama.cpp; leave it alone for safety.
        return values
    try:
        import types
        def _infer_torch_speed(self, prompt_ids, temperature=1.0, top_k=50):
            import torch
            prompt_tensor = torch.tensor(prompt_ids).unsqueeze(0).to(self.backbone.device)
            speech_end_id = self.tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")
            gen_values = get_vieneu_standard_speed_defaults()
            stability = get_vieneu_stability_defaults()
            max_new_tokens = int(gen_values.get("max_new_tokens", 768))
            min_new_tokens = int(gen_values.get("min_new_tokens", 32))
            # Keep generation within context even if prompt is long.
            max_context = int(gen_values.get("max_context", getattr(self, "max_context", 2048)))
            remaining = max(64, max_context - int(prompt_tensor.shape[-1]))
            max_new_tokens = max(64, min(max_new_tokens, remaining))
            min_new_tokens = max(1, min(min_new_tokens, max_new_tokens - 1 if max_new_tokens > 1 else 1))
            kwargs = {
                "max_new_tokens": max_new_tokens,
                "eos_token_id": speech_end_id,
                "do_sample": bool(stability.get("do_sample", True)),
                "temperature": float(temperature),
                "top_k": int(top_k),
                "use_cache": True,
                "min_new_tokens": min_new_tokens,
            }
            if "top_p" in stability:
                kwargs["top_p"] = float(stability["top_p"])
            if "repetition_penalty" in stability:
                kwargs["repetition_penalty"] = float(stability["repetition_penalty"])
            with torch.no_grad():
                output_tokens = self.backbone.generate(prompt_tensor, **kwargs)
            input_length = prompt_tensor.shape[-1]
            output_str = self.tokenizer.decode(output_tokens[0, input_length:].cpu().numpy().tolist(), add_special_tokens=False)
            return output_str
        tts._infer_torch = types.MethodType(_infer_torch_speed, tts)
        tts._vieneu_speed_patch_applied = True
        print(f"⚡ Standard speed patch enabled: {values}", flush=True)
    except Exception as exc:
        print(f"⚠️ Standard speed patch skipped: {exc}", flush=True)
    return values


def log_vieneu_stability_once():
    global _VIENEU_STABILITY_LOGGED
    try:
        already = _VIENEU_STABILITY_LOGGED
    except NameError:
        already = False
    if already:
        return
    _VIENEU_STABILITY_LOGGED = True
    values = get_vieneu_stability_defaults()
    print(
        "🎚️ VieNeu voice stability:",
        {
            "mode": get_vieneu_voice_stability_mode(),
            "render_input_mode": get_vieneu_render_input_mode_for_stability(),
            **values,
        },
        flush=True,
    )


def resolve_vieneu_emotion_for_segment(segment=None, defaults=None):
    segment = segment or {}
    explicit = segment.get("emotion") or segment.get("vieneu_emotion")
    if explicit:
        return normalize_vieneu_emotion(explicit, default=get_vieneu_emotion())
    segment_type = str(segment.get("segment_type") or "").strip().lower()
    if segment_type == "dialogue":
        return get_vieneu_dialogue_emotion()
    return get_vieneu_narration_emotion()

def get_vieneu_render_order_strategy():
    """How segments are rendered before final concat.

    manifest_order:
        Render in original order. Best when dual_preload_engine already holds both engines.
    non_dialogue_then_dialogue / two_pass_dialogue:
        Render narration/inner_monologue/seo_tag first, then dialogue. Final audio is still
        assembled in original manifest order. This is designed for single_switch_engine on T4:
        load storytelling once for non-dialogue, then switch once to natural for dialogue.
    """
    raw = os.getenv("VIENEU_RENDER_ORDER_STRATEGY", "manifest_order").strip().lower()
    aliases = {
        "original": "manifest_order",
        "original_order": "manifest_order",
        "manifest": "manifest_order",
        "normal": "manifest_order",
        "two_pass": "non_dialogue_then_dialogue",
        "two_pass_dialogue": "non_dialogue_then_dialogue",
        "non_dialogue_first": "non_dialogue_then_dialogue",
        "narration_then_dialogue": "non_dialogue_then_dialogue",
        "story_then_dialogue": "non_dialogue_then_dialogue",
        "storytelling_then_natural": "non_dialogue_then_dialogue",
    }
    return aliases.get(raw, raw)


def is_dialogue_segment_for_vieneu(segment):
    return str((segment or {}).get("segment_type") or "").strip().lower() == "dialogue"


def order_segment_pairs_for_vieneu_render(segment_pairs):
    strategy = get_vieneu_render_order_strategy()
    pairs = list(segment_pairs or [])
    if strategy in {"non_dialogue_then_dialogue", "dialogue_last", "two_pass_dialogue"}:
        non_dialogue = []
        dialogue = []
        for order, segment in pairs:
            if is_dialogue_segment_for_vieneu(segment):
                dialogue.append((order, segment))
            else:
                non_dialogue.append((order, segment))
        return non_dialogue + dialogue
    if strategy in {"dialogue_then_non_dialogue", "dialogue_first"}:
        non_dialogue = []
        dialogue = []
        for order, segment in pairs:
            if is_dialogue_segment_for_vieneu(segment):
                dialogue.append((order, segment))
            else:
                non_dialogue.append((order, segment))
        return dialogue + non_dialogue
    return pairs


def _make_vieneu_engine_kwargs(emotion=None):
    """Build VieNeu init kwargs from backend/device presets.

    Backends:
    - standard: PyTorch/GGUF backend; CPU or GPU; accepts emotion= at init.
    - fast: LMDeploy backend; GPU only; no emotion= at init, use infer emotion_tag.
    - turbo: lightweight CPU fallback; no emotion= at init.
    """
    device_mode = os.getenv("VIENEU_DEVICE_MODE", "gpu").strip().lower()
    quality_mode = os.getenv("VIENEU_QUALITY_MODE", "full_quality_v2").strip().lower()
    backend_mode = get_vieneu_backend_mode()
    engine_emotion = normalize_vieneu_emotion(emotion or get_vieneu_emotion())
    wants_gpu = device_mode in {"gpu", "cuda", "lmdeploy", "gpu_lmdeploy"}

    if backend_mode == "fast":
        # Fast/LMDeploy backend does NOT accept emotion= in __init__.
        # Emotion is applied later through infer(..., emotion_tag=...).
        kwargs = {"mode": "fast"}
        if quality_mode in {"fast", "fast_0_3b", "0.3b", "lite"}:
            default_backbone = "pnnbao-ump/VieNeu-TTS-0.3B"
        else:
            default_backbone = "pnnbao-ump/VieNeu-TTS-v2"
        backbone_repo = os.getenv("VIENEU_BACKBONE_REPO", default_backbone).strip()
        codec_repo = os.getenv("VIENEU_CODEC_REPO", "neuphonic/distill-neucodec").strip()
        if backbone_repo:
            kwargs["backbone_repo"] = backbone_repo
        if codec_repo:
            kwargs["codec_repo"] = codec_repo
        kwargs["backbone_device"] = os.getenv("VIENEU_BACKBONE_DEVICE", "cuda").strip() or "cuda"
        kwargs["codec_device"] = os.getenv("VIENEU_CODEC_DEVICE", "cuda").strip() or "cuda"

        for key, env_name, caster in (
            ("memory_util", "VIENEU_MEMORY_UTIL", float),
            ("tp", "VIENEU_TP", int),
            ("quant_policy", "VIENEU_QUANT_POLICY", int),
            ("max_batch_size", "VIENEU_MAX_BATCH_SIZE", int),
        ):
            raw = os.getenv(env_name, "").strip()
            if raw:
                try:
                    kwargs[key] = caster(raw)
                except ValueError:
                    pass
        raw_prefix = os.getenv("VIENEU_ENABLE_PREFIX_CACHING", "").strip().lower()
        if raw_prefix in {"1", "true", "yes", "on"}:
            kwargs["enable_prefix_caching"] = True
        elif raw_prefix in {"0", "false", "no", "off"}:
            kwargs["enable_prefix_caching"] = False
        return kwargs

    if backend_mode == "turbo":
        # Turbo backend does NOT accept emotion= in __init__.
        kwargs = {"mode": "turbo"}
        raw_repo = os.getenv("VIENEU_TURBO_BACKBONE_REPO", "").strip()
        raw_file = os.getenv("VIENEU_TURBO_BACKBONE_FILENAME", "").strip()
        if raw_repo:
            kwargs["backbone_repo"] = raw_repo
        if raw_file:
            kwargs["backbone_filename"] = raw_file
        return kwargs

    # Standard backend: the one that supports emotion= at init.
    standard_device = "cuda" if wants_gpu else "cpu"
    default_standard_repo = "pnnbao-ump/VieNeu-TTS-v2" if quality_mode in {"full_quality_v2", "standard_v2", "standard_full"} else "pnnbao-ump/VieNeu-TTS-0.3B"
    default_codec = "neuphonic/distill-neucodec" if wants_gpu else "neuphonic/neucodec-onnx-decoder-int8"
    kwargs = {
        "mode": "standard",
        "emotion": engine_emotion,
        "backbone_repo": os.getenv("VIENEU_STANDARD_BACKBONE_REPO", os.getenv("VIENEU_BACKBONE_REPO", default_standard_repo)).strip(),
        "codec_repo": os.getenv("VIENEU_CODEC_REPO", default_codec).strip(),
        "backbone_device": os.getenv("VIENEU_BACKBONE_DEVICE", standard_device).strip() or standard_device,
        "codec_device": os.getenv("VIENEU_CODEC_DEVICE", standard_device).strip() or standard_device,
    }
    # Empty string disables GGUF and uses PyTorch/Transformers weights.
    gguf_filename = os.getenv("VIENEU_GGUF_FILENAME", "").strip()
    if gguf_filename:
        kwargs["gguf_filename"] = gguf_filename
    return kwargs


def _init_vieneu_engine_for_slot(slot_label="shared", emotion=None):
    if Vieneu is None:
        raise RuntimeError(_vieneu_import_error_message())
    kwargs = _make_vieneu_engine_kwargs(emotion=emotion)
    allow_fallback = os.getenv("VIENEU_ALLOW_GGUF_FALLBACK", "0").strip().lower() in {"1", "true", "yes"}
    try:
        print(
            f"🚀 Initializing VieNeuTTS pure-Colab engine[{slot_label}] with kwargs:",
            {k: v for k, v in kwargs.items() if k not in {"hf_token", "token"}},
            flush=True,
        )
        apply_vieneu_fast_lmdeploy_patch_once()
        engine = Vieneu(**kwargs)
        apply_vieneu_standard_speed_patch(engine)
        print(f"✅ VieNeuTTS engine[{slot_label}] init done.", flush=True)
        return engine
    except Exception as exc:
        if not allow_fallback:
            fatal_msg = _format_engine_init_fatal_message(slot_label, kwargs, exc)
            fatal_exc = VieneuEngineInitFatalError(fatal_msg)
            _set_vieneu_engine_init_fatal_error(fatal_exc)
            print("❌ " + fatal_msg, flush=True)
            raise fatal_exc from exc
        print(f"⚠️ Init failed, fallback Vieneu() vì VIENEU_ALLOW_GGUF_FALLBACK=1: {exc}", flush=True)
        return Vieneu()


def _free_vieneu_gpu_memory():
    try:
        import gc
        gc.collect()
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def get_vieneu_engine(emotion=None):
    """Return a VieNeu engine for the requested emotion.

    shared_single_engine:
        One global engine using VIENEU_EMOTION.
    dual_preload_engine:
        Two global engines, loaded sequentially: narration/storytelling and dialogue/natural.
    single_switch_engine:
        One active engine at a time. If requested emotion changes, unload current engine then init the new one.
        Slower when narration/dialogue interleave, but safer for low VRAM.
    per_thread_engine/per_worker_engine:
        Legacy strategy, one engine per worker thread using VIENEU_EMOTION.
    """
    global _VIENEU_TTS_SINGLETON, _VIENEU_ACTIVE_SWITCH_ENGINE, _VIENEU_ACTIVE_SWITCH_EMOTION
    _raise_if_vieneu_engine_init_fatal_error()
    strategy = os.getenv("VIENEU_ENGINE_STRATEGY", "shared_single_engine").strip().lower()
    requested_emotion = normalize_vieneu_emotion(emotion or get_vieneu_emotion())

    # Fast/LMDeploy and turbo cannot be split by init emotion. Use one shared engine;
    # the segment emotion is still applied in synthesize_audio_once via emotion_tag.
    if not vieneu_init_supports_emotion():
        requested_emotion = "shared"

    if strategy in {"dual_preload", "dual_preload_engine", "dual_engine", "two_engine", "two_engines"}:
        with _VIENEU_ENGINE_INIT_LOCK:
            engine = _VIENEU_TTS_ENGINES_BY_EMOTION.get(requested_emotion)
            if engine is None:
                engine = _init_vieneu_engine_for_slot(f"dual_{requested_emotion}", emotion=requested_emotion)
                _VIENEU_TTS_ENGINES_BY_EMOTION[requested_emotion] = engine
            return engine

    if strategy in {"single_switch", "single_switch_engine", "switch_engine", "switch_by_emotion"}:
        with _VIENEU_ENGINE_INIT_LOCK:
            if _VIENEU_ACTIVE_SWITCH_ENGINE is not None and _VIENEU_ACTIVE_SWITCH_EMOTION == requested_emotion:
                return _VIENEU_ACTIVE_SWITCH_ENGINE
            if _VIENEU_ACTIVE_SWITCH_ENGINE is not None:
                print(f"♻️ Switching VieNeuTTS engine: {_VIENEU_ACTIVE_SWITCH_EMOTION} -> {requested_emotion}", flush=True)
                try:
                    close_fn = getattr(_VIENEU_ACTIVE_SWITCH_ENGINE, "close", None)
                    if callable(close_fn):
                        close_fn()
                except Exception as close_exc:
                    print(f"⚠️ VieNeuTTS close() warning: {close_exc}", flush=True)
                _VIENEU_ACTIVE_SWITCH_ENGINE = None
                _VIENEU_ACTIVE_SWITCH_EMOTION = None
                _free_vieneu_gpu_memory()
            _VIENEU_ACTIVE_SWITCH_ENGINE = _init_vieneu_engine_for_slot(f"switch_{requested_emotion}", emotion=requested_emotion)
            _VIENEU_ACTIVE_SWITCH_EMOTION = requested_emotion
            return _VIENEU_ACTIVE_SWITCH_ENGINE

    if strategy in {"per_thread", "per_thread_engine", "per_worker", "per_worker_engine", "parallel", "multi_worker_engine", "worker_pool_engine"}:
        # Standard multi-worker mode: each render thread owns its own engine per emotion.
        # This avoids the global single_switch_engine race when multiple workers render
        # storytelling/natural segments in parallel. For fast/turbo, requested_emotion was
        # normalized to "shared" above, so each thread still keeps only one shared backend.
        engines_by_emotion = getattr(_VIENEU_THREAD_LOCAL, "engines_by_emotion", None)
        if engines_by_emotion is None:
            engines_by_emotion = {}
            _VIENEU_THREAD_LOCAL.engines_by_emotion = engines_by_emotion
        engine = engines_by_emotion.get(requested_emotion)
        if engine is not None:
            return engine
        with _VIENEU_ENGINE_INIT_LOCK:
            engine = engines_by_emotion.get(requested_emotion)
            if engine is None:
                thread_name = threading.current_thread().name
                slot = f"{thread_name}_{requested_emotion}"
                engine = _init_vieneu_engine_for_slot(slot, emotion=requested_emotion)
                engines_by_emotion[requested_emotion] = engine
            return engine

    if _VIENEU_TTS_SINGLETON is not None:
        return _VIENEU_TTS_SINGLETON
    with _VIENEU_ENGINE_INIT_LOCK:
        if _VIENEU_TTS_SINGLETON is None:
            _VIENEU_TTS_SINGLETON = _init_vieneu_engine_for_slot("shared", emotion=requested_emotion)
        return _VIENEU_TTS_SINGLETON


def preload_vieneu_engines():
    strategy = os.getenv("VIENEU_ENGINE_STRATEGY", "shared_single_engine").strip().lower()
    if strategy not in {"dual_preload", "dual_preload_engine", "dual_engine", "two_engine", "two_engines"}:
        if strategy in {"shared_single", "shared_single_engine", "shared", "single"} and vieneu_preload_shared_engine_enabled():
            print(f"VieNeuTTS preloading shared engine early: strategy={strategy}", flush=True)
            engine = get_vieneu_engine(get_vieneu_emotion())
            return {"strategy": strategy, "preloaded": ["shared"], "emotion_runtime": "infer_emotion_tag" if not vieneu_init_supports_emotion() else "init_kwarg"}
        if strategy in {"per_thread", "per_thread_engine", "per_worker", "per_worker_engine", "parallel", "multi_worker_engine", "worker_pool_engine"}:
            print(
                f"VieNeuTTS preload skipped: strategy={strategy}; engines are created lazily per worker thread and per emotion.",
                flush=True,
            )
        else:
            print(f"VieNeuTTS preload skipped: strategy={strategy}", flush=True)
        return {"strategy": strategy, "preloaded": [], "emotion_runtime": "infer_emotion_tag" if not vieneu_init_supports_emotion() else "init_kwarg"}

    narration = get_vieneu_narration_emotion()
    dialogue = get_vieneu_dialogue_emotion()

    if not vieneu_init_supports_emotion():
        # Fast/LMDeploy does not accept emotion= at init. Loading two engines would duplicate
        # the same model and waste VRAM. Keep one shared engine and pass emotion_tag per segment.
        print(
            "ℹ️ VieNeu fast/turbo backend does not support emotion= at init; "
            "preloading one shared engine and applying emotion_tag per segment.",
            flush=True,
        )
        try:
            get_vieneu_engine(narration)
            return {"strategy": strategy, "preloaded": ["shared"], "emotion_runtime": "infer_emotion_tag"}
        except Exception as exc:
            if os.getenv("VIENEU_DUAL_FALLBACK_TO_SINGLE_SWITCH", "1").strip().lower() in {"1", "true", "yes", "on"}:
                print(f"⚠️ preload shared engine failed: {exc}", flush=True)
                os.environ["VIENEU_ENGINE_STRATEGY"] = "single_switch_engine"
                return {"strategy": "single_switch_engine", "preloaded": [], "fallback_reason": str(exc), "emotion_runtime": "infer_emotion_tag"}
            raise

    loaded = []
    try:
        print(f"🚀 Preloading VieNeuTTS narration engine: emotion={narration}", flush=True)
        get_vieneu_engine(narration)
        loaded.append(narration)
        if dialogue != narration:
            print(f"🚀 Preloading VieNeuTTS dialogue engine: emotion={dialogue}", flush=True)
            get_vieneu_engine(dialogue)
            loaded.append(dialogue)
        print(f"✅ VieNeuTTS preload done: {loaded}", flush=True)
        return {"strategy": strategy, "preloaded": loaded, "emotion_runtime": "init_kwarg"}
    except Exception as exc:
        if os.getenv("VIENEU_DUAL_FALLBACK_TO_SINGLE_SWITCH", "1").strip().lower() in {"1", "true", "yes", "on"}:
            print(f"⚠️ dual_preload_engine failed: {exc}", flush=True)
            print("⚠️ Fallback to single_switch_engine to avoid VRAM OOM.", flush=True)
            _VIENEU_TTS_ENGINES_BY_EMOTION.clear()
            _free_vieneu_gpu_memory()
            os.environ["VIENEU_ENGINE_STRATEGY"] = "single_switch_engine"
            return {"strategy": "single_switch_engine", "preloaded": [], "fallback_reason": str(exc), "emotion_runtime": "init_kwarg"}
        raise


def _norm_text_for_match(value):
    value = str(value or "").strip().lower()
    value = value.replace("_", " ").replace("-", " ")
    value = re.sub(r"\s+", " ", value)
    return value


def _find_preset_voice_id(tts, wanted_key):
    global _VIENEU_PRESET_LIST_CACHE
    wanted_key = VIENEU_VOICE_ALIASES.get(str(wanted_key or "").strip()) or VIENEU_VOICE_ALIASES.get(str(wanted_key or "").strip().lower()) or wanted_key or VIENEU_DEFAULT_VOICE
    if _VIENEU_PRESET_LIST_CACHE is None:
        try:
            _VIENEU_PRESET_LIST_CACHE = list(tts.list_preset_voices())
        except Exception:
            _VIENEU_PRESET_LIST_CACHE = []
    voices = _VIENEU_PRESET_LIST_CACHE
    hints = VIENEU_PRESET_HINTS.get(wanted_key, [wanted_key])
    for item in voices:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            display, voice_id = str(item[0]), str(item[1])
        else:
            display, voice_id = str(item), str(item)
        haystack = _norm_text_for_match(display + " " + voice_id)
        if any(_norm_text_for_match(hint) in haystack for hint in hints):
            return voice_id
    if voices:
        if wanted_key == "vinh" and len(voices) >= 2:
            return voices[1][1] if isinstance(voices[1], (list, tuple)) and len(voices[1]) >= 2 else str(voices[1])
        return voices[0][1] if isinstance(voices[0], (list, tuple)) and len(voices[0]) >= 2 else str(voices[0])
    return None


def get_vieneu_voice_data(voice_key, tts=None):
    voice_key = normalize_voice_name(voice_key, default_voice=VIENEU_DEFAULT_VOICE)
    tts = tts or get_vieneu_engine()
    cache_key = (id(tts), voice_key)
    if cache_key in _VIENEU_PRESET_VOICE_CACHE:
        return _VIENEU_PRESET_VOICE_CACHE[cache_key]
    voice_id = _find_preset_voice_id(tts, voice_key)
    if not voice_id:
        _VIENEU_PRESET_VOICE_CACHE[cache_key] = None
        return None
    try:
        voice_data = tts.get_preset_voice(voice_id)
    except Exception:
        voice_data = None
    _VIENEU_PRESET_VOICE_CACHE[cache_key] = voice_data
    return voice_data


async def get_voices():
    return {
        "Đoan - VieNeu preset (nữ miền Nam)": "doan",
        "Vĩnh - VieNeu preset (nam miền Nam)": "vinh",
    }


def normalize_voice_name(value, default_voice=VIENEU_DEFAULT_VOICE):
    raw = str(value or "").strip()
    if not raw:
        return default_voice
    return VIENEU_VOICE_ALIASES.get(raw) or VIENEU_VOICE_ALIASES.get(raw.lower()) or VIENEU_VOICE_ALIASES.get(_norm_text_for_match(raw)) or raw


def _segment_nested_gender(segment):
    """Return the best-effort semantic gender from a segment.

    Only used for dialogue routing. Non-dialogue must stay female by user rule.
    """
    segment = segment or {}
    candidates = [
        segment.get("gender"),
        segment.get("speaker_gender"),
        segment.get("voice_gender"),
    ]
    for field in ("speaker_source", "performed_voice_persona", "detection"):
        obj = segment.get(field)
        if isinstance(obj, dict):
            candidates.append(obj.get("gender"))
    for value in candidates:
        raw = str(value or "").strip().lower()
        if raw in {"male", "nam", "man", "m", "boy", "adult_male", "elder_male"}:
            return "male"
        if raw in {"female", "nữ", "nu", "woman", "f", "girl", "adult_female", "elder_female"}:
            return "female"
    return "unknown"


def resolve_vieneu_voice_for_segment(segment, defaults=None):
    """Strict southern voice routing.

    Required rule:
      - non-dialogue all -> Doan / female southern
      - dialogue male    -> Vinh / male southern
      - dialogue female/unknown/neutral -> Doan / female southern

    This intentionally ignores segment["voice"] for non-dialogue so JSON mistakes
    cannot make narration switch to male.
    """
    segment = segment or {}
    defaults = defaults or {}

    non_dialogue_voice = normalize_voice_name(
        os.getenv("VIENEU_NON_DIALOGUE_VOICE", defaults.get("non_dialogue_voice") or "doan"),
        default_voice="doan",
    )
    dialogue_female_voice = normalize_voice_name(
        os.getenv("VIENEU_DIALOGUE_FEMALE_VOICE", defaults.get("dialogue_female_voice") or "doan"),
        default_voice="doan",
    )
    dialogue_male_voice = normalize_voice_name(
        os.getenv("VIENEU_DIALOGUE_MALE_VOICE", defaults.get("dialogue_male_voice") or "vinh"),
        default_voice="vinh",
    )

    segment_type = str(segment.get("segment_type") or "").strip().lower()

    # Important: only actual dialogue is allowed to become male.
    if segment_type != "dialogue":
        return non_dialogue_voice

    # Prefer explicit voice only inside dialogue, and only if it clearly maps to Vinh/Doan.
    explicit_voice = normalize_voice_name(segment.get("voice"), default_voice="")
    if explicit_voice == "vinh":
        return dialogue_male_voice
    if explicit_voice == "doan":
        return dialogue_female_voice

    gender = _segment_nested_gender(segment)
    if gender == "male":
        return dialogue_male_voice

    return dialogue_female_voice


def resolve_segment_tts_settings(segment, defaults):
    defaults = defaults or {}
    voice = resolve_vieneu_voice_for_segment(segment, defaults)
    return {
        "voice": voice,
        "rate_pct": int(segment.get("rate_pct", defaults.get("default_rate_pct", 0))),
        "pitch_hz": int(segment.get("pitch_hz", defaults.get("default_pitch_hz", 0))),
        "pause_after_ms": int(segment.get("pause_after_ms", defaults.get("default_pause_after_ms", 220))),
        "emotion": resolve_vieneu_emotion_for_segment(segment, defaults) if "resolve_vieneu_emotion_for_segment" in globals() else segment.get("emotion", defaults.get("default_emotion", os.getenv("VIENEU_EMOTION", "storytelling"))),
    }


def resolve_voice_profile(segment, defaults):
    return resolve_segment_tts_settings(segment, defaults)


def _convert_any_audio_to_mp3(input_audio, output_path):
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        shutil.copyfile(input_audio, output_path)
        return output_path
    subprocess.run([
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(input_audio),
        "-ar", str(VIENEU_SAMPLE_RATE), "-ac", "1", "-codec:a", "libmp3lame", "-q:a", "4", str(output_path),
    ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_path


def _vieneu_save_audio(tts, audio, output_path):
    output_path = str(output_path)
    tmp_wav = output_path + ".vieneu_tmp.wav"
    try:
        tts.save(audio, tmp_wav)
    except Exception:
        tts.save(audio, output_path)
        if os.path.exists(output_path):
            return output_path
        raise
    _convert_any_audio_to_mp3(tmp_wav, output_path)
    try:
        os.remove(tmp_wav)
    except Exception:
        pass
    return output_path


async def synthesize_audio_once(text, voice, rate, pitch, audio_path, timeout_sec=None, emotion=None):
    cleaned = clean_tts_text(text)
    if not cleaned:
        raise ValueError("Text is empty after cleanup.")
    voice_key = normalize_voice_name(voice, default_voice=VIENEU_DEFAULT_VOICE)

    def _run():
        tts = get_vieneu_engine(emotion=emotion)
        voice_data = get_vieneu_voice_data(voice_key, tts=tts)
        log_vieneu_stability_once()
        infer_kwargs = build_vieneu_infer_kwargs_for_stability(tts)
        speed_values = get_vieneu_standard_speed_defaults()
        if "max_chars" in speed_values:
            infer_kwargs["max_chars"] = int(speed_values["max_chars"])
        if "apply_watermark" in speed_values:
            infer_kwargs["apply_watermark"] = bool(speed_values["apply_watermark"])
        if not vieneu_init_supports_emotion():
            # Fast/LMDeploy does not accept emotion= in __init__. Always pass the
            # emotion_tag key at infer-time so the backend receives an explicit
            # style decision for every segment:
            #   dialogue/natural       -> "<|emotion_0|>"
            #   non-dialogue/storytelling -> None (official storytelling/default tag)
            emotion_tag = vieneu_emotion_tag_for_infer(emotion)
            infer_kwargs["emotion_tag"] = emotion_tag
        if voice_data is not None:
            infer_kwargs["voice"] = voice_data
        audio = tts.infer(text=cleaned, **infer_kwargs)
        return _vieneu_save_audio(tts, audio, audio_path)

    timeout_sec = int(timeout_sec or SEGMENT_TTS_TIMEOUT_SEC)
    try:
        await asyncio.wait_for(asyncio.to_thread(_run), timeout=timeout_sec)
    except asyncio.TimeoutError as exc:
        if os.path.exists(audio_path):
            os.remove(audio_path)
        raise TimeoutError(
            f"vieneu_tts timeout after {timeout_sec}s (voice={voice_key}, emotion={emotion or get_vieneu_emotion()}, text_len={len(cleaned)}, rate_ignored={rate}, pitch_ignored={pitch})"
        ) from exc
    duration = float(get_audio_duration_sec(audio_path) or estimate_duration_from_text(cleaned, 0))
    return [{"start": 0.0, "end": round(duration, 3), "text": cleaned}]


async def synthesize_audio(text, voice, rate, pitch, audio_path, timeout_sec=None, emotion=None):
    chunks = split_text_into_tts_chunks(text, max_chars=SEGMENT_TTS_MAX_CHARS)
    if not chunks:
        raise ValueError("Text is empty after cleanup.")
    if len(chunks) == 1:
        return await synthesize_audio_once(chunks[0], voice, rate, pitch, audio_path, timeout_sec=timeout_sec, emotion=emotion)

    audio_dir = os.path.dirname(audio_path) or "."
    stem = Path(audio_path).stem
    suffix = Path(audio_path).suffix or ".mp3"
    part_files = []
    merged_events = []
    cursor = 0.0
    for idx, chunk_text in enumerate(chunks, start=1):
        chunk_path = os.path.join(audio_dir, f"{stem}_chunk_{idx:04d}{suffix}")
        last_error = None
        for attempt in range(1, SEGMENT_TTS_RETRIES + 1):
            try:
                chunk_events = await synthesize_audio_once(chunk_text, voice, rate, pitch, chunk_path, timeout_sec=timeout_sec, emotion=emotion)
                chunk_events = ensure_event_fallback(chunk_events, chunk_text, rate)
                merged_events.extend(offset_events(chunk_events, cursor))
                event_duration = max(event["end"] for event in chunk_events)
                actual_duration = get_audio_duration_sec(chunk_path)
                chunk_duration = max(float(actual_duration or 0), float(event_duration or 0))
                cursor = round(cursor + chunk_duration, 3)
                part_files.append(chunk_path)
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)
                if attempt < SEGMENT_TTS_RETRIES:
                    await asyncio.sleep(SEGMENT_RETRY_BACKOFF_SEC * attempt)
        if last_error is not None:
            for part_file in part_files:
                if os.path.exists(part_file):
                    os.remove(part_file)
            raise RuntimeError(
                f"Chunk {idx}/{len(chunks)} failed after {SEGMENT_TTS_RETRIES} retries "
                f"(voice={voice}, emotion={emotion or get_vieneu_emotion()}, text_len={len(chunk_text)}): {last_error}"
            ) from last_error
    concat_audio_files(part_files, audio_path)
    for part_file in part_files:
        if os.path.exists(part_file):
            os.remove(part_file)
    return merged_events


def segment_setting_key(segment, defaults=None):
    profile = resolve_segment_tts_settings(segment, defaults or DEFAULT_MANIFEST_STRATEGY)
    return (profile["voice"], 0, 0, profile.get("emotion", get_vieneu_emotion()))


def _copy_tree_contents(src_dir, dst_dir, filter_fn=None):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    copied = 0
    skipped = 0
    if not src_dir.exists():
        return {"copied": copied, "skipped": skipped}
    for src in src_dir.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(src_dir).as_posix()
        if filter_fn is not None and not filter_fn(rel):
            skipped += 1
            continue
        dest = dst_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        copied += 1
    return {"copied": copied, "skipped": skipped}


def _clear_local_drive_raw_success_payload_if_migrated(cache_name):
    if not has_local_final_parts_for_manifest(cache_name):
        return {"deleted": 0, "skipped": True, "reason": "local final_parts not ready; raw success kept"}
    remote_prefix = build_remote_cache_prefix(cache_name).strip("/")
    root = GOOGLE_DRIVE_OUTPUT_ROOT / remote_prefix
    deleted = 0
    success_dir = root / SUCCESS_CACHE_DIRNAME
    if success_dir.exists():
        shutil.rmtree(success_dir, ignore_errors=True)
        deleted += 1
    if root.exists():
        for path in list(root.iterdir()):
            if path.is_file() and (path.suffix == ".mp3" or (path.suffix == ".json" and not path.name.endswith(".failed.json") and path.name != CACHE_INDEX_FILENAME)):
                path.unlink(missing_ok=True)
                deleted += 1
    return {"deleted": deleted, "skipped": False}


def ensure_dataset_exists():
    global _DATASET_READY
    if api is None:
        GOOGLE_DRIVE_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        _DATASET_READY = True
        print(f"HF dataset disabled. Using Google Drive/local root: {GOOGLE_DRIVE_OUTPUT_ROOT}")
        return
    if _DATASET_READY:
        return
    try:
        api.repo_info(repo_id=DATASET_REPO, repo_type="dataset")
        print(f"Dataset {DATASET_REPO} is ready.")
        _DATASET_READY = True
    except Exception:
        try:
            api.create_repo(repo_id=DATASET_REPO, repo_type="dataset", private=False)
            print(f"Created dataset: {DATASET_REPO}")
            _DATASET_READY = True
        except Exception as exc:
            print(f"Unable to create dataset: {exc}")


def upload_staged_folder_once(staging_root, commit_message=None):
    if api is None:
        if not USE_GOOGLE_DRIVE_WHEN_NO_HF:
            raise RuntimeError("HF upload is disabled and Google Drive fallback is off.")
        GOOGLE_DRIVE_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        stats = _copy_tree_contents(staging_root, GOOGLE_DRIVE_OUTPUT_ROOT)
        return f"Copied staged folder to Google Drive/local root {GOOGLE_DRIVE_OUTPUT_ROOT} (copied={stats.get('copied', 0)}, skipped={stats.get('skipped', 0)})."
    ensure_dataset_exists()
    if not os.path.isdir(staging_root):
        raise RuntimeError(f"Batch upload staging folder does not exist: {staging_root}")
    has_files = any(Path(staging_root).rglob("*"))
    if not has_files:
        return "No files staged for upload."
    api.upload_folder(folder_path=staging_root, path_in_repo="", repo_id=DATASET_REPO, repo_type="dataset", commit_message=commit_message or "Batch upload VieNeu TTS outputs", ignore_patterns=["_sync_state.json"])
    return f"Uploaded staged folder to dataset {DATASET_REPO} in one commit."


def upload_manifest_cache_folder_once(cache_name, commit_message=None):
    cache_name = build_manifest_cache_name(cache_name)
    manifest_cache_dir = build_manifest_cache_dir(cache_name)
    if not manifest_cache_dir.exists() or not any(manifest_cache_dir.rglob("*")):
        return f"Manifest cache commit skipped: no files in {manifest_cache_dir}."
    remote_prefix = build_remote_cache_prefix(cache_name)
    if api is None:
        GOOGLE_DRIVE_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        raw_cleanup = _clear_local_drive_raw_success_payload_if_migrated(cache_name)
        remote_dir = GOOGLE_DRIVE_OUTPUT_ROOT / remote_prefix
        stats = _copy_tree_contents(manifest_cache_dir, remote_dir, filter_fn=should_upload_remote_cache_rel)
        if stats.get("copied", 0) <= 0:
            return f"Manifest cache copy skipped: no uploadable compact cache files in {manifest_cache_dir}."
        return (
            f"Copied compact manifest cache folder to Google Drive/local cache: {remote_dir} "
            f"| uploaded={stats.get('copied', 0)} skipped_raw_success={stats.get('skipped', 0)} "
            f"| local_raw_success_deleted={raw_cleanup.get('deleted', 0)}"
            + (f" | raw_success_cleanup_skipped={raw_cleanup.get('reason')}" if raw_cleanup.get('skipped') else "")
        )
    ensure_dataset_exists()
    raw_cleanup = clear_remote_raw_success_payload_if_migrated(cache_name)
    tmp = tempfile.TemporaryDirectory(prefix=f"compact_vieneu_cache_{sanitize_filename(cache_name)}_")
    try:
        filtered_dir = Path(tmp.name) / cache_name
        filtered_dir.mkdir(parents=True, exist_ok=True)
        snapshot_stats = build_filtered_manifest_cache_snapshot(manifest_cache_dir, filtered_dir)
        if not any(filtered_dir.rglob("*")):
            return f"Manifest cache commit skipped: no uploadable compact cache files in {manifest_cache_dir}."
        api.upload_folder(
            folder_path=str(filtered_dir),
            path_in_repo=remote_prefix,
            repo_id=DATASET_REPO,
            repo_type="dataset",
            commit_message=commit_message or f"Commit compact VieNeu TTS cache folder: {cache_name}",
        )
        return (
            f"Committed compact manifest cache folder in one commit: {remote_prefix} "
            f"| uploaded={snapshot_stats.get('copied', 0)} skipped_raw_success={snapshot_stats.get('skipped', 0)} "
            f"| remote_raw_success_deleted={raw_cleanup.get('deleted', 0)}"
            + (f" | raw_success_cleanup_skipped={raw_cleanup.get('reason')}" if raw_cleanup.get('skipped') else "")
        )
    finally:
        tmp.cleanup()


def upload_chunk_cache_file(local_file, cache_name, remote_name):
    remote_name = str(remote_name).lstrip("/")
    if api is None:
        dest = GOOGLE_DRIVE_OUTPUT_ROOT / build_remote_cache_prefix(cache_name) / remote_name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_file, dest)
        return
    ensure_dataset_exists()
    api.upload_file(path_or_fileobj=local_file, path_in_repo=f"{build_remote_cache_prefix(cache_name)}/{remote_name}", repo_id=DATASET_REPO, repo_type="dataset")


def download_chunk_cache_file(cache_name, remote_name, local_path):
    remote_name = str(remote_name).lstrip("/")
    if api is None:
        src = GOOGLE_DRIVE_OUTPUT_ROOT / build_remote_cache_prefix(cache_name) / remote_name
        if not src.exists():
            return False
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        shutil.copyfile(src, local_path)
        return True
    if hf_hub_download is None:
        return False
    try:
        ensure_dataset_exists()
        downloaded = hf_hub_download(repo_id=DATASET_REPO, repo_type="dataset", filename=f"{build_remote_cache_prefix(cache_name)}/{remote_name}", token=HF_TOKEN)
    except Exception:
        return False
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    shutil.copyfile(downloaded, local_path)
    return True


def sync_remote_cache_folder_to_local(cache_name, clear_local_first=True, cache_scope=CACHE_SCOPE_ALL):
    cache_name = build_manifest_cache_name(cache_name)
    cache_scope = str(cache_scope or CACHE_SCOPE_ALL)
    if cache_scope not in {CACHE_SCOPE_ALL, CACHE_SCOPE_FAILED_ONLY, CACHE_SCOPE_SUCCESS_ONLY}:
        cache_scope = CACHE_SCOPE_ALL
    remote_prefix = build_remote_cache_prefix(cache_name).strip("/")
    local_dir = build_manifest_cache_dir(cache_name)
    if clear_local_first and local_dir.exists():
        shutil.rmtree(local_dir, ignore_errors=True)
    local_dir.mkdir(parents=True, exist_ok=True)
    get_segment_success_dir(local_dir)
    get_segment_failed_dir(local_dir)
    if api is not None:
        try:
            ensure_dataset_exists()
            repo_files = api.list_repo_files(repo_id=DATASET_REPO, repo_type="dataset", token=HF_TOKEN)
            matched = [p for p in repo_files if _remote_cache_path_matches_scope(p, remote_prefix, cache_scope)]
            copied = 0
            errors = []
            for remote_path in matched:
                try:
                    downloaded = hf_hub_download(repo_id=DATASET_REPO, repo_type="dataset", filename=remote_path, token=HF_TOKEN)
                    rel = Path(remote_path).relative_to(remote_prefix)
                    dest = local_dir / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(downloaded, dest)
                    copied += 1
                except Exception as exc:
                    if len(errors) < 5:
                        errors.append(f"{remote_path}: {exc}")
            return {"enabled": True, "status": "synced" if copied else "empty", "method": "hf_hub_download_fallback", "files": copied, "errors": errors, "remote_prefix": remote_prefix, "local_dir": str(local_dir.resolve()), "cache_scope": cache_scope}
        except Exception as exc:
            return {"enabled": True, "status": "failed", "reason": str(exc), "files": 0, "remote_prefix": remote_prefix, "local_dir": str(local_dir.resolve()), "cache_scope": cache_scope}
    src_dir = GOOGLE_DRIVE_OUTPUT_ROOT / remote_prefix
    if not src_dir.exists():
        return {"enabled": True, "status": "empty", "reason": "no Google Drive/local cache folder matched", "files": 0, "remote_prefix": remote_prefix, "local_dir": str(local_dir.resolve()), "cache_scope": cache_scope}
    copied = 0
    for src in src_dir.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(src_dir).as_posix()
        fake_remote = remote_prefix + "/" + rel
        if not _remote_cache_path_matches_scope(fake_remote, remote_prefix, cache_scope):
            continue
        dest = local_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        copied += 1
    return {"enabled": True, "status": "synced" if copied else "empty", "method": "google_drive_local_copy", "files": copied, "remote_prefix": remote_prefix, "local_dir": str(local_dir.resolve()), "cache_scope": cache_scope}


# ==============================================================================
# Pure-Colab compact cache policy
# ==============================================================================
# Success segment files are only temporary working files. After each render pass we
# prebuild final_parts/ for every contiguous success range, including single
# isolated successes (FINAL_PART_MIN_SEGMENTS = 1), then remove raw success/*.mp3
# and success/*.json that are covered by final_parts. Failed markers remain under
# failed/. This keeps Drive/HF compact while preserving resume/retry behavior.

def _segment_stems_covered_by_final_parts(manifest_cache_dir):
    covered = set()
    parts_dir = get_final_parts_dir(manifest_cache_dir)
    if not parts_dir.exists():
        return covered
    for meta_path in parts_dir.glob("*.json"):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            for sig in meta.get("signatures") or []:
                order = int(sig.get("order", 0) or 0)
                seg_id = str(sig.get("segment_id") or "").strip() or f"seg_{order:06d}"
                stem = f"segment_{order:06d}__{sanitize_filename(seg_id)}"
                covered.add(stem)
        except Exception:
            continue
    return covered


def compact_manifest_success_cache(manifest_cache_dir, ordered_segments, rendered_by_segment_id, defaults, tmpdir, timeline_mode, progress_callback=None, total=None, failed_count=0, phase_label="compact_cache"):
    """Build final_parts from current successes and prune raw success cache.

    This runs even when the manifest is incomplete. That way a Colab timeout or
    later retry can resume from compact final_parts without rerendering successes.
    """
    if not rendered_by_segment_id:
        return {"parts": [], "raw_success_deleted": 0, "raw_success_kept": 0}

    old_min = globals().get("FINAL_PART_MIN_SEGMENTS", 1)
    globals()["FINAL_PART_MIN_SEGMENTS"] = 1
    try:
        parts = prebuild_contiguous_success_parts(
            ordered_segments,
            rendered_by_segment_id,
            defaults,
            manifest_cache_dir,
            tmpdir,
            timeline_mode,
            progress_callback=progress_callback,
            total=total or len(ordered_segments),
            failed_count=failed_count,
            phase_label=phase_label,
        )
    finally:
        globals()["FINAL_PART_MIN_SEGMENTS"] = old_min

    covered_stems = _segment_stems_covered_by_final_parts(manifest_cache_dir)
    success_dir = get_segment_success_dir(manifest_cache_dir)
    deleted = 0
    kept = 0
    if success_dir.exists():
        for path in list(success_dir.glob("*")):
            if not path.is_file():
                continue
            stem = _cache_file_base_name(path)
            if stem in covered_stems:
                try:
                    path.unlink(missing_ok=True)
                    deleted += 1
                except Exception:
                    kept += 1
            else:
                kept += 1
    return {"parts": parts, "raw_success_deleted": deleted, "raw_success_kept": kept}


if __name__ == "__main__":
    demo = asyncio.run(create_demo())
    demo.queue()
    demo.launch(share=True, debug=True)
