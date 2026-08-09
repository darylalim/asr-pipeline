import re
import tempfile
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import unquote, urlparse
from urllib.request import urlopen

import mlx_whisper
import streamlit as st
import yt_dlp
from mlx_whisper.tokenizer import LANGUAGES
from streamlit.runtime.uploaded_file_manager import UploadedFile

ASR_MODEL_REPO = "mlx-community/whisper-large-v3-turbo"
# Ordered most-likely-first, and deliberately short: the uploader dropzone lists
# these on one `text-overflow: ellipsis` line, so a long list truncates mid-word.
# See "Accepted formats" in CLAUDE.md before adding to either tuple.
AUDIO_FORMATS = (
    "mp3",
    "m4a",
    "wav",
    "opus",
)
VIDEO_FORMATS = (
    "mp4",
    "mov",
    "webm",
    "mkv",
)
LANGUAGE_CODES: list[str | None] = [None] + sorted(LANGUAGES, key=lambda c: LANGUAGES[c])
YOUTUBE_URL_RE = re.compile(r"^https?://(www\.|m\.)?(youtube\.com/|youtu\.be/)", re.IGNORECASE)
URL_RE = re.compile(r"^https?://", re.IGNORECASE)
MAX_URL_DOWNLOAD_BYTES = 500 * 1024 * 1024
PAGE_CONFIG: dict[str, Any] = {
    "page_title": "Whisper Transcribe",
    "page_icon": ":material/graphic_eq:",
    "layout": "centered",
}


class _RemoteAudio:
    def __init__(self, name: str, data: bytes) -> None:
        self.name = name
        self._data = data

    def read(self) -> bytes:
        return self._data


@st.cache_data(show_spinner="Downloading audio from YouTube...", max_entries=5, ttl="1h")
def _fetch_youtube_audio(url: str) -> tuple[bytes, str]:
    with tempfile.TemporaryDirectory() as tmpdir:
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": str(Path(tmpdir) / "%(title)s.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "restrictfilenames": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            downloaded = Path(ydl.prepare_filename(info))
        return downloaded.read_bytes(), downloaded.name


@st.cache_data(show_spinner="Downloading audio from URL...", max_entries=5, ttl="1h")
def _fetch_url_audio(url: str) -> tuple[bytes, str]:
    with urlopen(url, timeout=60) as resp:
        data = resp.read(MAX_URL_DOWNLOAD_BYTES + 1)
    if len(data) > MAX_URL_DOWNLOAD_BYTES:
        raise RuntimeError(f"URL response exceeds {MAX_URL_DOWNLOAD_BYTES // (1024 * 1024)} MB")
    filename = unquote(Path(urlparse(url).path).name) or "download"
    return data, filename


def _format_language(code: str | None) -> str:
    return "Detect" if code is None else LANGUAGES[code].title()


def _format_timestamp(seconds: float, decimal_marker: str = ".") -> str:
    ms = round(seconds * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d}{decimal_marker}{ms:03d}"


def _format_srt(result: dict) -> str:
    return "\n".join(
        f"{i}\n"
        f"{_format_timestamp(s['start'], decimal_marker=',')} --> "
        f"{_format_timestamp(s['end'], decimal_marker=',')}\n"
        f"{s['text'].strip().replace('-->', '->')}\n"
        for i, s in enumerate(result["segments"], start=1)
    )


_MARKDOWN_ESCAPE_RE = re.compile(r"([\\`*_~\[\]:$])")


def _escape_markdown(text: str) -> str:
    """Backslash-escape characters Streamlit's label-subset Markdown interprets.

    st.subheader renders the Markdown label subset, so a filename containing *, _,
    backticks, brackets, or : (emoji/Material-icon directives) — common in YouTube
    titles and underscored names — would otherwise mis-render. Escaping keeps the
    displayed name literal.
    """
    return _MARKDOWN_ESCAPE_RE.sub(r"\\\1", text)


def _validate_time_range(raw: str) -> str | None:
    """Return an error message if the time-range string is malformed, else None.

    Valid forms: blank (full file) or comma-separated non-negative seconds where
    each complete start,end pair has end > start (e.g. "30,90" or "0,60,120,180").
    A trailing unpaired value is a start that runs to the end of the file (e.g.
    "30" or "0,60,120"), matching mlx_whisper.transcribe's clip_timestamps.
    """
    if not raw:
        return None
    values: list[float] = []
    for token in (t.strip() for t in raw.split(",")):
        if not token:
            return "Time range has an empty value (check for a stray comma)."
        try:
            value = float(token)
        except ValueError:
            return f"Invalid time range: {token!r} is not a number."
        if value < 0:
            return "Time range values must be non-negative."
        values.append(value)
    for start, end in zip(values[::2], values[1::2]):
        if end <= start:
            return f"Time range end ({end:g}) must be greater than start ({start:g})."
    for prev, cur in pairwise(values):
        if cur < prev:
            return "Time range values must be in increasing order."
    return None


@st.cache_data(show_spinner=False, max_entries=20)
def _transcribe(
    audio_bytes: bytes,
    suffix: str,
    *,
    language: str | None = None,
    task: str = "transcribe",
    initial_prompt: str | None = None,
    no_verbatim: bool = False,
    condition_on_previous_text: bool = True,
    clip_timestamps: str = "0",
) -> dict:
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        result = mlx_whisper.transcribe(
            tmp.name,
            path_or_hf_repo=ASR_MODEL_REPO,
            language=language,
            task=task,
            initial_prompt=initial_prompt,
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.4,
            condition_on_previous_text=condition_on_previous_text,
            word_timestamps=no_verbatim,
            hallucination_silence_threshold=2.0 if no_verbatim else None,
            clip_timestamps=clip_timestamps,
        )
    if not result.get("text", "").strip():
        raise RuntimeError("Transcription produced no text")
    return result


def _handle_transcription(
    uploaded_files: Sequence[UploadedFile | _RemoteAudio],
    *,
    language: str | None,
    task: str,
    include_subtitles: bool,
    initial_prompt: str | None = None,
    no_verbatim: bool = False,
    condition_on_previous_text: bool = True,
    clip_timestamps: str = "0",
) -> None:
    transcriptions: list[dict] = []
    # Publish up front so a previous batch is cleared even if nothing succeeds
    # here. Streamlit interrupts a running script at the next ForwardMsg — the
    # status.update() below is such a point, and RerunException is a
    # BaseException that `except Exception` will not catch — so assigning only
    # after the loop would discard every file already transcribed.
    st.session_state["transcription"] = transcriptions
    total = len(uploaded_files)
    with st.status(f"Transcribing {total} file(s)...", expanded=True) as status:
        for i, uploaded_file in enumerate(uploaded_files, start=1):
            status.update(label=f"Transcribing {uploaded_file.name} ({i}/{total})...")
            name = Path(uploaded_file.name)
            try:
                result = _transcribe(
                    uploaded_file.read(),
                    name.suffix,
                    language=language,
                    task=task,
                    initial_prompt=initial_prompt,
                    no_verbatim=no_verbatim,
                    condition_on_previous_text=condition_on_previous_text,
                    clip_timestamps=clip_timestamps,
                )
                transcriptions.append(
                    {
                        "result": result,
                        "file_stem": f"{name.stem}_{name.suffix.lstrip('.')}_transcript",
                        "filename": uploaded_file.name,
                        "include_subtitles": include_subtitles,
                    }
                )
                # Redundant while session_state holds this exact list (append
                # mutates it in place), but kept explicit so the progressive
                # publish does not silently break if `transcriptions` is ever
                # rebound rather than mutated.
                st.session_state["transcription"] = transcriptions
            except RuntimeError as e:
                st.error(f"Transcription failed for {uploaded_file.name}: {e}")
            except Exception as e:
                st.error(f"Unexpected error for {uploaded_file.name}: {e}")
                st.exception(e)
        status.update(
            label=f"Transcribed {len(transcriptions)}/{total} file(s)",
            state="complete",
        )


def _labeled_toggle(label: str, help_text: str) -> bool:
    label_col, input_col = st.columns([3, 1], vertical_alignment="center")
    with label_col:
        st.markdown(label, help=help_text)
    with input_col, st.container(horizontal_alignment="right"):
        return st.toggle(label, value=False, label_visibility="collapsed")


def _field_label(label: str, help_text: str) -> None:
    label_col, _ = st.columns([3, 1], vertical_alignment="center")
    with label_col:
        st.markdown(label, help=help_text)


def _transcription_kwargs(
    *,
    language: str | None,
    translate: bool,
    include_subtitles: bool,
    initial_prompt: str | None,
    no_verbatim: bool,
    decode_independently: bool,
    clip_timestamps: str,
) -> dict:
    return {
        "language": language,
        "task": "translate" if translate else "transcribe",
        "include_subtitles": include_subtitles,
        "initial_prompt": initial_prompt,
        "no_verbatim": no_verbatim,
        "condition_on_previous_text": not decode_independently,
        "clip_timestamps": clip_timestamps,
    }


def _display_transcription() -> None:
    transcriptions = st.session_state.get("transcription") or []
    for i, data in enumerate(transcriptions):
        include_subtitles = data["include_subtitles"]
        if include_subtitles:
            initial = _format_srt(data["result"])
        else:
            initial = data["result"]["text"].strip()
        st.subheader(_escape_markdown(data["filename"]))
        transcript = st.text_area(
            "Transcript",
            initial,
            height=300,
            label_visibility="collapsed",
            key=f"transcript_{i}",
        )
        ext, mime = ("srt", "application/x-subrip") if include_subtitles else ("txt", "text/plain")
        _, download_col = st.columns([3, 1])
        with download_col:
            st.download_button(
                "Download",
                transcript,
                f"{data['file_stem']}.{ext}",
                mime,
                icon=":material/download:",
                key=f"download_{ext}_{i}",
                help="Downloads as .srt when subtitles are enabled, .txt otherwise.",
                width="stretch",
            )


# UI
st.set_page_config(**PAGE_CONFIG)
st.title("Whisper Transcribe")

upload_tab, record_tab, youtube_tab, url_tab = st.tabs(
    [
        ":material/upload: Upload",
        ":material/mic: Record",
        ":material/smart_display: YouTube",
        ":material/link: URL",
    ],
    # on_change="rerun" enables the per-tab `.open` flag used to gate the remote
    # fetches below; `key` exposes the active tab's label in session state so
    # AppTest can drive tab switches (it has no tab-selection API of its own).
    on_change="rerun",
    key="input_tabs",
)
with upload_tab:
    uploaded_files = st.file_uploader(
        "Upload audio or video files",
        type=AUDIO_FORMATS + VIDEO_FORMATS,
        label_visibility="collapsed",
        accept_multiple_files=True,
    )
    for uploaded_file in uploaded_files:
        st.audio(uploaded_file)

with record_tab:
    recorded_audio = st.audio_input("Record audio", label_visibility="collapsed")
    if recorded_audio:
        st.audio(recorded_audio)

with youtube_tab:
    youtube_url = st.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...",
        label_visibility="collapsed",
    ).strip()
    youtube_audio: _RemoteAudio | None = None
    # Gate the fetch — not the text input — on tab visibility. st.tabs computes
    # hidden bodies by default, so an ungated fetch downloads while the user is on
    # another tab, and the Transcribe dispatch prioritizes uploads anyway. The
    # input stays outside the guard because Streamlit drops state for widgets it
    # doesn't render, which would clear the typed URL on every tab switch.
    if youtube_tab.open and youtube_url and YOUTUBE_URL_RE.match(youtube_url):
        try:
            data, filename = _fetch_youtube_audio(youtube_url)
            youtube_audio = _RemoteAudio(filename, data)
            st.audio(data)
        except yt_dlp.utils.DownloadError as e:
            st.error(f"Could not download from YouTube: {e}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            st.exception(e)

with url_tab:
    file_url = st.text_input(
        "Audio/video file URL",
        placeholder="Audio/video file URL",
        label_visibility="collapsed",
    ).strip()
    url_audio: _RemoteAudio | None = None
    # Fetch gated on tab visibility for the same reason as the YouTube tab above.
    if url_tab.open and file_url and URL_RE.match(file_url):
        if YOUTUBE_URL_RE.match(file_url):
            st.info("This looks like a YouTube URL — use the YouTube tab.")
        else:
            try:
                data, filename = _fetch_url_audio(file_url)
                url_audio = _RemoteAudio(filename, data)
                st.audio(data)
            except (URLError, RuntimeError) as e:
                st.error(f"Could not download from URL: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")
                st.exception(e)

language_label_col, language_col = st.columns([3, 1], vertical_alignment="center")
with language_label_col:
    st.markdown(
        "Primary language",
        help=(
            "The primary language spoken in an uploaded file. "
            "By default, the primary language will be detected automatically."
        ),
    )
with language_col:
    language = st.selectbox(
        "Primary language",
        LANGUAGE_CODES,
        format_func=_format_language,
        label_visibility="collapsed",
    )

translate = _labeled_toggle(
    "Translate to English",
    "Translates audio to English instead of transcribing in the source language.",
)
include_subtitles = _labeled_toggle(
    "Include subtitles",
    "Best for adding subtitles to a video. When enabled, the project will be "
    "initialized with subtitles which you can then alter in the editor.",
)
no_verbatim = _labeled_toggle(
    "No verbatim",
    "When enabled, the transcription will be cleaned up by removing "
    "filler words, false starts, and repetitions.",
)
with st.expander("Advanced options", icon=":material/tune:"):
    decode_independently = _labeled_toggle(
        "Decode segments independently",
        "When enabled, each 30-second window is transcribed without context "
        "from prior windows. More robust on noisy or music-heavy audio.",
    )

    _field_label(
        "Time range",
        'Comma-separated start,end pairs in seconds (e.g., "30,90" for a '
        'single clip, "0,60,120,180" for multiple clips). Leave blank to '
        "transcribe the full file.",
    )
    time_range_input = st.text_input(
        "Time range",
        placeholder="e.g., 30,90 (leave blank for full file)",
        label_visibility="collapsed",
    ).strip()
    time_range_error = _validate_time_range(time_range_input)
    clip_timestamps = time_range_input or "0"

    _field_label(
        "Keyterms",
        "Up to 50 keyterms to be boosted during transcription. "
        "Boosted terms are more likely to appear in the output.",
    )
    keyterms = st.multiselect(
        "Keyterms",
        options=[],
        accept_new_options=True,
        max_selections=50,
        placeholder="Add keyterms...",
        label_visibility="collapsed",
    )
    initial_prompt = ", ".join(keyterms) or None

audio_sources = (
    uploaded_files
    or ([recorded_audio] if recorded_audio else [])
    or ([youtube_audio] if youtube_audio else [])
    or ([url_audio] if url_audio else [])
)
# Render outside the Advanced options expander so a disabled Transcribe button
# always shows its reason, even when the expander holding the input is collapsed.
if time_range_error:
    st.error(time_range_error, icon=":material/error:")
_, action_col = st.columns([3, 1])
with action_col:
    transcribe_clicked = st.button(
        "Transcribe",
        icon=":material/graphic_eq:",
        type="primary",
        disabled=not audio_sources or bool(time_range_error),
        width="stretch",
    )

if transcribe_clicked and audio_sources and not time_range_error:
    _handle_transcription(
        audio_sources,
        **_transcription_kwargs(
            language=language,
            translate=translate,
            include_subtitles=include_subtitles,
            initial_prompt=initial_prompt,
            no_verbatim=no_verbatim,
            decode_independently=decode_independently,
            clip_timestamps=clip_timestamps,
        ),
    )

# Wrapped in a fragment so transcript edits/downloads rerun only this section
# instead of the whole script (which re-evaluates all four input tabs).
st.fragment(_display_transcription)()
