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
# Ceiling on bytes either remote fetch will pull into memory and cache. Governs
# both the URL and the YouTube path (hence the name — it was MAX_URL_DOWNLOAD_BYTES
# while only the URL path was capped). Distinct from server.maxUploadSize, which
# bounds a per-file browser PUT that never enters these caches.
MAX_DOWNLOAD_BYTES = 500 * 1024 * 1024
# Shared width of every right-hand control: the language selectbox and the
# Transcribe and Download buttons. It has to be an explicit number because
# st.selectbox has no width="content" (its default is "stretch", which fills a
# horizontal container); the two buttons take the same value so all three share
# a right edge. Originally reverse-engineered from st.columns([3, 1]) — the
# `centered` layout's main block has a 704px content box and st.columns puts a
# 32px gap between the two columns, so the right column was (704 - 32) / 4 =
# 168px — and kept at that value so the rendered layout is unchanged now that
# the columns are gone.
SELECT_WIDTH = 168
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


# cache_resource, not cache_data, and the deviation is deliberate: performance.md
# scopes cache_resource to unserializable objects, and bytes are serializable. But
# cache_data stores entries *pickled* and returns a fresh copy per call, so an
# active entry costs three resident buffers — the pickled entry, a per-rerun
# unpickled copy (these fetches re-invoke on every rerun while their tab is open),
# and the MediaFileManager buffer backing the st.audio preview. cache_resource
# hands back the same object and collapses the first two. Safe here only because
# the return is a tuple of immutables: a shared mutable would be a cross-session
# aliasing bug. Note _clear_caches must call st.cache_resource.clear() too.
@st.cache_resource(show_spinner="Downloading audio from YouTube...", max_entries=5, ttl="1h")
def _fetch_youtube_audio(url: str) -> tuple[bytes, str]:
    with tempfile.TemporaryDirectory() as tmpdir:
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": str(Path(tmpdir) / "%(title)s.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "restrictfilenames": True,
            "max_filesize": MAX_DOWNLOAD_BYTES,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            downloaded = Path(ydl.prepare_filename(info))
        # Both halves are needed. `max_filesize` aborts the download early, but
        # yt-dlp only consults it where a Content-Length is known (downloader/
        # http.py, external.py) — no fragmented HLS/DASH downloader reads it at
        # all, which is exactly the multi-hour livestream VOD this exists to
        # stop. The stat() gate catches what slipped through: the file is
        # already on disk, and what is being bounded is the slurp into one
        # bytes object and the cache entry holding it, not the disk write.
        if downloaded.stat().st_size > MAX_DOWNLOAD_BYTES:
            raise RuntimeError(f"YouTube audio exceeds {MAX_DOWNLOAD_BYTES // (1024 * 1024)} MB")
        return downloaded.read_bytes(), downloaded.name


@st.cache_resource(
    show_spinner="Downloading audio from URL...", max_entries=5, ttl="1h"
)  # See above.
def _fetch_url_audio(url: str) -> tuple[bytes, str]:
    with urlopen(url, timeout=60) as resp:
        data = resp.read(MAX_DOWNLOAD_BYTES + 1)
    if len(data) > MAX_DOWNLOAD_BYTES:
        raise RuntimeError(f"URL response exceeds {MAX_DOWNLOAD_BYTES // (1024 * 1024)} MB")
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
    # Bump the batch id alongside that publish so _display_transcription's widget
    # keys change with the batch. A keyed st.text_area restores its session-state
    # value and ignores the `value` argument, so reusing transcript_{i} across
    # batches renders the *previous* batch's text under the new filename — and the
    # Download button, whose payload is the text area's return value, serves it.
    st.session_state["batch_id"] = st.session_state.get("batch_id", 0) + 1
    total = len(uploaded_files)
    with st.status(f"Transcribing {total} file(s)...", expanded=True) as status:
        for i, uploaded_file in enumerate(uploaded_files, start=1):
            # Escape before interpolating anywhere Markdown renders. An st.status
            # label takes the Markdown label subset — which includes images, so a
            # filename carrying `![](https://host/x.png)` would fetch on *every*
            # file, not just a failure — and st.error below renders full Markdown.
            # Filenames are not trusted input: _fetch_url_audio percent-decodes
            # them off the URL path, so `%5B`/`%28` arrive as live syntax.
            name_md = _escape_markdown(uploaded_file.name)
            status.update(label=f"Transcribing {name_md} ({i}/{total})...")
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
                st.error(f"Transcription failed for {name_md}: {e}", icon=":material/error:")
            except Exception as e:
                st.error(f"Unexpected error for {name_md}: {e}", icon=":material/error:")
                st.exception(e)
        status.update(
            label=f"Transcribed {len(transcriptions)}/{total} file(s)",
            state="complete",
        )


def _control_row():
    """One labelled-control row: label left, control right, vertically centred.

    A horizontal container rather than st.columns([3, 1]) because the ratio was
    never load-bearing here — only the flush-right edge is — and columns stack
    vertically on a narrow viewport, which breaks the row into two lines.
    `horizontal_alignment="distribute"` is space-between with two children and
    plain left-alignment with one, so _field_label shares the same row metrics.
    """
    return st.container(
        horizontal=True,
        horizontal_alignment="distribute",
        vertical_alignment="center",
    )


def _labeled_toggle(label: str, help_text: str) -> bool:
    with _control_row():
        st.markdown(label, help=help_text)
        return st.toggle(label, value=False, label_visibility="collapsed")


def _field_label(label: str, help_text: str) -> None:
    with _control_row():
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
    # Namespaces the widget keys below by batch; see _handle_transcription.
    batch = st.session_state.get("batch_id", 0)
    for i, data in enumerate(transcriptions):
        include_subtitles = data["include_subtitles"]
        if include_subtitles:
            initial = _format_srt(data["result"])
        else:
            initial = data["result"]["text"].strip()
        # One bordered box per result. Sections stack flat otherwise, so in a
        # multi-file batch one file's Download button abuts the next file's
        # heading with nothing marking the seam. No-op visually for a single file.
        with st.container(border=True):
            st.subheader(_escape_markdown(data["filename"]))
            transcript = st.text_area(
                "Transcript",
                initial,
                height=300,
                label_visibility="collapsed",
                key=f"transcript_b{batch}_{i}",
            )
            ext, mime = (
                ("srt", "application/x-subrip") if include_subtitles else ("txt", "text/plain")
            )
            with st.container(horizontal=True, horizontal_alignment="right"):
                st.download_button(
                    "Download",
                    transcript,
                    f"{data['file_stem']}.{ext}",
                    mime,
                    icon=":material/download:",
                    key=f"download_{ext}_b{batch}_{i}",
                    # The second sentence is not padding. st.download_button
                    # materializes non-callable `data` at *render* time and hands
                    # the frontend a pre-baked URL, while st.text_area commits
                    # only on blur or Ctrl/Cmd+Enter — so a click with an
                    # uncommitted edit serves the previous text. Verified in real
                    # Chrome: first click got the pre-edit string, second got the
                    # edit. Not fixable in-app (st.form rejects download buttons,
                    # and a deferred callable runs before the pending update
                    # lands), so the tooltip is the mitigation.
                    help=(
                        "Downloads as .srt when subtitles are enabled, .txt otherwise. "
                        "Commit an edit first — click outside the box or press "
                        "Ctrl/Cmd+Enter — or the download will miss it."
                    ),
                    # Nothing here depends on a post-download rerun — the payload
                    # is the text area's already-committed return value.
                    on_click="ignore",
                    width=SELECT_WIDTH,
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
    # Reserve the preview's slot here and run the fetch further down, after the
    # controls have rendered. Streamlit paints top to bottom, so a fetch at this
    # position leaves the language selector, every toggle, and Advanced options
    # greyed out as stale for the length of the download. Writing back into this
    # container keeps the preview — and the cache's own download spinner, the
    # only progress signal on this path — inside the tab where they belong.
    youtube_slot = st.container()

with url_tab:
    file_url = st.text_input(
        "Audio/video file URL",
        placeholder="Audio/video file URL",
        label_visibility="collapsed",
    ).strip()
    url_slot = st.container()  # Deferred like the YouTube preview above.

with _control_row():
    st.markdown(
        "Primary language",
        help=(
            "The primary language spoken in an uploaded file. "
            "By default, the primary language will be detected automatically."
        ),
    )
    language = st.selectbox(
        "Primary language",
        LANGUAGE_CODES,
        width=SELECT_WIDTH,
        format_func=_format_language,
        label_visibility="collapsed",
    )

translate = _labeled_toggle(
    "Translate to English",
    "Translates audio to English instead of transcribing in the source language.",
)
include_subtitles = _labeled_toggle(
    "Include subtitles",
    "Best for adding subtitles to a video. Shows the transcript as editable, "
    "timestamped SRT cues, and switches the Download button from .txt to .srt.",
)
no_verbatim = _labeled_toggle(
    "No verbatim",
    "Skips silent stretches where Whisper appears to be hallucinating text, "
    "such as over music or applause after speech ends. Does not remove "
    "filler words or repetitions.",
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

# Remote fetches run here, below every control, and write back into the slots
# reserved inside their tabs. Each is gated on tab visibility — not on the text
# input above, which must always render: st.tabs computes hidden bodies by
# default, so an ungated fetch downloads while the user is on another tab, and
# Streamlit drops state for widgets it doesn't render, which would clear the
# typed URL on every tab switch.
youtube_audio: _RemoteAudio | None = None
if youtube_tab.open and youtube_url and YOUTUBE_URL_RE.match(youtube_url):
    with youtube_slot:
        try:
            data, filename = _fetch_youtube_audio(youtube_url)
            youtube_audio = _RemoteAudio(filename, data)
            st.audio(data)
        except yt_dlp.utils.DownloadError as e:
            st.error(f"Could not download from YouTube: {e}", icon=":material/error:")
        except Exception as e:
            st.error(f"Unexpected error: {e}", icon=":material/error:")
            st.exception(e)

url_audio: _RemoteAudio | None = None
if url_tab.open and file_url and URL_RE.match(file_url):
    with url_slot:
        if YOUTUBE_URL_RE.match(file_url):
            st.info("This looks like a YouTube URL — use the YouTube tab.")
        else:
            try:
                data, filename = _fetch_url_audio(file_url)
                url_audio = _RemoteAudio(filename, data)
                st.audio(data)
            except (URLError, RuntimeError) as e:
                st.error(f"Could not download from URL: {e}", icon=":material/error:")
            except Exception as e:
                st.error(f"Unexpected error: {e}", icon=":material/error:")
                st.exception(e)

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
with st.container(horizontal=True, horizontal_alignment="right"):
    transcribe_clicked = st.button(
        "Transcribe",
        icon=":material/graphic_eq:",
        type="primary",
        disabled=not audio_sources or bool(time_range_error),
        width=SELECT_WIDTH,
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
