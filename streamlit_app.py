import math
import re
import tempfile
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import unquote, urlparse
from urllib.request import urlopen

# mlx/core is a compiled extension (core.cpython-313-darwin.so) that ships no
# .pyi alongside its py.typed marker, so ty cannot introspect it and reports
# unresolved-import. The suppression is scoped to this line rather than turned
# into a [tool.ty] rule override, so a genuinely missing import still fails.
# mlx_whisper imports it the same way; ty only checks first-party code, which is
# why the same statement inside site-packages raises nothing.
import mlx.core as mx  # ty: ignore[unresolved-import]
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
# st.audio's `format` is not a hint. It is passed verbatim through
# _marshall_av_media into MediaFileManager.add() with no sniffing anywhere, and
# becomes both the Content-Type header the media route serves and the extension
# in the /media/<hash>.<ext> URL. Its default is "audio/wav", so without this map
# every preview — an .mp3 upload, a YouTube Opus stream — is advertised as WAV.
# Chrome and Firefox sniff the container and play it anyway; browsers that trust
# the declared type can refuse. Deliberately not mimetypes.guess_type: it maps
# .m4a to the non-standard audio/mp4a-latm that browsers do not recognize, reads
# a table that varies by platform, and returns None for the extensionless name
# _fetch_url_audio falls back to. Covers more than AUDIO_FORMATS/VIDEO_FORMATS
# because the YouTube and URL paths never consult those tuples.
MEDIA_MIME_TYPES = {
    "mp3": "audio/mpeg",
    "m4a": "audio/mp4",
    "wav": "audio/wav",
    "opus": "audio/ogg",
    "oga": "audio/ogg",
    "ogg": "audio/ogg",
    "flac": "audio/flac",
    "aac": "audio/aac",
    "mp4": "video/mp4",
    "mov": "video/quicktime",
    "webm": "video/webm",
    "mkv": "video/x-matroska",
}
# A container extension names the container, not its contents. yt-dlp's
# `bestaudio` yields audio-only .webm (Opus) and .m4a/.mp4 (AAC), and declaring
# video/webm on an <audio> element is the same mis-declaration MEDIA_MIME_TYPES
# exists to prevent — so a caller that *knows* there is no video track says so
# and gets the audio/* sibling. Only the two containers bestaudio actually
# produces are listed; .mkv/.mov audio-only does not occur on these paths.
AUDIO_ONLY_MIME_TYPES = {
    "video/webm": "audio/webm",
    "video/mp4": "audio/mp4",
}
# Fallback for an unrecognized extension, including _fetch_url_audio's extensionless
# "download". Must stay non-empty: the media route does `media_type=mimetype or
# "text/plain"`, so an empty string would serve audio as text.
DEFAULT_MEDIA_MIME = "audio/wav"
ERROR_ICON = ":material/error:"
# The transcript format choice, as the segmented control renders it. This governs
# two things at once -- what the results text area shows (plain text vs timestamped
# SRT cues) and which extension the Download button serves (.txt vs .srt) -- which
# is why it is a two-option format picker rather than the "Include subtitles" toggle
# it used to be: a boolean names one of the two states and leaves the other implied,
# so the .srt consequence was reachable only through the help tooltip. Order is
# display order, and FORMAT_PLAIN_TEXT is the default.
FORMAT_PLAIN_TEXT = "Plain text"
FORMAT_SUBTITLES = "Subtitles"
TRANSCRIPT_FORMATS = (FORMAT_PLAIN_TEXT, FORMAT_SUBTITLES)
LANGUAGE_CODES: list[str | None] = [None] + sorted(LANGUAGES, key=lambda c: LANGUAGES[c])
YOUTUBE_URL_RE = re.compile(r"^https?://(www\.|m\.)?(youtube\.com/|youtu\.be/)", re.IGNORECASE)
URL_RE = re.compile(r"^https?://", re.IGNORECASE)
# Ceiling on bytes either remote fetch will pull into memory and cache. Governs
# both the URL and the YouTube path (hence the name — it was MAX_URL_DOWNLOAD_BYTES
# while only the URL path was capped). Distinct from server.maxUploadSize, which
# bounds a per-file browser PUT that never enters these caches.
MAX_DOWNLOAD_BYTES = 500 * 1024 * 1024
# Shared width of every right-hand control: the language selectbox, the Time range
# input, and the Transcribe and Download buttons. It has to be an explicit number
# because st.selectbox has no width="content" (its default is "stretch", which
# fills a horizontal container); the others take the same value so all four match.
#
# It is a shared *width*, not a shared right *edge*. Re-measured at a 1200px
# viewport: the selectbox and Download both land at x=768, right=936, while
# Transcribe lands at x=784, right=952. The split is not arbitrary — it is exactly
# "inside a bordered card" versus "not". A st.container(border=True) has computed
# padding: 15px plus a 1px border, insetting its content box from 704px to 672px,
# and the selectbox (controls card) and Download (results card) each sit in one.
# Transcribe sits outside both, so it stays flush with the 704px content edge —
# which is also where the controls card's own right border falls, so the button
# lines up with the card above it rather than with the selectbox inside it. That is
# the correct look and is not removable without custom CSS, which this app does not
# use: do not "fix" it by dropping border=True or hand-padding.
#
# Originally reverse-engineered from st.columns([3, 1]) — the `centered` layout's
# main block has a 704px content box and st.columns puts a 32px gap between the two
# columns, so the right column was (704 - 32) / 4 = 168px — and kept at that value
# so the rendered layout is unchanged now that the columns are gone.
SELECT_WIDTH = 168
# Height of the st.skeleton standing in for a remote tab's st.audio preview while
# the fetch runs, measured off the rendered player rather than guessed. Matching it
# is the whole point of the skeleton: the cache's show_spinner already says a
# download is happening, so what the placeholder adds is holding the preview's space
# so the controls below do not jump when the player replaces it. A wrong value trades
# one jump for a smaller one. Re-measure if the preview ever stops being a bare
# st.audio: getComputedStyle on [data-testid="stAudio"] reports the rendered height.
AUDIO_PREVIEW_HEIGHT = 40
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
#
# max_entries=2, not 5, and it is the *only* lever on the memory ceiling. These two
# caches hold raw audio, so the worst case is entries x MAX_DOWNLOAD_BYTES x 2
# caches -- 5 GB at the old value, 2 GB at this one. ttl looks like it should help
# and does not: Streamlit's ttl_cache expires lazily, so an expired entry reads as
# absent but is only actually dropped by a write, an expire() call, or a
# length/size query. It bounds staleness, not resident memory, and an idle server
# can hold expired payloads indefinitely. max_entries is what bounds how many can
# coexist. The cost of the smaller number is a re-download when a user cycles
# through more than two remote sources inside the ttl window; 2 still covers going
# back to the previous one, which is the common case.
#
# show_spinner=False, and this reverses an earlier decision on purpose. These two
# fetches used to pass a "Downloading audio from..." message on the grounds that,
# unlike _transcribe, nothing wraps them in an st.status and the cache spinner was
# the only progress signal. The st.skeleton in each tab's reserved slot is now that
# signal, and the spinner actively fights it: the spinner is an *extra* element with
# no reserved space, so during a download it grows the slot by its own height and
# pushes the settings card and Transcribe button down ~40px, which then snap back
# when it clears. Measured against a deliberately slow local server, with both, the
# card rendered stale-faded with the spinner text overlapping it; with the skeleton
# alone the page does not move at all, because a skeleton sized to the preview it
# replaces is layout-neutral by construction. Reserving space is the whole point of
# the pattern, and the spinner is the thing that breaks the reservation.
@st.cache_resource(show_spinner=False, max_entries=2, ttl="1h")
def _fetch_youtube_audio(url: str) -> tuple[bytes, str, str]:
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
        # The extension alone cannot tell an audio-only WebM from a video one, but
        # `info` can: with format="bestaudio/best" yt-dlp reports the selected
        # stream's codecs, so vcodec == "none" means there is no video track. A
        # missing key falls back to the container's own type, which is what an
        # unknown/merged selection deserves.
        audio_only = info.get("vcodec") == "none"
        return (
            downloaded.read_bytes(),
            downloaded.name,
            _media_mime(downloaded.name, audio_only=audio_only),
        )


@st.cache_resource(show_spinner=False, max_entries=2, ttl="1h")  # See above.
def _fetch_url_audio(url: str) -> tuple[bytes, str, str]:
    with urlopen(url, timeout=60) as resp:
        data = resp.read(MAX_DOWNLOAD_BYTES + 1)
        declared = resp.headers.get("Content-Type", "")
    if len(data) > MAX_DOWNLOAD_BYTES:
        raise RuntimeError(f"URL response exceeds {MAX_DOWNLOAD_BYTES // (1024 * 1024)} MB")
    filename = unquote(Path(urlparse(url).path).name) or "download"
    return data, filename, _url_mime(declared, filename)


def _url_mime(declared: str, filename: str) -> str:
    """Prefer the server's own Content-Type, falling back to the extension map.

    The header is authoritative and covers what an extension cannot see: a
    content-negotiated or query-driven URL, a redirect target with no extension,
    and the extensionless "download" fallback — all of which the map can only
    answer with DEFAULT_MEDIA_MIME, i.e. the very audio/wav mis-declaration
    _media_mime exists to avoid. Only audio/* and video/* are trusted, though: a
    server that answers text/html (an error page) or application/octet-stream
    must not get to set the declared type of an <audio> element.
    """
    mime = declared.split(";")[0].strip().lower()
    return mime if mime.startswith(("audio/", "video/")) else _media_mime(filename)


def _media_mime(filename: str, *, audio_only: bool = False) -> str:
    """Content-Type for an st.audio preview, derived from the filename's extension.

    See MEDIA_MIME_TYPES for why st.audio's "audio/wav" default is not good enough
    and why this is a hand-written map rather than mimetypes.guess_type. Pass
    audio_only=True when the caller knows the container holds no video track — see
    AUDIO_ONLY_MIME_TYPES.
    """
    mime = MEDIA_MIME_TYPES.get(Path(filename).suffix.lstrip(".").lower(), DEFAULT_MEDIA_MIME)
    return AUDIO_ONLY_MIME_TYPES.get(mime, mime) if audio_only else mime


def _plural(count: int, noun: str) -> str:
    """`1 file` / `2 files`, so the status label never reads `1 file(s)`.

    Every batch renders this string, and a single file is the common case — which
    is exactly the one the parenthesised form gets wrong.
    """
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


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


_MARKDOWN_ESCAPE_RE = re.compile(r"([\\`*_~\[\]:$&])")


def _escape_markdown(text: str) -> str:
    """Render `text` literally at any of this app's Markdown sinks.

    Filenames are untrusted: `_fetch_url_audio` percent-decodes them off the URL
    path, so `%5B` / `%28` arrive as live syntax and `%0A` as a real newline.

    Two mechanisms, because Markdown has two layers:

    *Inline* constructs are backslash-escaped. A filename containing *, _,
    backticks, brackets, or : (emoji/Material-icon directives) — common in YouTube
    titles and underscored names — would otherwise mis-render. `&` is in the class
    because micromark's characterReference is a parse-time construct: without it
    `clip&#58;streamlit&#58;.mp3` decodes to a live `:streamlit:` that the
    frontend's post-parse pass swaps for the logo image, and `Rock &amp; Roll.mp3`
    displays as `Rock & Roll.mp3`. `&` is ASCII punctuation, so `\\&` is a valid
    CommonMark characterEscape.

    *Block* constructs are defused by collapsing whitespace, not by escaping:
    headings, blockquotes, lists, thematic breaks and GFM tables all need a line
    start, so removing newlines removes every one of them at once — a smaller
    change than adding #, >, -, +, | and the ordered-list forms to the class.
    This has to happen here rather than at one call site, because *two* sinks
    render without the frontend's isLabel flag (which is what would otherwise
    auto-escape block markers and strip block elements):

    - `st.error`'s body — AlertElement passes isLabel for the alert title only.
    - `st.subheader` — the heading component does `[first, ...rest] = body.split("\\n")`
      and renders `rest` through a bare `StreamlitMarkdown` with no isLabel and no
      disallowedElements, so everything after the first newline is full Markdown.
    """
    return _MARKDOWN_ESCAPE_RE.sub(r"\\\1", " ".join(text.split()))


def _error(message: str) -> None:
    """Render an error alert with the app's icon, as literal text.

    The *whole* message is escaped, so callers pass raw text — filenames, URLs,
    exception strings — and never escape at the interpolation site. Escaping the
    fixed literals along with them is harmless: `:` becomes `\\:`, which renders
    as a colon.

    Both halves matter. st.error's body is one of the two sinks Streamlit renders
    without the frontend's isLabel flag (see _escape_markdown), and the exception
    text reaching it is untrusted: yt-dlp's UnsupportedError is literally
    f"Unsupported URL: {url}" and YOUTUBE_URL_RE is prefix-anchored, so a pasted
    `https://youtu.be/![](https://host/p.png)` arrives verbatim and would fire an
    outbound request. _validate_time_range likewise echoes the raw input back.
    Routing every call through here also makes the icon structural instead of a
    literal repeated at seven call sites, where a new one renders a bare box.
    """
    st.error(_escape_markdown(message), icon=ERROR_ICON)


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
        # float() happily accepts "nan", "inf" and any overflowing literal such
        # as 1e400, and none of them trip the checks below: nan < 0 is False and
        # every nan comparison is False, so a malformed range used to validate
        # clean, enable Transcribe, and fail inside mlx_whisper instead.
        if not math.isfinite(value):
            return f"Invalid time range: {token!r} is not a finite number."
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


def _split_clips(clip_timestamps: str) -> list[str]:
    """Split a `clip_timestamps` string into one `start,end` string per clip.

    `"0,60,120,180"` → `["0,60", "120,180"]`. A trailing unpaired start keeps its
    open end (`"0,60,120"` → `["0,60", "120"]`), which mlx_whisper reads as
    running to the end of the file.

    This exists because **mlx_whisper does not honour more than one clip.** Its
    decode loop is `for seek_clip_start, seek_clip_end in seek_clips:` wrapping
    `while seek < seek_clip_end:` — `seek_clip_start` is bound and never read,
    and there is no `seek = seek_clip_start`, so `seek` carries over from the
    previous clip and each later range simply continues from wherever the last
    one stopped. Given `0,60,120,180` it decodes 0–180 straight through,
    including the 60–120 the user excluded. (`clip_idx` at `transcribe.py:247` is
    initialised and never incremented — vestigial from the same loop upstream.)
    A single pair is unaffected, because `seek` is initialised to
    `seek_clips[0][0]`, which is why `30,90` was always correct and only the
    multi-clip tail was broken — and why the bug outlived a green test named
    `multi_clip` that asserted only that the string was forwarded.
    """
    values = [t.strip() for t in clip_timestamps.split(",") if t.strip()]
    return [",".join(values[i : i + 2]) for i in range(0, len(values), 2)] or ["0"]


def _merge_transcriptions(results: list[dict]) -> dict:
    """Concatenate per-clip results into the shape a single call returns.

    Segment timestamps are already absolute — mlx_whisper derives them from
    `seek`, which starts at the clip's own start frame — so segments concatenate
    without adjustment and `_format_srt` needs no change. `language` comes from
    the first clip: with `language=None` each clip auto-detects independently and
    could in principle disagree, and one file gets one label.
    """
    return {
        "text": "".join(r["text"] for r in results),
        "segments": [segment for r in results for segment in r["segments"]],
        "language": results[0]["language"],
    }


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
    # One mlx_whisper call per clip, because it decodes straight through the gaps
    # between them — see _split_clips. A single clip (the "0" default, and every
    # plain `start,end` pair) takes exactly one call and its result is returned
    # untouched, so the common path is byte-identical to before. The accepted cost
    # on the multi-clip path is that each call re-runs ffmpeg over the whole file;
    # clips are typically two or three, and decoding the ranges the user actually
    # asked for is worth more than one shared decode of ranges they excluded.
    clips = _split_clips(clip_timestamps)
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        results = [
            mlx_whisper.transcribe(
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
                clip_timestamps=clip,
            )
            for clip in clips
        ]
    result = results[0] if len(results) == 1 else _merge_transcriptions(results)
    # Only the merged text is checked, so one silent clip inside a multi-clip
    # range does not fail the whole transcription.
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
    # Per-file failures are collected here and rendered *after* the status block,
    # not inside it. st.status collapses on its first update(label=...): update()
    # clears the proto's `expanded` field unless it is passed again (see
    # mutable_status_container.py), and the frontend's label-change branch then
    # resets the open state to that now-false backend value. So an alert written
    # into the status body during the loop landed inside a collapsed container
    # carrying a *success* indicator: a failed file rendered a green check,
    # "Transcribed 0/1 file", and no visible explanation anywhere on the page. The
    # status stays expandable, so the text was reachable — but nothing on screen
    # suggested a failure had happened or that anything was hidden, which is the
    # part that made it a defect rather than a disclosure.
    failures: list[tuple[str, Exception | None]] = []
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
    try:
        with st.status(f"Transcribing {_plural(total, 'file')}...", expanded=True) as status:
            for i, uploaded_file in enumerate(uploaded_files, start=1):
                # Escape before interpolating anywhere Markdown renders. An st.status
                # label takes the Markdown label subset — which includes images, so a
                # filename carrying `![](https://host/x.png)` would fetch on *every*
                # file, not just a failure — and st.error below renders full Markdown.
                # See _escape_markdown for what it does and does not cover.
                name_md = _escape_markdown(uploaded_file.name)
                status.update(label=f"Transcribing {name_md} ({i}/{total})...")
                name = Path(uploaded_file.name)
                # Rewind before reading. UploadedFile subclasses io.BytesIO, and the
                # deserialized widget value is cached in session state
                # (WStates.__getitem__ stores Value(deserialized)), so the *same*
                # object — and the same cursor — survives every rerun. read() leaves
                # it at EOF, so a second Transcribe on an unchanged recording would
                # hand _transcribe b"". This used to be masked by st.audio's own
                # data.seek(0) inside _marshall_av_media, which ran once per rerun for
                # each preview; that is a side effect of a display call, not a
                # contract, and the Record tab deliberately renders no preview.
                # _RemoteAudio has no cursor and needs no rewind.
                if isinstance(uploaded_file, UploadedFile):
                    uploaded_file.seek(0)
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
                    failures.append((f"Transcription failed for {uploaded_file.name}: {e}", None))
                except Exception as e:
                    failures.append((f"Unexpected error for {uploaded_file.name}: {e}", e))
            status.update(
                label=f"Transcribed {len(transcriptions)}/{_plural(total, 'file')}",
                state="error" if failures else "complete",
            )
    finally:
        # Reclaim MLX's allocator cache now the batch is done. mlx keeps freed
        # device buffers on a free list instead of returning them, so a finished
        # transcription leaves them resident for the life of the server process.
        # Measured here against whisper-large-v3-turbo with word_timestamps=True:
        # 894 MB held after an 8-second file and 1.25 GB after two minutes, all of
        # it reclaimed, for 4.2 ms. The next run costs 0.76 s against 0.74 s with
        # the cache warm — it re-allocates rather than reloading, because the
        # *model* lives in `active` memory (a flat 1543 MB across every
        # measurement) and clear_cache() does not touch it.
        #
        # Verify with mx.get_cache_memory(), NOT with `top`. RSS moved 2.6 MB while
        # 894 MB was reclaimed, because these are device-mapped MTLBuffers — anyone
        # checking the process's resident size concludes this did nothing.
        #
        # mx.set_cache_limit() is the obvious cheaper alternative and is worse: it
        # caps the free list *during* decode, throttling the run it is meant to
        # help. A batch boundary costs nothing in the middle of the work.
        #
        # `finally`, not a plain call after the block: a tab switch or any widget
        # raises RerunException mid-batch, and an interrupted run would otherwise
        # hold its buffers until whenever the next transcription finishes.
        mx.clear_cache()
    # Replayed outside the status block, so the alerts land in the page body where
    # they stay visible. A BaseException (RerunException) unwinds past this without
    # replaying anything, which is correct: that run's output is discarded whole.
    for message, unexpected in failures:
        _error(message)
        if unexpected is not None:
            st.exception(unexpected)


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
# Orientation for a first-time visitor, who otherwise meets a bare title and a
# dropzone. st.caption rather than st.info: design.md scopes the callout styles to
# instructions and problems, and this is neither. Deliberately says "transcribed"
# rather than a flat "nothing leaves your Mac" — the YouTube and URL tabs do reach
# the network, so the stronger claim would be false in two of the four modes.
st.caption("Audio and video, transcribed and translated locally on your Mac.")

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
        st.audio(uploaded_file, format=_media_mime(uploaded_file.name))

with record_tab:
    # No st.audio preview here, unlike the other three tabs. st.audio_input is not
    # a bare capture control — it renders its own WaveSurfer player (interactive
    # waveform, timecode, Play/Pause as soon as a recording exists, and a
    # "Clear recording" action), so an st.audio call would stack a second,
    # visually different player on the same bytes.
    recorded_audio = st.audio_input("Record audio", label_visibility="collapsed")

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
        # A worked example, not a restatement of the (collapsed) label — which is
        # what this placeholder used to be, and the one remote tab that did not
        # show the user what a valid value looks like.
        placeholder="https://example.com/audio.mp3",
        label_visibility="collapsed",
    ).strip()
    url_slot = st.container()  # Deferred like the YouTube preview above.

# One bordered card around every control. They have been described as "grouped by
# intent (input → output → advanced)" since they were written, but until now that
# grouping was pure source ordering: four identical rows rendered flat between the
# dropzone and the expander, with nothing marking where input ends and output
# begins. The card gives the group an edge, and reuses the
# st.container(border=True) that already wraps each result in
# _display_transcription, so settings and results read as the same kind of surface.
#
# Advanced options is nested *inside* the card rather than left as its own
# top-level block, and both halves of that are deliberate. Semantically it is the
# third member of this group, not a peer of it. Visually, leaving it outside put
# two identically-bordered boxes a few px apart and the card read as though it had
# been cut in half; inset by the card's padding the expander instead reads as
# subordinate to it. Verified by rendering both.
with st.container(border=True):
    with _control_row():
        st.markdown(
            "Primary language",
            help=(
                # Not "an uploaded file": only one of the four input modes is an
                # upload, and this selector governs all of them.
                "The primary language spoken in the audio. "
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

    # The intent seam: language describes the input, the three toggles below shape
    # the output. Uniform row spacing renders all four as one undifferentiated run.
    st.space("small")

    translate = _labeled_toggle(
        "Translate to English",
        "Translates audio to English instead of transcribing in the source language.",
    )
    with _control_row():
        st.markdown(
            "Transcript format",
            help="Plain text, or timestamped SRT subtitle cues — best for adding "
            "subtitles to a video. The transcript stays editable either way, and "
            "this also switches the Download button between .txt and .srt.",
        )
        # required=True on top of default= is what makes the return a `str` rather
        # than `str | None`: without it, clicking the selected segment deselects it
        # and a single-select segmented control returns None, which would silently
        # read as "plain text" here and give the control a third, invisible state.
        transcript_format = st.segmented_control(
            "Transcript format",
            TRANSCRIPT_FORMATS,
            default=FORMAT_PLAIN_TEXT,
            required=True,
            label_visibility="collapsed",
        )
    include_subtitles = transcript_format == FORMAT_SUBTITLES
    no_verbatim = _labeled_toggle(
        "No verbatim",
        "Skips silent stretches where Whisper appears to be hallucinating text, "
        "such as over music or applause after speech ends. Does not remove "
        "filler words or repetitions.",
    )
    # The second seam, matching the one above: input | output | advanced. With only
    # the first one the card reads as two tiers rather than three.
    st.space("small")

    with st.expander("Advanced options", icon=":material/tune:"):
        decode_independently = _labeled_toggle(
            "Decode segments independently",
            "When enabled, each 30-second window is transcribed without context "
            "from prior windows. More robust on noisy or music-heavy audio.",
        )

        # Inline at SELECT_WIDTH rather than stacked under a _field_label, so this
        # row matches the toggle above it and the four rows in the card. The
        # placeholder sheds "(leave blank for full file)" to fit — that sentence is
        # not lost, the help tooltip already carries it. Keyterms below stays
        # stacked because its chips genuinely need the full width.
        with _control_row():
            st.markdown(
                "Time range",
                help='Comma-separated start,end pairs in seconds (e.g., "30,90" for a '
                'single clip, "0,60,120,180" for multiple clips). Leave blank to '
                "transcribe the full file.",
            )
            time_range_input = st.text_input(
                "Time range",
                placeholder="e.g., 30,90",
                width=SELECT_WIDTH,
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
    # `with youtube_slot, st.skeleton(...)`, not `youtube_slot.skeleton()`. The
    # skeleton's context-manager form does not redirect bare st.* calls into
    # itself -- they land in the *parent* container -- so entering youtube_slot
    # first is what keeps the cache's download spinner, the st.audio preview and
    # the two error alerts inside the tab. Calling youtube_slot.skeleton() alone
    # would put the skeleton in the tab and strand everything else below the
    # controls, which is the exact layout bug reserving the slot exists to avoid.
    with youtube_slot, st.skeleton(height=AUDIO_PREVIEW_HEIGHT):
        try:
            data, filename, mime = _fetch_youtube_audio(youtube_url)
            youtube_audio = _RemoteAudio(filename, data)
            st.audio(data, format=mime)
        # RuntimeError alongside DownloadError: the 500 MB stat gate inside
        # _fetch_youtube_audio raises it, and without it here that already-tested
        # guard surfaced as "Unexpected error" with a traceback attached. The URL
        # path below already pairs its own RuntimeError with URLError this way.
        except (yt_dlp.utils.DownloadError, RuntimeError) as e:
            _error(f"Could not download from YouTube: {e}")
        except Exception as e:
            _error(f"Unexpected error: {e}")
            st.exception(e)

url_audio: _RemoteAudio | None = None
if url_tab.open and file_url and URL_RE.match(file_url):
    with url_slot, st.skeleton(height=AUDIO_PREVIEW_HEIGHT):  # Nested as above.
        if YOUTUBE_URL_RE.match(file_url):
            # Icon matches the YouTube tab's own, so the callout points at its
            # destination. st.info has no default icon, same as st.error — see
            # _error, which exists to keep that from being forgotten.
            st.info(
                "This looks like a YouTube URL — use the YouTube tab.",
                icon=":material/smart_display:",
            )
        else:
            try:
                data, filename, mime = _fetch_url_audio(file_url)
                url_audio = _RemoteAudio(filename, data)
                st.audio(data, format=mime)
            except (URLError, RuntimeError) as e:
                _error(f"Could not download from URL: {e}")
            except Exception as e:
                _error(f"Unexpected error: {e}")
                st.exception(e)

# The tab the user is looking at wins, and that is not what a flat priority chain
# does. Upload and Record declare their widgets in ungated tab bodies, so their
# values are sticky across tab switches, while youtube_audio/url_audio exist only
# while their own tab is open. A plain `uploaded_files or ... or url_audio` chain
# therefore let an earlier upload outrank the URL whose preview was on screen:
# the fetch ran, the player rendered, the button enabled — and Transcribe
# silently transcribed the upload. Three things hid it. Both previews render at
# once, so two plausible sources are visible; the results subheader is the only
# signal of which one ran and it arrives after a full model pass; and
# _transcribe's cache makes the second click return instantly, which reads as
# "the URL happened to produce the same text".
#
# The old order is kept as the *fallback*, for when the open tab has no source of
# its own — an empty Record tab with an earlier upload still loaded — so the
# button never goes dead while a usable source exists. That leaves a deliberate
# residue: in exactly that case Transcribe still runs the upload.
tab_sources: tuple[tuple[Any, Sequence[UploadedFile | _RemoteAudio]], ...] = (
    (upload_tab, uploaded_files),
    (record_tab, [recorded_audio] if recorded_audio else []),
    (youtube_tab, [youtube_audio] if youtube_audio else []),
    (url_tab, [url_audio] if url_audio else []),
)
audio_sources = next(
    (sources for tab, sources in tab_sources if tab.open and sources),
    next((sources for _, sources in tab_sources if sources), []),
)
# Render outside the Advanced options expander so a disabled Transcribe button
# always shows its reason, even when the expander holding the input is collapsed.
if time_range_error:
    _error(time_range_error)
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
