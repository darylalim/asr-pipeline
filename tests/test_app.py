from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
import streamlit as st
from streamlit.proto.Common_pb2 import FileURLs
from streamlit.runtime.memory_media_file_storage import get_extension_for_mimetype
from streamlit.runtime.uploaded_file_manager import UploadedFile, UploadedFileRec
from streamlit.testing.v1 import AppTest

from streamlit_app import (
    ASR_MODEL_REPO,
    AUDIO_FORMATS,
    DEFAULT_MEDIA_MIME,
    ERROR_ICON,
    FORMAT_PLAIN_TEXT,
    FORMAT_SUBTITLES,
    MAX_DOWNLOAD_BYTES,
    PAGE_CONFIG,
    SELECT_WIDTH,
    TRANSCRIPT_FORMATS,
    VIDEO_FORMATS,
    _display_transcription,
    _error,
    _escape_markdown,
    _fetch_url_audio,
    _fetch_youtube_audio,
    _format_language,
    _format_srt,
    _format_timestamp,
    _handle_transcription,
    _media_mime,
    _plural,
    _RemoteAudio,
    _split_clips,
    _transcribe,
    _transcription_kwargs,
    _validate_time_range,
)

MOCK_WHISPER_RESULT = {
    "text": "Hello world",
    "segments": [
        {
            "id": 0,
            "seek": 0,
            "start": 0.0,
            "end": 2.5,
            "text": " Hello world",
            "tokens": [50364, 2425, 1002, 50414],
            "temperature": 0.0,
            "avg_logprob": -0.25,
            "compression_ratio": 1.2,
            "no_speech_prob": 0.01,
            "words": [
                {"word": " Hello", "start": 0.0, "end": 1.0, "probability": 0.98},
                {"word": " world", "start": 1.0, "end": 2.5, "probability": 0.95},
            ],
        }
    ],
    "language": "en",
}

SRT_HELLO = "1\n00:00:00,000 --> 00:00:02,500\nHello world\n"

# Spelled out rather than imported from streamlit_app — this *is* the assertion,
# so reading the app's own value would make the pin tautological. The second
# sentence is load-bearing: st.download_button bakes its payload at render time,
# so an uncommitted text-area edit is silently dropped (verified in Chrome).
DOWNLOAD_HELP = (
    "Downloads as .srt when subtitles are enabled, .txt otherwise. "
    "Commit an edit first — click outside the box or press "
    "Ctrl/Cmd+Enter — or the download will miss it."
)


# --- Helpers ---


def _make_file(name="interview.mp3", data=b"fake audio bytes"):
    f = MagicMock()
    f.name = name
    f.read.return_value = data
    return f


def _stub_urlopen(mock_urlopen, data, content_type=None):
    response = MagicMock()
    response.read.return_value = data
    # A real dict, not a MagicMock: _fetch_url_audio does headers.get(...) and then
    # string-tests the result, and every string method on a MagicMock returns a
    # truthy MagicMock -- so a mocked header would silently take the "trust the
    # server" branch and hand st.audio a MagicMock as its format.
    response.headers = {} if content_type is None else {"Content-Type": content_type}
    mock_urlopen.return_value.__enter__.return_value = response
    return response


def _ydl_info(title="Test", **extra):
    return {"title": title, **extra}


def _stub_ytdlp(mock_yt_dlp, path, title="Test", **info):
    return _stub_ytdlp_class(mock_yt_dlp.YoutubeDL, path, title, **info)


def _stub_ytdlp_class(mock_ydl_cls, path, title="Test", **info):
    """Stub a patched yt_dlp.YoutubeDL *class*.

    _stub_ytdlp patches `streamlit_app.yt_dlp`, which the AppTest cases cannot use
    -- AppTest re-executes the script each run and rebinds its imports, so those
    patch `yt_dlp.YoutubeDL` directly and call this instead.
    """
    ydl = MagicMock()
    ydl.extract_info.return_value = _ydl_info(title, **info)
    ydl.prepare_filename.return_value = str(path)
    mock_ydl_cls.return_value.__enter__.return_value = ydl
    return ydl


def _make_transcription(
    include_subtitles=False,
    file_stem="interview_transcript",
    filename="interview.mp3",
    text=None,
):
    result = MOCK_WHISPER_RESULT
    if text is not None:
        # Only the keys _display_transcription and _format_srt read.
        result = {"text": text, "segments": [{"start": 0.0, "end": 2.5, "text": text}]}
    return {
        "result": result,
        "file_stem": file_stem,
        "filename": filename,
        "include_subtitles": include_subtitles,
    }


def _expected_transcribe_kwargs(**overrides):
    base = {
        "language": None,
        "task": "transcribe",
        "initial_prompt": None,
        "no_verbatim": False,
        "condition_on_previous_text": True,
        "clip_timestamps": "0",
    }
    return base | overrides


def _handle_transcription_kwargs(**overrides):
    return {"language": None, "task": "transcribe", "include_subtitles": False} | overrides


def _ui_state(**overrides):
    base = {
        "language": None,
        "translate": False,
        "include_subtitles": False,
        "initial_prompt": None,
        "no_verbatim": False,
        "decode_independently": False,
        "clip_timestamps": "0",
    }
    return base | overrides


# --- Fixtures ---


@pytest.fixture(autouse=True)
def _clear_caches():
    _transcribe.clear()
    _fetch_youtube_audio.clear()
    _fetch_url_audio.clear()
    # The wrappers above only reach caches created by the imported module.
    # AppTest re-executes the script as a separate module with its own
    # cache store, which would otherwise persist for the whole session
    # and let a fetch-was-skipped assertion pass on a stale cache hit.
    st.cache_data.clear()
    # Both are required: the two fetch functions are @st.cache_resource, which
    # lives in a different singleton store than @st.cache_data (_transcribe).
    # Dropping this makes the tab-gate cases order-dependent.
    st.cache_resource.clear()


@pytest.fixture
def mock_mlx():
    with patch("streamlit_app.mlx_whisper") as m:
        m.transcribe.return_value = MOCK_WHISPER_RESULT
        yield m


@pytest.fixture
def mock_uploaded_file():
    return _make_file()


@pytest.fixture
def mock_st():
    with patch("streamlit_app.st") as m:
        m.session_state = {}
        m.columns.side_effect = lambda spec, **_: [
            MagicMock() for _ in range(spec if isinstance(spec, int) else len(spec))
        ]
        m.text_area.side_effect = lambda label, value, **_: value
        yield m


# --- Constants ---


def test_asr_model_repo():
    assert ASR_MODEL_REPO == "mlx-community/whisper-large-v3-turbo"


def test_transcript_formats():
    # Order is display order in the segmented control, and index 0 is the default.
    assert TRANSCRIPT_FORMATS == (FORMAT_PLAIN_TEXT, FORMAT_SUBTITLES)
    assert TRANSCRIPT_FORMATS == ("Plain text", "Subtitles")


def test_audio_formats():
    assert AUDIO_FORMATS == (
        "mp3",
        "m4a",
        "wav",
        "opus",
    )


def test_video_formats():
    assert VIDEO_FORMATS == (
        "mp4",
        "mov",
        "webm",
        "mkv",
    )


def test_format_list_fits_the_dropzone_hint():
    # The dropzone renders "<size> per file • MP3, M4A, ..." on a single
    # `white-space: nowrap; text-overflow: ellipsis` line with ~479px for the
    # format list at the centered layout's max width. Measured against Source
    # Sans 14px, each entry costs ~31-40px, so the list has to stay short.
    hint = ", ".join(f.upper() for f in AUDIO_FORMATS + VIDEO_FORMATS)
    assert len(hint) <= 60, f"{hint!r} will truncate in the uploader dropzone"


# --- _RemoteAudio / _fetch_youtube_audio / _fetch_url_audio ---


def test_remote_audio_adapter():
    audio = _RemoteAudio("video.m4a", b"audio bytes")
    assert audio.name == "video.m4a"
    assert audio.read() == b"audio bytes"


@patch("streamlit_app.yt_dlp")
def test_fetch_youtube_audio_returns_bytes_and_filename(mock_yt_dlp, tmp_path):
    fake_file = tmp_path / "Test_Video.m4a"
    fake_file.write_bytes(b"fake youtube audio")
    ydl = _stub_ytdlp(mock_yt_dlp, fake_file, title="Test Video")

    data, filename, mime = _fetch_youtube_audio("https://youtube.com/watch?v=fetch_bytes")

    assert data == b"fake youtube audio"
    assert filename == "Test_Video.m4a"
    assert mime == "audio/mp4"
    ydl.extract_info.assert_called_once_with(
        "https://youtube.com/watch?v=fetch_bytes",
        download=True,
    )


@patch("streamlit_app.yt_dlp")
def test_fetch_youtube_audio_uses_safe_options(mock_yt_dlp, tmp_path):
    fake_file = tmp_path / "video.webm"
    fake_file.write_bytes(b"webm bytes")
    _stub_ytdlp(mock_yt_dlp, fake_file, title="video")

    _fetch_youtube_audio("https://youtube.com/watch?v=safe_options")

    opts = mock_yt_dlp.YoutubeDL.call_args.args[0]
    assert opts["format"] == "bestaudio/best"
    assert opts["noplaylist"] is True
    assert opts["restrictfilenames"] is True
    assert opts["quiet"] is True


@patch("streamlit_app.urlopen")
def test_fetch_url_audio_returns_bytes_and_filename(mock_urlopen):
    _stub_urlopen(mock_urlopen, b"file bytes")

    data, filename, mime = _fetch_url_audio("https://example.com/audio.mp3")

    assert data == b"file bytes"
    assert filename == "audio.mp3"
    assert mime == "audio/mpeg"
    mock_urlopen.assert_called_once_with("https://example.com/audio.mp3", timeout=60)


@pytest.mark.parametrize(
    "url,expected_filename",
    [
        ("https://example.com/path/audio.wav?t=42", "audio.wav"),
        ("https://example.com/My%20Talk.mp3", "My Talk.mp3"),
        ("https://example.com/", "download"),
    ],
    ids=["strips_query", "decodes_percent", "fallback_when_no_path"],
)
@patch("streamlit_app.urlopen")
def test_fetch_url_audio_filename(mock_urlopen, url, expected_filename):
    _stub_urlopen(mock_urlopen, b"bytes")
    _, filename, _ = _fetch_url_audio(url)
    assert filename == expected_filename


@pytest.mark.parametrize(
    "content_type,url,expected",
    [
        ("audio/flac", "https://example.com/track.mp3", "audio/flac"),
        ("audio/mpeg; charset=binary", "https://example.com/x", "audio/mpeg"),
        ("VIDEO/MP4", "https://example.com/x", "video/mp4"),
        # The extension map cannot see a content-negotiated URL or an
        # extensionless redirect target -- exactly where it would answer
        # audio/wav, the mis-declaration this whole mechanism exists to avoid.
        ("audio/ogg", "https://example.com/stream?id=42", "audio/ogg"),
        # Untrustworthy declarations fall back to the extension. A server that
        # answers with an HTML error page or a generic blob must not get to set
        # the declared type of an <audio> element.
        ("text/html", "https://example.com/track.mp3", "audio/mpeg"),
        ("application/octet-stream", "https://example.com/track.m4a", "audio/mp4"),
        (None, "https://example.com/track.opus", "audio/ogg"),
        (None, "https://example.com/", "audio/wav"),
    ],
    ids=[
        "server_overrides_extension",
        "strips_parameters",
        "case_insensitive",
        "no_extension_to_guess",
        "rejects_html",
        "rejects_octet_stream",
        "absent_header",
        "absent_header_and_extension",
    ],
)
@patch("streamlit_app.urlopen")
def test_fetch_url_audio_prefers_the_served_content_type(mock_urlopen, content_type, url, expected):
    _stub_urlopen(mock_urlopen, b"bytes", content_type=content_type)
    _, _, mime = _fetch_url_audio(url)
    assert mime == expected


@patch("streamlit_app.yt_dlp")
def test_fetch_youtube_audio_declares_audio_only_containers(mock_yt_dlp, tmp_path):
    # bestaudio yields Opus in a WebM container with no video track. The extension
    # alone says video/webm, which is the same mis-declaration on an <audio>
    # element that audio/wav-for-everything was; `info` carries the codecs.
    fake_file = tmp_path / "Clip.webm"
    fake_file.write_bytes(b"opus bytes")
    _stub_ytdlp(mock_yt_dlp, fake_file, title="Clip", vcodec="none", acodec="opus")

    _, _, mime = _fetch_youtube_audio("https://youtube.com/watch?v=audio_only")

    assert mime == "audio/webm"


@patch("streamlit_app.yt_dlp")
def test_fetch_youtube_audio_keeps_the_container_type_when_a_video_track_exists(
    mock_yt_dlp, tmp_path
):
    # The `best` half of "bestaudio/best" can select a muxed stream, and an
    # unknown selection (no vcodec key) must not be assumed audio-only either.
    fake_file = tmp_path / "Clip.webm"
    fake_file.write_bytes(b"muxed bytes")
    _stub_ytdlp(mock_yt_dlp, fake_file, title="Clip", vcodec="vp9", acodec="opus")

    _, _, mime = _fetch_youtube_audio("https://youtube.com/watch?v=muxed")

    assert mime == "video/webm"


@patch("streamlit_app.MAX_DOWNLOAD_BYTES", 10)
@patch("streamlit_app.urlopen")
def test_fetch_url_audio_rejects_oversized_response(mock_urlopen):
    response = _stub_urlopen(mock_urlopen, b"x" * 11)

    with pytest.raises(RuntimeError, match="exceeds"):
        _fetch_url_audio("https://example.com/too-big.mp3")

    response.read.assert_called_once_with(11)


@patch("streamlit_app.MAX_DOWNLOAD_BYTES", 10)
@patch("streamlit_app.yt_dlp")
def test_fetch_youtube_audio_rejects_oversized_download(mock_yt_dlp, tmp_path):
    # yt-dlp's own max_filesize only fires where a Content-Length is known and
    # is never read by the fragmented downloaders, so the on-disk stat() gate is
    # what actually stops a livestream VOD from being slurped into one bytes
    # object. Deleting it must fail here.
    fake_file = tmp_path / "huge.m4a"
    fake_file.write_bytes(b"x" * 11)
    _stub_ytdlp(mock_yt_dlp, fake_file, title="huge")

    with pytest.raises(RuntimeError, match="exceeds"):
        _fetch_youtube_audio("https://youtube.com/watch?v=too_big")


@patch("streamlit_app.yt_dlp")
def test_fetch_youtube_audio_passes_max_filesize(mock_yt_dlp, tmp_path):
    fake_file = tmp_path / "video.webm"
    fake_file.write_bytes(b"webm bytes")
    _stub_ytdlp(mock_yt_dlp, fake_file, title="video")

    _fetch_youtube_audio("https://youtube.com/watch?v=max_filesize")

    assert mock_yt_dlp.YoutubeDL.call_args.args[0]["max_filesize"] == MAX_DOWNLOAD_BYTES


# --- _transcribe ---


def test_transcribe_success(mock_mlx):
    result = _transcribe(b"fake audio", ".mp3")
    assert result["text"] == "Hello world"
    assert len(result["segments"]) == 1


def test_transcribe_calls_mlx_with_correct_params(mock_mlx):
    _transcribe(b"fake audio params", ".mp3", language="en", task="transcribe")
    call = mock_mlx.transcribe.call_args
    assert call.args[0].endswith(".mp3")
    assert call.kwargs["path_or_hf_repo"] == "mlx-community/whisper-large-v3-turbo"
    assert call.kwargs["language"] == "en"
    assert call.kwargs["task"] == "transcribe"
    assert call.kwargs["no_speech_threshold"] == 0.6
    assert call.kwargs["logprob_threshold"] == -1.0
    assert call.kwargs["compression_ratio_threshold"] == 2.4


def test_transcribe_defaults(mock_mlx):
    _transcribe(b"fake audio defaults", ".mp3")
    kwargs = mock_mlx.transcribe.call_args.kwargs
    assert kwargs["language"] is None
    assert kwargs["task"] == "transcribe"
    assert kwargs["initial_prompt"] is None
    assert kwargs["word_timestamps"] is False
    assert kwargs["hallucination_silence_threshold"] is None
    assert kwargs["condition_on_previous_text"] is True
    assert kwargs["clip_timestamps"] == "0"


@pytest.mark.parametrize(
    "call_kwargs,expected",
    [
        ({"language": "fr", "task": "translate"}, {"task": "translate", "language": "fr"}),
        ({"initial_prompt": "Anthropic, MLX"}, {"initial_prompt": "Anthropic, MLX"}),
        ({"no_verbatim": True}, {"word_timestamps": True, "hallucination_silence_threshold": 2.0}),
        ({"condition_on_previous_text": False}, {"condition_on_previous_text": False}),
        ({"clip_timestamps": "30,90"}, {"clip_timestamps": "30,90"}),
    ],
    ids=["translate", "initial_prompt", "no_verbatim", "no_context", "single_clip"],
)
def test_transcribe_forwards_kwargs(mock_mlx, call_kwargs, expected):
    _transcribe(b"audio", ".mp3", **call_kwargs)
    kwargs = mock_mlx.transcribe.call_args.kwargs
    assert {k: kwargs[k] for k in expected} == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("0", ["0"]),
        ("30,90", ["30,90"]),
        ("0,60,120,180", ["0,60", "120,180"]),
        ("0,60,120", ["0,60", "120"]),
        (" 30 , 90 ", ["30,90"]),
        ("", ["0"]),
    ],
    ids=["full_file", "single_pair", "two_pairs", "trailing_start", "whitespace", "blank"],
)
def test_split_clips(raw, expected):
    assert _split_clips(raw) == expected


def test_transcribe_splits_multi_clip_ranges(mock_mlx):
    # mlx_whisper does not honour more than one clip: its decode loop binds
    # seek_clip_start and never re-seeks to it, so "0,60,120,180" decoded 0-180
    # straight through -- including the 60-120 the user excluded. One call per
    # pair is the workaround. This case replaces a `multi_clip` parametrize entry
    # that asserted only that the string was *forwarded*, which is exactly why a
    # fully green suite shipped the wrong audio for the app's own README example.
    _transcribe(b"audio", ".mp3", clip_timestamps="0,60,120,180")

    calls = [c.kwargs["clip_timestamps"] for c in mock_mlx.transcribe.call_args_list]
    assert calls == ["0,60", "120,180"]


def test_transcribe_keeps_a_trailing_start_open_ended(mock_mlx):
    # An odd trailing value is a start that runs to the end of the file -- that is
    # mlx_whisper's own reading of clip_timestamps -- so it must stay unpaired
    # rather than being padded into a degenerate range.
    _transcribe(b"audio", ".mp3", clip_timestamps="0,60,120")

    calls = [c.kwargs["clip_timestamps"] for c in mock_mlx.transcribe.call_args_list]
    assert calls == ["0,60", "120"]


def test_transcribe_single_clip_makes_one_call(mock_mlx):
    # The common path stays exactly one model pass with the string unchanged.
    _transcribe(b"audio", ".mp3", clip_timestamps="30,90")

    assert mock_mlx.transcribe.call_count == 1
    assert mock_mlx.transcribe.call_args.kwargs["clip_timestamps"] == "30,90"


def test_transcribe_merges_multi_clip_results(mock_mlx):
    # Segment timestamps are already absolute (mlx derives them from `seek`, which
    # starts at the clip's own start frame), so segments concatenate with no
    # adjustment and _format_srt is untouched. Language comes from the first clip.
    mock_mlx.transcribe.side_effect = [
        {
            "text": " one",
            "segments": [{"start": 0.0, "end": 1.0, "text": " one"}],
            "language": "en",
        },
        {
            "text": " two",
            "segments": [{"start": 120.0, "end": 121.0, "text": " two"}],
            "language": "fr",
        },
    ]

    result = _transcribe(b"audio", ".mp3", clip_timestamps="0,60,120,180")

    assert result["text"] == " one two"
    assert [s["start"] for s in result["segments"]] == [0.0, 120.0]
    assert result["language"] == "en"


def test_transcribe_raises_only_when_every_clip_is_silent(mock_mlx):
    # A silent clip inside a multi-clip range must not fail the whole file: the
    # empty-text guard applies to the merged text, not to each pass.
    silent = {"text": "   ", "segments": [], "language": "en"}
    spoken = {
        "text": " hello",
        "segments": [{"start": 120.0, "end": 121.0, "text": " hello"}],
        "language": "en",
    }
    mock_mlx.transcribe.side_effect = [silent, spoken]
    assert _transcribe(b"audio", ".mp3", clip_timestamps="0,60,120,180")["text"].strip() == "hello"

    mock_mlx.transcribe.side_effect = [silent, silent]
    with pytest.raises(RuntimeError, match="no text"):
        _transcribe(b"other audio", ".mp3", clip_timestamps="0,60,120,180")


def test_transcribe_no_text_raises(mock_mlx):
    mock_mlx.transcribe.return_value = {"text": "   ", "segments": [], "language": "en"}
    with pytest.raises(RuntimeError, match="no text"):
        _transcribe(b"fake audio empty", ".mp3")


def test_transcribe_cleans_up_temp_file(mock_mlx):
    called_paths = []

    def capture(path, **_):
        called_paths.append(path)
        return MOCK_WHISPER_RESULT

    mock_mlx.transcribe.side_effect = capture
    _transcribe(b"fake audio cleanup", ".mp3")
    assert len(called_paths) == 1
    assert not Path(called_paths[0]).exists()


# --- _handle_transcription ---


@patch("streamlit_app._transcribe", return_value=MOCK_WHISPER_RESULT)
def test_handle_transcription_stores_result(mock_transcribe, mock_st, mock_uploaded_file):
    _handle_transcription(
        [mock_uploaded_file], language=None, task="transcribe", include_subtitles=False
    )

    transcriptions = mock_st.session_state["transcription"]
    assert len(transcriptions) == 1
    data = transcriptions[0]
    assert data["result"] == MOCK_WHISPER_RESULT
    assert data["file_stem"] == "interview_mp3_transcript"
    assert data["filename"] == "interview.mp3"
    assert data["include_subtitles"] is False


@patch("streamlit_app._transcribe", return_value=MOCK_WHISPER_RESULT)
def test_handle_transcription_stores_include_subtitles_true(
    mock_transcribe, mock_st, mock_uploaded_file
):
    _handle_transcription(
        [mock_uploaded_file], language=None, task="transcribe", include_subtitles=True
    )
    assert mock_st.session_state["transcription"][0]["include_subtitles"] is True


@patch("streamlit_app._transcribe", side_effect=RuntimeError("Transcription produced no text"))
def test_handle_transcription_runtime_error(mock_transcribe, mock_st, mock_uploaded_file):
    _handle_transcription(
        [mock_uploaded_file], language=None, task="transcribe", include_subtitles=False
    )
    # `\:` renders as a plain colon. _error escapes the whole message rather than
    # only the untrusted spans, so the fixed literals pick up backslashes too.
    mock_st.error.assert_called_once_with(
        r"Transcription failed for interview.mp3\: Transcription produced no text",
        icon=":material/error:",
    )
    assert mock_st.session_state["transcription"] == []


@patch("streamlit_app._transcribe", side_effect=ValueError("unexpected"))
def test_handle_transcription_unexpected_error(mock_transcribe, mock_st, mock_uploaded_file):
    _handle_transcription(
        [mock_uploaded_file], language=None, task="transcribe", include_subtitles=False
    )
    mock_st.error.assert_called_once_with(
        r"Unexpected error for interview.mp3\: unexpected", icon=":material/error:"
    )
    mock_st.exception.assert_called_once()


@pytest.mark.parametrize(
    "error,expected",
    [
        (RuntimeError("boom"), r"Transcription failed for my\_song \[live\].mp3\: boom"),
        (ValueError("boom"), r"Unexpected error for my\_song \[live\].mp3\: boom"),
    ],
)
def test_handle_transcription_escapes_filename_in_error(error, expected, mock_st):
    # st.error renders *full* Markdown — a strictly larger subset than the label
    # subset _display_transcription's subheader escapes for. Filenames arrive
    # percent-decoded from the URL tab, so they are untrusted.
    with patch("streamlit_app._transcribe", side_effect=error):
        _handle_transcription(
            [_make_file(name="my_song [live].mp3")], **_handle_transcription_kwargs()
        )

    mock_st.error.assert_called_once_with(expected, icon=":material/error:")


@patch("streamlit_app._transcribe", return_value=MOCK_WHISPER_RESULT)
def test_handle_transcription_escapes_filename_in_status_label(mock_transcribe, mock_st):
    # The status label renders on every file, not just failures, and its Markdown
    # subset includes images — so an unescaped `![](https://host/x.png)` in a
    # filename would fetch on the happy path.
    _handle_transcription([_make_file(name="clip [1].mp3")], **_handle_transcription_kwargs())

    status = mock_st.status.return_value.__enter__.return_value
    status.update.assert_any_call(label=r"Transcribing clip \[1\].mp3 (1/1)...")


@pytest.mark.parametrize(
    "count,opening,closing",
    [
        (1, "Transcribing 1 file...", "Transcribed 1/1 file"),
        (2, "Transcribing 2 files...", "Transcribed 2/2 files"),
    ],
    ids=["single", "batch"],
)
@patch("streamlit_app._transcribe", return_value=MOCK_WHISPER_RESULT)
def test_handle_transcription_pluralizes_the_status_labels(
    mock_transcribe, mock_st, count, opening, closing
):
    # Both of these labels used to read "file(s)". Neither was asserted anywhere,
    # which is why the parenthesised form survived -- and a single file, the case
    # it gets wrong, is the common one.
    _handle_transcription(
        [_make_file(name=f"clip{i}.mp3") for i in range(count)],
        **_handle_transcription_kwargs(),
    )

    mock_st.status.assert_called_once_with(opening, expanded=True)
    status = mock_st.status.return_value.__enter__.return_value
    status.update.assert_called_with(label=closing, state="complete")


@pytest.mark.parametrize(
    "message,expected",
    [
        (
            "Could not download from YouTube: Unsupported URL: "
            "https://youtu.be/![](https://attacker.example/p.png)",
            # `(` and `)` are deliberately not in the class and do not need to be:
            # an image needs the bracket half, and `[`/`]` are escaped, so the
            # parenthesized destination is inert on its own.
            r"Could not download from YouTube\: Unsupported URL\: "
            r"https\://youtu.be/!\[\](https\://attacker.example/p.png)",
        ),
        ("Invalid time range: '![](https://x/p.png)' is not a number.", None),
    ],
    ids=["fetch_exception", "validation_message"],
)
def test_error_escapes_the_whole_message(mock_st, message, expected):
    # The exception text is untrusted, not just the filename: yt-dlp's
    # UnsupportedError is literally f"Unsupported URL: {url}" and YOUTUBE_URL_RE is
    # prefix-anchored, so a crafted URL reaches the alert body verbatim -- and
    # st.error's body renders full Markdown, so `![](...)` fires a request.
    _error(message)

    (rendered,), kwargs = mock_st.error.call_args
    assert kwargs == {"icon": ERROR_ICON}
    assert "![](" not in rendered
    if expected is not None:
        assert rendered == expected


def test_handle_transcription_rewinds_the_cursor_before_reading(mock_st):
    # UploadedFile subclasses io.BytesIO, and the deserialized widget value is
    # cached in session state (WStates.__getitem__ stores Value(deserialized)), so
    # the same object -- and the same cursor -- survives every rerun. read() leaves
    # it at EOF. st.audio used to rewind it as a side effect of rendering a preview
    # (_marshall_av_media calls data.seek(0)); the Record tab renders none, so a
    # second Transcribe on an unchanged recording would otherwise transcribe b"".
    rec = UploadedFileRec("id", "recording.wav", "audio/wav", b"real audio bytes")
    recording = UploadedFile(rec, FileURLs())

    with patch("streamlit_app._transcribe", return_value=MOCK_WHISPER_RESULT) as mock_transcribe:
        _handle_transcription([recording], **_handle_transcription_kwargs())
        _handle_transcription([recording], **_handle_transcription_kwargs())

    assert [c.args[0] for c in mock_transcribe.call_args_list] == [b"real audio bytes"] * 2


def test_handle_transcription_collapses_whitespace_in_filename(mock_st):
    # _fetch_url_audio percent-decodes off the URL path, so `%0A` arrives as a real
    # newline. st.error's body is the one sink rendered without the frontend's
    # isLabel flag, which is what auto-escapes `#`/`>` and strips block elements
    # elsewhere — so an uncollapsed newline would open a heading and a blockquote
    # inside the error box. Every block construct needs a line start.
    with patch("streamlit_app._transcribe", side_effect=RuntimeError("boom")):
        _handle_transcription(
            [_make_file(name="clip\n\n# Big\n\n> quote.mp3")], **_handle_transcription_kwargs()
        )

    # `#` and `>` are deliberately *not* in the escape class — losing the line
    # start is what defuses them, so they survive as literal characters.
    (message,), kwargs = mock_st.error.call_args
    assert "\n" not in message
    assert message == r"Transcription failed for clip # Big > quote.mp3\: boom"
    assert kwargs == {"icon": ":material/error:"}


@pytest.mark.parametrize(
    "ui_kwargs,expected_overrides",
    [
        (
            _handle_transcription_kwargs(language="fr", task="translate", include_subtitles=True),
            {"language": "fr", "task": "translate"},
        ),
        (
            _handle_transcription_kwargs(initial_prompt="Anthropic, MLX"),
            {"initial_prompt": "Anthropic, MLX"},
        ),
        (_handle_transcription_kwargs(no_verbatim=True), {"no_verbatim": True}),
        (
            _handle_transcription_kwargs(condition_on_previous_text=False),
            {"condition_on_previous_text": False},
        ),
        (_handle_transcription_kwargs(clip_timestamps="30,90"), {"clip_timestamps": "30,90"}),
    ],
    ids=["translate", "initial_prompt", "no_verbatim", "no_context", "clip"],
)
@patch("streamlit_app._transcribe", return_value=MOCK_WHISPER_RESULT)
def test_handle_transcription_forwards_kwargs(
    mock_transcribe, mock_st, mock_uploaded_file, ui_kwargs, expected_overrides
):
    _handle_transcription([mock_uploaded_file], **ui_kwargs)
    mock_transcribe.assert_called_once_with(
        b"fake audio bytes",
        ".mp3",
        **_expected_transcribe_kwargs(**expected_overrides),
    )


@patch("streamlit_app._transcribe", return_value=MOCK_WHISPER_RESULT)
def test_handle_transcription_multiple_files(mock_transcribe, mock_st):
    files = [_make_file("first.mp3", b"first audio"), _make_file("second.mp3", b"second audio")]
    _handle_transcription(files, language=None, task="transcribe", include_subtitles=False)

    transcriptions = mock_st.session_state["transcription"]
    assert len(transcriptions) == 2
    assert transcriptions[0]["filename"] == "first.mp3"
    assert transcriptions[1]["filename"] == "second.mp3"
    assert mock_transcribe.call_count == 2


@patch("streamlit_app._transcribe")
def test_handle_transcription_partial_failure(mock_transcribe, mock_st):
    mock_transcribe.side_effect = [
        MOCK_WHISPER_RESULT,
        RuntimeError("Transcription produced no text"),
        MOCK_WHISPER_RESULT,
    ]
    files = [_make_file(f"{stem}.mp3") for stem in ("first", "second", "third")]

    _handle_transcription(files, language=None, task="transcribe", include_subtitles=False)

    transcriptions = mock_st.session_state["transcription"]
    assert len(transcriptions) == 2
    assert transcriptions[0]["filename"] == "first.mp3"
    assert transcriptions[1]["filename"] == "third.mp3"
    mock_st.error.assert_called_once_with(
        r"Transcription failed for second.mp3\: Transcription produced no text",
        icon=":material/error:",
    )


@patch("streamlit_app._transcribe", side_effect=RuntimeError("boom"))
def test_handle_transcription_renders_errors_after_the_status_closes(mock_transcribe, mock_st):
    # st.status collapses on its first update(label=...) -- update() clears the
    # proto's `expanded` field unless it is passed again, and the frontend's
    # label-change branch then resets the open state to that now-false value. So
    # an alert written into the status body during the loop lands in a box the
    # user cannot open: a failed file rendered a green check, "Transcribed 0/1
    # file", and no visible explanation anywhere on the page. The fix is
    # positional, so the assertion is too -- st.error must run after the status
    # context manager has exited.
    _handle_transcription([_make_file()], **_handle_transcription_kwargs())

    names = [c[0] for c in mock_st.mock_calls]
    assert names.index("error") > names.index("status().__exit__")
    # And a batch that lost a file is not "complete".
    status = mock_st.status.return_value.__enter__.return_value
    status.update.assert_called_with(label="Transcribed 0/1 file", state="error")


@patch("streamlit_app._transcribe")
def test_handle_transcription_keeps_finished_files_when_interrupted(mock_transcribe, mock_st):
    # Streamlit aborts a running script by raising RerunException (a BaseException,
    # so `except Exception` does not catch it) at the next ForwardMsg — which
    # status.update() is. Results published so far must survive that unwind.
    class _Interrupt(BaseException):
        pass

    mock_transcribe.side_effect = [MOCK_WHISPER_RESULT, _Interrupt()]
    files = [_make_file(f"{stem}.mp3") for stem in ("first", "second")]

    with pytest.raises(_Interrupt):
        _handle_transcription(files, language=None, task="transcribe", include_subtitles=False)

    transcriptions = mock_st.session_state["transcription"]
    assert len(transcriptions) == 1
    assert transcriptions[0]["filename"] == "first.mp3"


@patch("streamlit_app.mx")
@patch("streamlit_app._transcribe", return_value=MOCK_WHISPER_RESULT)
def test_handle_transcription_reclaims_the_mlx_cache(mock_transcribe, mock_mx, mock_st):
    # mlx keeps freed device buffers on a free list instead of returning them, so
    # a finished batch stays resident for the life of the server process.
    # Measured against whisper-large-v3-turbo with word_timestamps=True: 894 MB
    # held after an 8-second file, 1.25 GB after two minutes, all reclaimed in
    # 4.2 ms, with the model itself untouched in `active` memory.
    _handle_transcription([_make_file()], **_handle_transcription_kwargs())

    mock_mx.clear_cache.assert_called_once_with()


@patch("streamlit_app.mx")
def test_handle_transcription_reclaims_the_mlx_cache_when_interrupted(mock_mx, mock_st):
    # This is why it is a `finally` and not a call after the block. A tab switch
    # -- or any widget -- raises RerunException, a BaseException, mid-batch; an
    # interrupted run would otherwise hold its buffers until whenever the next
    # transcription happens to finish, which may be never.
    class _Interrupt(BaseException):
        pass

    with (
        patch("streamlit_app._transcribe", side_effect=_Interrupt()),
        pytest.raises(_Interrupt),
    ):
        _handle_transcription([_make_file()], **_handle_transcription_kwargs())

    mock_mx.clear_cache.assert_called_once_with()


@patch("streamlit_app._transcribe")
def test_handle_transcription_clears_previous_batch(mock_transcribe, mock_st):
    mock_st.session_state["transcription"] = [_make_transcription(filename="stale.mp3")]
    mock_transcribe.side_effect = RuntimeError("Transcription produced no text")

    _handle_transcription([_make_file()], language=None, task="transcribe", include_subtitles=False)

    assert mock_st.session_state["transcription"] == []


@patch("streamlit_app._transcribe")
def test_handle_transcription_bumps_batch_id(mock_transcribe, mock_st):
    # The batch id namespaces _display_transcription's widget keys, so every
    # batch must advance it — including one where every file fails, since the
    # stale text area would otherwise outlive the results it was rendered from.
    mock_transcribe.return_value = MOCK_WHISPER_RESULT
    kwargs = _handle_transcription_kwargs()

    _handle_transcription([_make_file()], **kwargs)
    assert mock_st.session_state["batch_id"] == 1

    _handle_transcription([_make_file("second.mp3")], **kwargs)
    assert mock_st.session_state["batch_id"] == 2

    mock_transcribe.side_effect = RuntimeError("Transcription produced no text")
    _handle_transcription([_make_file("third.mp3")], **kwargs)
    assert mock_st.session_state["batch_id"] == 3


# --- _transcription_kwargs ---


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({"translate": True, "language": "fr"}, {"task": "translate"}),
        ({"translate": False, "language": "fr"}, {"task": "transcribe"}),
        ({"decode_independently": True}, {"condition_on_previous_text": False}),
        ({"decode_independently": False}, {"condition_on_previous_text": True}),
    ],
    ids=["translate_on", "translate_off", "no_context", "with_context"],
)
def test_transcription_kwargs_mappings(overrides, expected):
    kwargs = _transcription_kwargs(**_ui_state(**overrides))
    assert {k: kwargs[k] for k in expected} == expected


def test_transcription_kwargs_passes_through_unchanged_fields():
    kwargs = _transcription_kwargs(
        **_ui_state(
            language="en",
            include_subtitles=True,
            initial_prompt="hello",
            no_verbatim=True,
            clip_timestamps="30,90",
        )
    )
    assert kwargs["language"] == "en"
    assert kwargs["include_subtitles"] is True
    assert kwargs["initial_prompt"] == "hello"
    assert kwargs["no_verbatim"] is True
    assert kwargs["clip_timestamps"] == "30,90"


# --- _display_transcription ---


def test_display_transcription_no_session_state(mock_st):
    _display_transcription()
    mock_st.text_area.assert_not_called()


def test_display_transcription_shows_transcript(mock_st):
    mock_st.session_state["transcription"] = [_make_transcription()]

    _display_transcription()

    mock_st.text_area.assert_called_once_with(
        "Transcript",
        "Hello world",
        height=300,
        label_visibility="collapsed",
        key="transcript_b0_0",
    )
    mock_st.subheader.assert_called_once_with("interview.mp3")


def test_display_transcription_txt_download(mock_st):
    mock_st.session_state["transcription"] = [_make_transcription()]

    _display_transcription()

    mock_st.download_button.assert_called_once_with(
        "Download",
        "Hello world",
        "interview_transcript.txt",
        "text/plain",
        icon=":material/download:",
        key="download_txt_b0_0",
        help=DOWNLOAD_HELP,
        on_click="ignore",
        width=SELECT_WIDTH,
    )


def test_display_transcription_srt_download(mock_st):
    mock_st.session_state["transcription"] = [_make_transcription(include_subtitles=True)]

    _display_transcription()

    mock_st.download_button.assert_called_once_with(
        "Download",
        SRT_HELLO,
        "interview_transcript.srt",
        "application/x-subrip",
        icon=":material/download:",
        key="download_srt_b0_0",
        help=DOWNLOAD_HELP,
        on_click="ignore",
        width=SELECT_WIDTH,
    )


def test_display_transcription_subtitles_on(mock_st):
    mock_st.session_state["transcription"] = [_make_transcription(include_subtitles=True)]

    _display_transcription()

    mock_st.text_area.assert_called_once_with(
        "Transcript",
        SRT_HELLO,
        height=300,
        label_visibility="collapsed",
        key="transcript_b0_0",
    )
    mock_st.subheader.assert_called_once_with("interview.mp3")


def test_display_transcription_download_reflects_edits(mock_st):
    mock_st.text_area.side_effect = lambda label, value, **_: "edited transcript text"
    mock_st.session_state["transcription"] = [_make_transcription()]

    _display_transcription()

    mock_st.download_button.assert_called_once_with(
        "Download",
        "edited transcript text",
        "interview_transcript.txt",
        "text/plain",
        icon=":material/download:",
        key="download_txt_b0_0",
        help=DOWNLOAD_HELP,
        on_click="ignore",
        width=SELECT_WIDTH,
    )


def test_display_transcription_right_aligns_download(mock_st):
    mock_st.session_state["transcription"] = [_make_transcription()]
    _display_transcription()
    # "right", never "distribute": a standalone element in a distributed
    # container is left-aligned, which would silently unstick the shared edge.
    mock_st.container.assert_any_call(horizontal=True, horizontal_alignment="right")


def test_display_transcription_wraps_each_result_in_a_bordered_container(mock_st):
    mock_st.session_state["transcription"] = [
        _make_transcription(filename="first.mp3"),
        _make_transcription(filename="second.mp3"),
    ]

    _display_transcription()

    assert mock_st.container.call_args_list.count(call(border=True)) == 2


def test_display_transcription_multiple_files(mock_st):
    mock_st.session_state["transcription"] = [
        _make_transcription(file_stem="first_transcript", filename="first.mp3"),
        _make_transcription(file_stem="second_transcript", filename="second.mp3"),
    ]

    _display_transcription()

    assert mock_st.text_area.call_count == 2
    assert mock_st.download_button.call_count == 2
    assert mock_st.subheader.call_count == 2
    mock_st.subheader.assert_any_call("first.mp3")
    mock_st.subheader.assert_any_call("second.mp3")


def test_display_transcription_escapes_filename_in_subheader(mock_st):
    mock_st.session_state["transcription"] = [_make_transcription(filename="my_song [live].mp3")]

    _display_transcription()

    mock_st.subheader.assert_called_once_with(r"my\_song \[live\].mp3")


def test_display_transcription_collapses_whitespace_in_subheader(mock_st):
    # st.subheader is NOT protected by the frontend's isLabel guard, contrary to
    # what a Markdown "label subset" implies: the heading component does
    # `[first, ...rest] = body.split("\n")` and renders `rest` through a bare
    # StreamlitMarkdown with no isLabel and no disallowedElements. So every line
    # after the first is full Markdown -- the same unguarded sink as st.error's
    # body. This filename is reachable verbatim from a URL ending
    # `/clip%0A%0A%23%20Big%0A%0A%3E%20quote.mp3`, since _fetch_url_audio
    # percent-decodes and the raw name is what lands in the transcription dict.
    mock_st.session_state["transcription"] = [
        _make_transcription(filename="clip\n\n# Big\n\n> quote.mp3")
    ]

    _display_transcription()

    mock_st.subheader.assert_called_once_with("clip # Big > quote.mp3")


def test_display_transcription_keys_are_namespaced_by_batch(mock_st):
    mock_st.session_state["batch_id"] = 7
    mock_st.session_state["transcription"] = [_make_transcription()]

    _display_transcription()

    assert mock_st.text_area.call_args.kwargs["key"] == "transcript_b7_0"
    assert mock_st.download_button.call_args.kwargs["key"] == "download_txt_b7_0"


# --- formatting helpers ---


@pytest.mark.parametrize(
    "code,expected",
    [(None, "Detect"), ("en", "English"), ("fr", "French")],
    ids=["none_returns_detect", "lowercase_code", "title_cased"],
)
def test_format_language(code, expected):
    assert _format_language(code) == expected


@pytest.mark.parametrize(
    "count,expected",
    [(1, "1 file"), (2, "2 files"), (0, "0 files")],
    ids=["singular", "plural", "zero_is_plural"],
)
def test_plural(count, expected):
    assert _plural(count, "file") == expected


@pytest.mark.parametrize(
    "seconds,decimal_marker,expected",
    [
        (0.0, ".", "00:00:00.000"),
        (65.5, ".", "00:01:05.500"),
        (3661.123, ".", "01:01:01.123"),
        (65.5, ",", "00:01:05,500"),
    ],
    ids=["zero", "minutes_seconds", "hours", "comma_marker"],
)
def test_format_timestamp(seconds, decimal_marker, expected):
    assert _format_timestamp(seconds, decimal_marker=decimal_marker) == expected


def test_format_srt():
    assert _format_srt(MOCK_WHISPER_RESULT) == SRT_HELLO


def test_format_srt_multiple_segments():
    result = {
        "segments": [
            {"start": 0.0, "end": 2.5, "text": " Hello"},
            {"start": 2.5, "end": 5.0, "text": " World"},
        ]
    }
    assert _format_srt(result) == (
        "1\n00:00:00,000 --> 00:00:02,500\nHello\n\n2\n00:00:02,500 --> 00:00:05,000\nWorld\n"
    )


def test_format_srt_escapes_arrow():
    result = {
        "segments": [
            {"start": 0.0, "end": 2.5, "text": " before --> after"},
        ]
    }
    assert _format_srt(result) == "1\n00:00:00,000 --> 00:00:02,500\nbefore -> after\n"


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("interview.mp3", "audio/mpeg"),
        ("interview.m4a", "audio/mp4"),
        ("interview.wav", "audio/wav"),
        ("interview.opus", "audio/ogg"),
        ("clip.mp4", "video/mp4"),
        ("clip.mov", "video/quicktime"),
        ("clip.webm", "video/webm"),
        ("clip.mkv", "video/x-matroska"),
        ("SHOUTING.MP3", "audio/mpeg"),
        ("archive.flac", "audio/flac"),
        # Neither is in AUDIO_FORMATS/VIDEO_FORMATS, and both are reachable: the
        # YouTube and URL fetches never consult those tuples, and _fetch_url_audio
        # falls back to the extensionless "download" for an empty URL path.
        ("mystery.xyz", "audio/wav"),
        ("download", "audio/wav"),
    ],
    ids=[
        "mp3",
        "m4a",
        "wav",
        "opus",
        "mp4",
        "mov",
        "webm",
        "mkv",
        "uppercase",
        "not_in_upload_formats",
        "unknown_extension",
        "no_extension",
    ],
)
def test_media_mime(filename, expected):
    # st.audio's format= default is "audio/wav" for every input, and it becomes the
    # served Content-Type and the media URL's extension — not a hint.
    assert _media_mime(filename) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("interview.mp3", "interview.mp3"),
        ("interview_part_1.mp3", r"interview\_part\_1.mp3"),
        ("Song [Official Video].mp3", r"Song \[Official Video\].mp3"),
        ("a*b`c~d:e$f&g", r"a\*b\`c\~d\:e\$f\&g"),
        (r"back\slash", r"back\\slash"),
        # `&` is escaped because micromark's characterReference is a *parse-time*
        # construct: unescaped, `&#58;` decodes to a colon early enough for the
        # frontend's post-parse pass to turn `:streamlit:` into the logo image.
        ("Rock &amp; Roll.mp3", r"Rock \&amp; Roll.mp3"),
        ("clip&#58;streamlit&#58;.mp3", r"clip\&#58;streamlit\&#58;.mp3"),
        # Block markers are defused by losing the line start, not by escaping --
        # `#` and `>` come back literal. Both sinks that render without the
        # frontend's isLabel flag (st.error's body and st.subheader's lines after
        # the first) depend on this, so it lives here rather than at a call site.
        ("clip\n\n# Big\n\n> quote.mp3", "clip # Big > quote.mp3"),
        ("spaced   out\ttabbed.mp3", "spaced out tabbed.mp3"),
    ],
    ids=[
        "plain",
        "underscores",
        "brackets",
        "all_specials",
        "backslash",
        "named_entity",
        "numeric_entity",
        "block_markers",
        "whitespace_runs",
    ],
)
def test_escape_markdown(raw, expected):
    assert _escape_markdown(raw) == expected


@pytest.mark.parametrize(
    "raw",
    ["", "30,90", "0,60,120,180", " 30 , 90 ", "0,0.5", "1.5,2", "30", "30,60,90"],
    ids=[
        "blank",
        "single",
        "multi",
        "whitespace",
        "decimal_start",
        "decimal_pair",
        "trailing_start",
        "trailing_start_multi",
    ],
)
def test_validate_time_range_valid(raw):
    assert _validate_time_range(raw) is None


@pytest.mark.parametrize(
    "raw",
    [
        "abc",
        "30,abc",
        "-5,10",
        "90,30",
        "30,30",
        "30,",
        ",30",
        "60,90,0,30",
        # float() accepts all four of these, and every comparison against a nan is
        # False -- so they passed validation, enabled Transcribe, and failed
        # inside mlx_whisper instead. "1e400" overflows to inf on the way in.
        "nan",
        "inf",
        "0,inf",
        "1e400",
    ],
    ids=[
        "non_numeric",
        "non_numeric_pair",
        "negative",
        "end_before_start",
        "equal_pair",
        "trailing_comma",
        "leading_comma",
        "out_of_order",
        "nan",
        "inf",
        "inf_end",
        "overflow_literal",
    ],
)
def test_validate_time_range_invalid(raw):
    assert _validate_time_range(raw) is not None


# --- module UI (AppTest) ---
#
# These exercise the module-level UI (page config, tabs, buttons, fragment) by
# running the real script through Streamlit's AppTest runtime, complementing the
# mocked-`st` unit tests above.


APP_PATH = Path(__file__).resolve().parent.parent / "streamlit_app.py"


def _run_app(transcription=None, active_tab=None):
    at = AppTest.from_file(str(APP_PATH), default_timeout=5)
    if transcription is not None:
        at.session_state["transcription"] = transcription
    if active_tab is not None:
        # AppTest has no tab-selection API; st.tabs' `key` holds the active label.
        at.session_state["input_tabs"] = active_tab
    return at.run()


def test_page_config():
    assert PAGE_CONFIG == {
        "page_title": "Whisper Transcribe",
        "page_icon": ":material/graphic_eq:",
        "layout": "centered",
    }


def test_app_renders_without_exception():
    at = _run_app()
    assert not at.exception
    assert [t.value for t in at.title] == ["Whisper Transcribe"]


def test_tabs_have_material_icon_labels():
    at = _run_app()
    assert [t.label for t in at.tabs] == [
        ":material/upload: Upload",
        ":material/mic: Record",
        ":material/smart_display: YouTube",
        ":material/link: URL",
    ]


def test_transcribe_button_has_icon_and_is_disabled_without_audio():
    button = _run_app().button[0]
    assert button.label == "Transcribe"
    assert button.icon == ":material/graphic_eq:"
    assert button.disabled is True


def test_invalid_time_range_shows_inline_error():
    at = _run_app()
    next(t for t in at.text_input if t.label == "Time range").set_value("90,30").run()
    assert [e.value for e in at.error] == [_validate_time_range("90,30")]


def test_valid_time_range_shows_no_error():
    at = _run_app()
    next(t for t in at.text_input if t.label == "Time range").set_value("30,90").run()
    assert at.error == []


def test_results_render_download_button_with_icon():
    # Seeded results render through the st.fragment(_display_transcription)() wrap.
    at = _run_app([_make_transcription()])
    assert not at.exception
    assert [s.value for s in at.subheader] == ["interview.mp3"]
    assert at.text_area[0].value == "Hello world"
    download = at.get("download_button")[0]
    assert download.label == "Download"
    assert download.icon == ":material/download:"


def test_no_results_renders_no_download_button():
    assert _run_app().get("download_button") == []


def _publish(at, transcription, batch):
    # Mirrors _handle_transcription's publish: the results plus the batch id
    # that namespaces the transcript widget keys.
    at.session_state["transcription"] = transcription
    at.session_state["batch_id"] = batch
    return at.run()


def test_new_batch_replaces_previous_transcript_text():
    # A keyed st.text_area restores its session-state value and ignores the
    # `value` argument, so a batch-invariant key renders the *previous* batch's
    # text under the new filename -- and the Download button, whose payload is
    # the text area's return value, serves it. This needs two renders with
    # different data; no single-render test can see it.
    at = AppTest.from_file(str(APP_PATH), default_timeout=5)
    _publish(at, [_make_transcription(filename="first.mp3")], batch=1)
    assert at.text_area[0].value == "Hello world"

    # Edits must still stick *within* a batch -- that is the point of the key.
    at.text_area[0].set_value("edited by hand").run()
    assert at.text_area[0].value == "edited by hand"

    _publish(at, [_make_transcription(filename="second.mp3", text="Second file text")], batch=2)
    assert not at.exception
    assert [s.value for s in at.subheader] == ["second.mp3"]
    assert at.text_area[0].value == "Second file text"


def test_new_batch_replaces_previous_transcript_when_subtitles_toggled():
    # Flipping include_subtitles changes only the *download* key (txt -> srt),
    # so without the batch namespace the transcript text area stays stale and
    # the SRT cues never reach the screen.
    at = AppTest.from_file(str(APP_PATH), default_timeout=5)
    _publish(at, [_make_transcription(filename="first.mp3")], batch=1)
    assert at.text_area[0].value == "Hello world"

    _publish(
        at,
        [
            _make_transcription(
                filename="second.mp3", text="Second file text", include_subtitles=True
            )
        ],
        batch=2,
    )
    assert not at.exception
    assert at.text_area[0].value == "1\n00:00:00,000 --> 00:00:02,500\nSecond file text\n"


# The remote-fetch tabs gate their download on `tab.open` because st.tabs
# computes hidden tab bodies by default. Network entry points are patched on
# their own modules (urllib.request / yt_dlp) rather than on streamlit_app,
# because AppTest re-executes the script each run and rebinds its imports.

UPLOAD_TAB = ":material/upload: Upload"
RECORD_TAB = ":material/mic: Record"
YOUTUBE_TAB = ":material/smart_display: YouTube"
URL_TAB = ":material/link: URL"


def _type_url(label, url, active_tab):
    at = _run_app(active_tab=active_tab)
    next(t for t in at.text_input if t.label == label).set_value(url)
    return at.run()


def _typed_value(at, label):
    return next(t for t in at.text_input if t.label == label).value


@pytest.mark.parametrize(
    "active_tab,expected_calls",
    [(UPLOAD_TAB, 0), (URL_TAB, 1)],
    ids=["skipped_when_hidden", "fetched_when_active"],
)
def test_url_fetch_gated_on_tab_visibility(active_tab, expected_calls):
    url = "https://example.com/audio.mp3"
    with patch("urllib.request.urlopen") as mock_urlopen:
        _stub_urlopen(mock_urlopen, b"file bytes")
        at = _type_url("Audio/video file URL", url, active_tab)
    assert not at.exception
    assert mock_urlopen.call_count == expected_calls
    # The URL is retained either way — only the fetch is gated, not the widget.
    assert _typed_value(at, "Audio/video file URL") == url


@pytest.mark.parametrize(
    "active_tab,expected_calls",
    [(UPLOAD_TAB, 0), (YOUTUBE_TAB, 1)],
    ids=["skipped_when_hidden", "fetched_when_active"],
)
def test_youtube_fetch_gated_on_tab_visibility(active_tab, expected_calls, tmp_path):
    url = "https://youtube.com/watch?v=gated"
    fake_file = tmp_path / "Clip.m4a"
    fake_file.write_bytes(b"yt bytes")
    with patch("yt_dlp.YoutubeDL") as mock_ydl_cls:
        ydl = _stub_ytdlp_class(mock_ydl_cls, fake_file, "Clip")
        at = _type_url("YouTube URL", url, active_tab)
    assert not at.exception
    assert ydl.extract_info.call_count == expected_calls
    assert _typed_value(at, "YouTube URL") == url


def _tab(at, label):
    return next(t for t in at.tabs if t.label == label)


# The fetches themselves run below every control so a slow download does not
# grey out the language selector, the toggles, and Advanced options. They write
# back into an st.container() reserved inside their tab, so the preview must
# still resolve *within* that tab -- dropping the slot would strand it at the
# bottom of the page, under the Transcribe button.


def _assert_declared_mime(element, mime):
    """Assert an st.audio element was given `format=mime`, via its media URL.

    The mimetype is not in the proto, but Streamlit derives the URL's extension
    from it, so the suffix is the only observable proxy. Both sides are computed
    with Streamlit's own lookup rather than hardcoded, because that lookup is
    `mimetypes.guess_extension` -- the same platform-varying table MEDIA_MIME_TYPES
    exists to avoid. The inequality guard keeps the assertion from passing
    vacuously if a host's tables map both types to the same (or an empty) suffix.
    """
    expected = get_extension_for_mimetype(mime)
    assert expected and expected != get_extension_for_mimetype(DEFAULT_MEDIA_MIME)
    assert element.proto.url.endswith(expected)


def test_url_preview_renders_inside_its_tab():
    with patch("urllib.request.urlopen") as mock_urlopen:
        _stub_urlopen(mock_urlopen, b"file bytes")
        at = _type_url("Audio/video file URL", "https://example.com/audio.mp3", URL_TAB)
    assert not at.exception
    assert len(_tab(at, URL_TAB).get("audio")) == 1
    assert _tab(at, UPLOAD_TAB).get("audio") == []
    # Also pins the format= wiring: test_media_mime and the _fetch_* tests cover
    # which mimetype is chosen, but only a rendered element shows it was passed.
    # Dropping format= from the call site serves every preview as .wav again.
    _assert_declared_mime(_tab(at, URL_TAB).get("audio")[0], "audio/mpeg")


def test_youtube_preview_renders_inside_its_tab(tmp_path):
    fake_file = tmp_path / "Clip.m4a"
    fake_file.write_bytes(b"yt bytes")
    with patch("yt_dlp.YoutubeDL") as mock_ydl_cls:
        _stub_ytdlp_class(mock_ydl_cls, fake_file, "Clip")
        at = _type_url("YouTube URL", "https://youtube.com/watch?v=slot", YOUTUBE_TAB)
    assert not at.exception
    assert len(_tab(at, YOUTUBE_TAB).get("audio")) == 1
    assert _tab(at, UPLOAD_TAB).get("audio") == []
    _assert_declared_mime(_tab(at, YOUTUBE_TAB).get("audio")[0], "audio/mp4")


@pytest.mark.parametrize(
    "active_tab,url,expected",
    [
        (URL_TAB, "https://example.com/remote.mp3", "remote.mp3"),
        (RECORD_TAB, None, "upload.mp3"),
    ],
    ids=["active_tab_wins", "empty_tab_falls_back"],
)
def test_transcribe_dispatches_from_the_active_tab(active_tab, url, expected):
    # Upload and Record declare their widgets in ungated tab bodies, so an upload
    # stays loaded across tab switches, while youtube_audio/url_audio exist only
    # while their own tab is open. Under a flat `uploaded_files or ...` chain the
    # sticky upload outranked the URL whose preview was on screen: the fetch ran,
    # the player rendered, the button enabled, and Transcribe silently ran the
    # upload. Only the first case is a mutation check -- the second pins the
    # deliberate fallback, so it passes either way: an open tab with no source of
    # its own must not leave Transcribe dead while another source is loaded.
    with (
        patch("urllib.request.urlopen") as mock_urlopen,
        patch("mlx_whisper.transcribe", return_value=MOCK_WHISPER_RESULT),
    ):
        _stub_urlopen(mock_urlopen, b"remote bytes")
        at = _run_app(active_tab=active_tab)
        # CLAUDE.md long claimed AppTest could not seed a file_uploader. It can as
        # of 1.61.1 -- FileUploader.set_value takes (name, bytes, mime), or a
        # sequence of those for accept_multiple_files=True.
        at.file_uploader[0].set_value([("upload.mp3", b"upload bytes", "audio/mpeg")])
        at.run()
        if url is not None:
            next(t for t in at.text_input if t.label == "Audio/video file URL").set_value(url)
            at.run()
        at.button[0].click().run()

    assert not at.exception
    assert [d["filename"] for d in at.session_state["transcription"]] == [expected]


def test_youtube_runtime_error_renders_an_alert():
    # _fetch_youtube_audio's 500 MB stat gate raises RuntimeError. Until it joined
    # the DownloadError branch, that already-tested guard fell through to the
    # generic handler and rendered "Unexpected error" *plus* a traceback --
    # at.exception is what separates the two.
    with patch("yt_dlp.YoutubeDL") as mock_ydl_cls:
        ydl = MagicMock()
        ydl.extract_info.side_effect = RuntimeError("YouTube audio exceeds 500 MB")
        mock_ydl_cls.return_value.__enter__.return_value = ydl
        at = _type_url("YouTube URL", "https://youtube.com/watch?v=big", YOUTUBE_TAB)

    assert not at.exception
    assert [e.value for e in at.error] == [
        r"Could not download from YouTube\: YouTube audio exceeds 500 MB"
    ]


def test_active_remote_tab_enables_transcribe():
    with patch("urllib.request.urlopen") as mock_urlopen:
        _stub_urlopen(mock_urlopen, b"file bytes")
        at = _type_url("Audio/video file URL", "https://example.com/enable.mp3", URL_TAB)
    assert at.button[0].disabled is False


def test_transcript_format_defaults_to_plain_text():
    # This pins `default=` only. The companion `required=True` is deliberately NOT
    # asserted here because AppTest cannot see it: per the docstring it stops a user
    # from deselecting the chosen option in the browser ("clicking an already-selected
    # option does nothing"), and AppTest's ButtonGroup.unselect() is a no-op for a
    # single-select group whether or not required is set -- verified by deleting
    # required=True and re-running, which changes nothing observable. So dropping
    # required=True is a silent regression as far as this suite is concerned: it would
    # let the widget return None, which reads as plain text through the
    # `== FORMAT_SUBTITLES` comparison while looking like nothing is selected.
    at = _run_app()
    assert at.segmented_control[0].value == FORMAT_PLAIN_TEXT


@pytest.mark.parametrize(
    "choice,expected",
    [(FORMAT_PLAIN_TEXT, False), (FORMAT_SUBTITLES, True)],
    ids=["plain_text", "subtitles"],
)
def test_transcript_format_drives_include_subtitles(choice, expected):
    # `include_subtitles = transcript_format == FORMAT_SUBTITLES` is a module-level
    # comparison, so the only place the mapping is observable is what
    # _handle_transcription stores. Drive the whole path: a URL source to enable
    # Transcribe, a stubbed model, then read the recorded flag back. An inverted
    # comparison would otherwise ship silently -- it changes no widget state, only
    # the download's extension and the text area's contents.
    with (
        patch("urllib.request.urlopen") as mock_urlopen,
        patch("mlx_whisper.transcribe", return_value=MOCK_WHISPER_RESULT),
    ):
        _stub_urlopen(mock_urlopen, b"file bytes")
        at = _type_url("Audio/video file URL", "https://example.com/fmt.mp3", URL_TAB)
        at.segmented_control[0].set_value(choice).run()
        at.button[0].click().run()

    assert not at.exception
    assert at.session_state["transcription"][0]["include_subtitles"] is expected
