from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from streamlit_app import (
    ASR_MODEL_REPO,
    AUDIO_FORMATS,
    PAGE_CONFIG,
    VIDEO_FORMATS,
    _display_transcription,
    _escape_markdown,
    _fetch_url_audio,
    _fetch_youtube_audio,
    _format_language,
    _format_srt,
    _format_timestamp,
    _handle_transcription,
    _RemoteAudio,
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


# --- Helpers ---


def _make_file(name="interview.mp3", data=b"fake audio bytes"):
    f = MagicMock()
    f.name = name
    f.read.return_value = data
    return f


def _stub_urlopen(mock_urlopen, data):
    response = MagicMock()
    response.read.return_value = data
    mock_urlopen.return_value.__enter__.return_value = response
    return response


def _stub_ytdlp(mock_yt_dlp, path, title="Test"):
    ydl = MagicMock()
    ydl.extract_info.return_value = {"title": title}
    ydl.prepare_filename.return_value = str(path)
    mock_yt_dlp.YoutubeDL.return_value.__enter__.return_value = ydl
    return ydl


def _make_transcription(
    include_subtitles=False, file_stem="interview_transcript", filename="interview.mp3"
):
    return {
        "result": MOCK_WHISPER_RESULT,
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
    # cache_data store, which would otherwise persist for the whole session
    # and let a fetch-was-skipped assertion pass on a stale cache hit.
    st.cache_data.clear()


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

    data, filename = _fetch_youtube_audio("https://youtube.com/watch?v=fetch_bytes")

    assert data == b"fake youtube audio"
    assert filename == "Test_Video.m4a"
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

    data, filename = _fetch_url_audio("https://example.com/audio.mp3")

    assert data == b"file bytes"
    assert filename == "audio.mp3"
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
    _, filename = _fetch_url_audio(url)
    assert filename == expected_filename


@patch("streamlit_app.MAX_URL_DOWNLOAD_BYTES", 10)
@patch("streamlit_app.urlopen")
def test_fetch_url_audio_rejects_oversized_response(mock_urlopen):
    response = _stub_urlopen(mock_urlopen, b"x" * 11)

    with pytest.raises(RuntimeError, match="exceeds"):
        _fetch_url_audio("https://example.com/too-big.mp3")

    response.read.assert_called_once_with(11)


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
        ({"clip_timestamps": "0,60,120,180"}, {"clip_timestamps": "0,60,120,180"}),
    ],
    ids=["translate", "initial_prompt", "no_verbatim", "no_context", "single_clip", "multi_clip"],
)
def test_transcribe_forwards_kwargs(mock_mlx, call_kwargs, expected):
    _transcribe(b"audio", ".mp3", **call_kwargs)
    kwargs = mock_mlx.transcribe.call_args.kwargs
    assert {k: kwargs[k] for k in expected} == expected


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
    mock_st.error.assert_called_once_with(
        "Transcription failed for interview.mp3: Transcription produced no text"
    )
    assert mock_st.session_state["transcription"] == []


@patch("streamlit_app._transcribe", side_effect=ValueError("unexpected"))
def test_handle_transcription_unexpected_error(mock_transcribe, mock_st, mock_uploaded_file):
    _handle_transcription(
        [mock_uploaded_file], language=None, task="transcribe", include_subtitles=False
    )
    mock_st.error.assert_called_once_with("Unexpected error for interview.mp3: unexpected")
    mock_st.exception.assert_called_once()


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
        "Transcription failed for second.mp3: Transcription produced no text"
    )


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


@patch("streamlit_app._transcribe")
def test_handle_transcription_clears_previous_batch(mock_transcribe, mock_st):
    mock_st.session_state["transcription"] = [_make_transcription(filename="stale.mp3")]
    mock_transcribe.side_effect = RuntimeError("Transcription produced no text")

    _handle_transcription([_make_file()], language=None, task="transcribe", include_subtitles=False)

    assert mock_st.session_state["transcription"] == []


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
        key="transcript_0",
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
        key="download_txt_0",
        help="Downloads as .srt when subtitles are enabled, .txt otherwise.",
        width="stretch",
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
        key="download_srt_0",
        help="Downloads as .srt when subtitles are enabled, .txt otherwise.",
        width="stretch",
    )


def test_display_transcription_subtitles_on(mock_st):
    mock_st.session_state["transcription"] = [_make_transcription(include_subtitles=True)]

    _display_transcription()

    mock_st.text_area.assert_called_once_with(
        "Transcript",
        SRT_HELLO,
        height=300,
        label_visibility="collapsed",
        key="transcript_0",
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
        key="download_txt_0",
        help="Downloads as .srt when subtitles are enabled, .txt otherwise.",
        width="stretch",
    )


def test_display_transcription_right_aligns_download(mock_st):
    mock_st.session_state["transcription"] = [_make_transcription()]
    _display_transcription()
    mock_st.columns.assert_called_once_with([3, 1])


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


# --- formatting helpers ---


@pytest.mark.parametrize(
    "code,expected",
    [(None, "Detect"), ("en", "English"), ("fr", "French")],
    ids=["none_returns_detect", "lowercase_code", "title_cased"],
)
def test_format_language(code, expected):
    assert _format_language(code) == expected


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
    "raw,expected",
    [
        ("interview.mp3", "interview.mp3"),
        ("interview_part_1.mp3", r"interview\_part\_1.mp3"),
        ("Song [Official Video].mp3", r"Song \[Official Video\].mp3"),
        ("a*b`c~d:e$f", r"a\*b\`c\~d\:e\$f"),
        (r"back\slash", r"back\\slash"),
    ],
    ids=["plain", "underscores", "brackets", "all_specials", "backslash"],
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
    ["abc", "30,abc", "-5,10", "90,30", "30,30", "30,", ",30", "60,90,0,30"],
    ids=[
        "non_numeric",
        "non_numeric_pair",
        "negative",
        "end_before_start",
        "equal_pair",
        "trailing_comma",
        "leading_comma",
        "out_of_order",
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


# The remote-fetch tabs gate their download on `tab.open` because st.tabs
# computes hidden tab bodies by default. Network entry points are patched on
# their own modules (urllib.request / yt_dlp) rather than on streamlit_app,
# because AppTest re-executes the script each run and rebinds its imports.

UPLOAD_TAB = ":material/upload: Upload"
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
        ydl = MagicMock()
        ydl.extract_info.return_value = {"title": "Clip"}
        ydl.prepare_filename.return_value = str(fake_file)
        mock_ydl_cls.return_value.__enter__.return_value = ydl
        at = _type_url("YouTube URL", url, active_tab)
    assert not at.exception
    assert ydl.extract_info.call_count == expected_calls
    assert _typed_value(at, "YouTube URL") == url


def test_active_remote_tab_enables_transcribe():
    with patch("urllib.request.urlopen") as mock_urlopen:
        _stub_urlopen(mock_urlopen, b"file bytes")
        at = _type_url("Audio/video file URL", "https://example.com/enable.mp3", URL_TAB)
    assert at.button[0].disabled is False
