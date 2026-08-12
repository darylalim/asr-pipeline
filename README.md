# Whisper Transcribe

[![CI](https://github.com/darylalim/whisper-transcribe/actions/workflows/ci.yml/badge.svg)](https://github.com/darylalim/whisper-transcribe/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/downloads/)

Transcribe and translate audio and video **locally on your Mac** — no cloud, no uploads, no cost. This Streamlit application is powered by OpenAI's Whisper and accelerated on Apple Silicon with MLX (Apple's machine-learning framework). Bring your own files, record straight from the browser, or paste a YouTube or media URL.

![Whisper Transcribe — the app's four input tabs (Upload, Record, YouTube, URL) above a bordered settings card holding the primary-language selector, a translate toggle, a Plain text / Subtitles transcript-format control, a no-verbatim toggle, and a collapsed Advanced options panel, with a Transcribe button below it](docs/screenshot.png)

## Features

- **[OpenAI Whisper large-v3-turbo](https://huggingface.co/mlx-community/whisper-large-v3-turbo)** via [mlx-whisper](https://pypi.org/project/mlx-whisper/), accelerated on Apple Silicon
- **On-device processing** — audio is transcribed entirely on your machine; nothing is uploaded (only the YouTube and URL input modes — plus the one-time model-weights download on first run — use the network)
- **100-language transcription** with auto-detect or manual selection
- **Translate non-English audio to English**
- **Four input modes** — multi-file upload (up to 500 MB per file), in-browser recording, YouTube links, and direct audio/video file URLs
- **Editable subtitle preview**, exportable as SRT (the standard subtitle file format)
- **No verbatim** — skips hallucinated text over music, applause, and other non-speech audio
- **Decode segments independently** — more robust on noisy or music-heavy audio
- **Time-range clipping** — transcribe only selected portions (comma-separated `start,end` pairs in seconds)
- **Keyterms** — bias decoding toward proper nouns and jargon (up to 50 terms)
- **Instant repeat results** — identical file-and-settings combinations are served from cache
- **Light and dark theme** with Material Symbol icons, switchable in the app's settings menu

## How it works

You provide audio or video through one of four tabs (upload, record, YouTube, or URL). The app writes the audio to a temporary file and runs `mlx_whisper.transcribe()` with the Whisper large-v3-turbo model locally on Apple Silicon via MLX. The result is cached, rendered as editable plain text (or SRT when subtitles are enabled), and can be downloaded as `.txt` or `.srt`. See [CLAUDE.md](CLAUDE.md) for the full architecture.

![A completed transcription in Whisper Transcribe — a "Transcribed 1/1 file" status above a bordered result section holding the filename, the editable transcript, and a Download button to save it as .txt or .srt](docs/screenshot-result.png)

## Requirements

- macOS on Apple Silicon (M1 / M2 / M3 / M4)
- Python 3.12+
- [FFmpeg](https://formulae.brew.sh/formula/ffmpeg)
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
git clone https://github.com/darylalim/whisper-transcribe.git
cd whisper-transcribe
brew install ffmpeg
uv sync
```

## Usage

```bash
uv run streamlit run streamlit_app.py
```

Upload one or more files (audio: `mp3, m4a, wav, opus`; video: `mp4, mov, webm, mkv`), record audio in-browser, paste a YouTube URL, or paste an audio/video file URL, then click **Transcribe**.

> **First run:** the first time you transcribe, the Whisper large-v3-turbo weights (~1.5 GB) are downloaded from Hugging Face and cached locally, so the first transcription takes longer and needs an internet connection. Subsequent transcriptions run offline.

Optional controls:

- **Primary language** — auto-detected by default
- **Translate to English** — translate non-English audio
- **Transcript format** — choose **Plain text** or **Subtitles**; picking Subtitles shows an editable SRT subtitle preview and switches the **Download** button from `.txt` to `.srt`. Subtitle cues are wrapped to 42 characters per line, the broadcast convention, with the lines balanced rather than one filled and one left short
- **No verbatim** — skip silent stretches where Whisper appears to be hallucinating text, such as over music or applause after speech ends; it does not remove filler words or repetitions
- **Decode segments independently** — disable prior-window context; more robust on noisy or music-heavy audio, at the cost of slightly choppier wording where 30 s windows meet
- **Time range** — transcribe only selected portions; comma-separated `start,end` pairs in seconds (e.g., `30,90` for one clip, `0,60,120,180` for multiple); invalid ranges are flagged inline
- **Keyterms** — bias decoding toward specific terms (proper nouns, jargon)

**Decode segments independently**, **Time range**, and **Keyterms** are grouped under an **Advanced options** expander, inside the same settings card as the controls above.

## Development

After `uv sync`, the project's checks run through uv:

```bash
uv run pytest             # run the test suite
uv run ruff check .       # lint
uv run ruff format .      # format
uv run ty check           # type-check
```

CI runs the same tools on every push to `main` and on pull requests — it uses `ruff format --check .` to *verify* formatting rather than apply it, so run `ruff format .` locally before committing. The workflow targets a **macos-14 (Apple Silicon) runner** because of the compute backend, not wheel availability: `mlx` pulls in a backend automatically only on macOS (`mlx-metal`), whereas on Linux it is an opt-in extra — so a Linux runner installs cleanly and then fails at `import mlx_whisper`. Please make sure `ruff`, `ty`, and `pytest` pass before opening a pull request.

## Troubleshooting

- **`ffmpeg` not found** — install it with `brew install ffmpeg` and verify with `ffmpeg -version`. FFmpeg is required for decoding audio and video.
- **Intel Mac / non–Apple Silicon** — `mlx-whisper` requires Apple Silicon and will not run on Intel Macs.
- **Long pause on the first transcription** — the model weights (~1.5 GB) are downloading from Hugging Face (see the *First run* note above); this needs a network connection and only happens once.
- **YouTube download fails** — this usually means `yt-dlp` is out of date; update it with `uv lock --upgrade-package yt-dlp && uv sync`.
- **Remote download rejected** — both remote fetches are capped at 500 MB: direct audio/video URLs, and YouTube audio (a multi-hour livestream VOD can exceed it).
- **Download is missing your last edit** — the transcript box commits when it loses focus, so click outside it (or press Ctrl/Cmd+Enter) before pressing **Download**.

## License

This project is licensed under the [MIT License](LICENSE).
