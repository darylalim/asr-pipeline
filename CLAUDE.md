# CLAUDE.md

Streamlit application for transcription and translation using OpenAI Whisper on Apple Silicon with MLX.

## Setup

```bash
uv sync
uv run streamlit run streamlit_app.py
```

## Commands

- **Lint**: `uv run ruff check .`
- **Format**: `uv run ruff format .`
- **Typecheck**: `uv run ty check`
- **Test**: `uv run pytest`

When working with Python, invoke the relevant `/astral:<skill>` for uv, ty, and ruff to ensure best practices are followed.

When changing the Streamlit UI (tabs, widgets, theme, layout, caching, fragments), invoke the `developing-with-streamlit` skill to stay version-correct against the locked Streamlit version (currently 1.61.1; the floor stays `>=1.58`).

## Automation

- **Hooks** (`.claude/settings.json`, checked in): a `PostToolUse` hook runs `ruff format` + `ruff check --fix` on edited `*.py` files; a `Stop` hook gates on `ty check` + `pytest`, feeding failures back for repair (guarded against re-engage loops via `stop_hook_active`). Personal overrides live in the gitignored `.claude/settings.local.json`
- **CI** (`.github/workflows/ci.yml`): runs `ruff check` + `ruff format --check` + `ty check` + `pytest` on push to `main` and on PRs. Pinned to a **`macos-14` (Apple Silicon) runner** — required because `mlx-whisper` ships no Linux wheels, so `uv sync` can't resolve on Linux. Uses `uv sync --locked` and a SHA-pinned `astral-sh/setup-uv` with caching

## Code Style

- snake_case for functions/variables, PascalCase for classes
- Type annotations on all parameters and returns
- `RuntimeError` for known transcription failures (no custom exception class)
- Import sorting via ruff with combine-as-imports

## Dependencies

- `mlx-whisper` — speech recognition on Apple Silicon
- `streamlit` — web UI (floor `>=1.58`, the minimum the app's API usage requires; developed and tested against the locked 1.61.1)
- `yt-dlp` — YouTube audio download
- `ffmpeg` — audio/video decoding (system dependency)
- `ruff` — linting and formatting (dev)
- `ty` — type checking (dev)
- `pytest` — testing (dev)
- `watchdog` — event-based file watching for Streamlit's auto-reload (dev). Streamlit already
  depends on it, but declares the requirement as `platform_system != "Darwin"` — so on Apple
  Silicon it is *not* installed transitively, and `streamlit run` falls back to
  `PollingPathWatcher` (the source of the "For better performance, install the Watchdog module"
  notice). Declaring it here restores the `FSEventsObserver` backend. Dev-only because it affects
  the edit-reload loop, not transcription; the app is fully functional without it

## Architecture

- `streamlit_app.py` — single-file app entry point
- `tests/test_app.py` — unit tests

### Model

Direct `mlx_whisper.transcribe()` call with `ASR_MODEL_REPO = "mlx-community/whisper-large-v3-turbo"`. MLX accelerates natively on Apple Silicon. Results are cached via `@st.cache_data(show_spinner=False, max_entries=20)`; the spinner is disabled because per-file progress is rendered by the `st.status` wrapper in `_handle_transcription`.

Model weights are **not** wrapped in `st.cache_resource`. `mlx_whisper` already holds them on a process-level class attribute (`ModelHolder.get_model` in `mlx_whisper/transcribe.py`), so the repo is loaded once per server process; adding `st.cache_resource` would layer a second cache over an existing one.

### Fetch caching

`_fetch_youtube_audio` and `_fetch_url_audio` are `@st.cache_data(max_entries=5, ttl="1h")`. These return **raw audio bytes**, so the two caches together are the app's dominant memory cost: 2 caches × 5 entries, and only the URL path is size-capped (`MAX_URL_DOWNLOAD_BYTES` = 500 MB), putting the bounded side alone at ~2.5 GB and the pair well past that. `max_entries` alone never evicts an entry that nothing displaces, which is what `ttl` addresses.

Be precise about what `ttl` does, though: Streamlit's `ttl_cache.py` expires **lazily** — expired entries read as absent, but are "only actually removed by a write, by `expire()`, or by a length/size query." So `ttl` bounds staleness and lets a later fetch reclaim the slot; it is *not* a background reaper, and a fully idle server can still hold expired payloads until something touches the cache. It shrinks the window, not the worst case.

**Known gap:** `_fetch_youtube_audio` has no size cap at all — `downloaded.read_bytes()` (line 73) slurps whatever `yt-dlp` wrote. A multi-hour livestream VOD is loaded whole into one `bytes` object and cached. The URL path's cap does not apply here; capping this would mean `yt-dlp`'s `max_filesize` option, since the download has already happened by the time the bytes are read.

`_transcribe` needs no `ttl` for the same reason it needs no size bound beyond `max_entries=20` — it caches result dicts (text + segments), not media.

### Input Modes

- `st.set_page_config(**PAGE_CONFIG)` is the first Streamlit call (before `st.title`); `PAGE_CONFIG` is a module constant (browser tab title `Whisper Transcribe`, a `:material/graphic_eq:` page icon, `centered` layout) so its values stay unit-testable
- **Upload** / **Record** / **YouTube** / **URL** tabs (`st.tabs`, each label prefixed with a Material Symbol icon — `upload`, `mic`, `smart_display`, `link`) — Upload accepts multiple files (`accept_multiple_files=True`) with one `st.audio` preview per file; the uploader's `type` is `AUDIO_FORMATS + VIDEO_FORMATS`, which Streamlit auto-lists in the dropzone (no hand-maintained duplicate); Record takes a single recording with its own preview; YouTube takes a URL (gated by a `youtube.com` / `youtu.be` regex and stripped of whitespace), downloads the best audio stream via `yt-dlp` (cached — see **Fetch caching** below — plus `restrictfilenames=True`, `noplaylist=True`), and shows an `st.audio` preview of the bytes; URL takes any `http(s)` audio/video file URL (gated by an `https?://` regex; YouTube URLs short-circuit to an `st.info` redirecting to the YouTube tab), downloads via `urllib.request.urlopen` with a 60-second timeout and a 500 MB cap (`MAX_URL_DOWNLOAD_BYTES`, cached), derives the filename from the URL path (percent-decoded, fallback `download`), and shows an `st.audio` preview
- Below the tabs, controls are grouped by intent (input → output → advanced) for visual hierarchy. Always-visible, in order: **Primary language** selector (input), then the output group — **Translate to English** toggle, **Include subtitles** toggle, **No verbatim** toggle (enables `word_timestamps=True` + `hallucination_silence_threshold=2.0` to skip hallucinations on non-speech audio like music outros). The three power-user controls live in a collapsed **`st.expander("Advanced options", icon=":material/tune:")`** (progressive disclosure — defaults still apply when closed; **do not** add `on_change="rerun"` + `if expander.open:` lazy gating here, see below): **Decode segments independently** toggle (sets `condition_on_previous_text=False` so each 30 s window decodes without prior-window context — robust on noisy audio at the cost of cross-boundary fluency), **Time range** text input (forwarded as `clip_timestamps`; comma-separated `start,end` pairs in seconds, e.g. `30,90` or `0,60,120,180`; blank → `"0"` for the full file; validated by `_validate_time_range` — malformed input disables the Transcribe button until corrected, with the `st.error` rendered outside the Advanced options expander (above the button) so the disabled reason stays visible even when the expander holding the input is collapsed), and **Keyterms** chip input (`st.multiselect` with `accept_new_options=True`, max 50 chips, joined with `, ` and forwarded as `initial_prompt`). Below everything, a right-aligned, full-width (`width="stretch"`) **Transcribe** button with a `:material/graphic_eq:` icon. `Include subtitles` stays in the always-visible output group (never the expander) because it has a user-visible side effect — it flips the download between `.srt` and `.txt`
- The Transcribe button dispatches in priority order: uploaded files → recording → YouTube audio → URL audio. Each non-upload source is wrapped in a single-element list. YouTube and URL sources share a `_RemoteAudio` adapter exposing `.name` (a safe filename, including extension when available) and `.read()` so they flow through `_handle_transcription` without changes. UI flags are routed through `_transcription_kwargs`, which centralizes the `translate → task` mapping and the `decode_independently → condition_on_previous_text` inversion so a script-level negation can't silently disappear. Subtitles controls both the text area's initial content (SRT-formatted segments when on, plain text when off) and the format the **Download** button serves (`.srt` vs `.txt`); the text area is always editable
- `_transcribe` writes audio bytes to a temp file, calls `mlx_whisper.transcribe()`, and caches results (`language=None` → Whisper auto-detects)
- `_handle_transcription` wraps the batch in `st.status(...)` (label updates to `Transcribing {name} ({i}/{total})...` per file, transitions to `complete` at the end), transcribes each upload, and stores the resulting list of `{result, file_stem, filename, include_subtitles}` dicts in `st.session_state["transcription"]`. The list is published **before** the loop and re-published after each file, never only at the end: Streamlit interrupts a running script at the next ForwardMsg (`status.update()` is one) by raising `RerunException`, a `BaseException` that the `except Exception` handler deliberately does not catch. A single post-loop assignment would therefore throw away every file already transcribed whenever any rerun lands mid-batch — a tab switch, a toggle, any widget. The pre-loop publish additionally clears a previous batch when nothing in this one succeeds. The `file_stem` includes the source extension (e.g., `interview_mp3_transcript`) to disambiguate downloads when two uploads share a stem. Per-file errors are reported inline via `st.error` and don't stop the rest of the batch
- `_display_transcription` renders one stacked section per stored result: `st.subheader(filename)` (the filename is Markdown-escaped via `_escape_markdown`, since `st.subheader` renders the Markdown label subset — otherwise `_`, `*`, brackets, or `:` directives in names like YouTube titles would mis-render) + an editable text area (plain text or SRT segments per `include_subtitles`) + a right-aligned, full-width (`width="stretch"`) **Download** button with a `:material/download:` icon that captures the text area's edited content. The button sits in the right column of an `st.columns([3, 1])` split — same ratio as the Transcribe button — so the two share an edge and width. The label is always `Download` regardless of format; the file extension (`.srt` when subtitles are on, `.txt` otherwise) is set via the filename + MIME type args, and a `help=` tooltip (`"Downloads as .srt when subtitles are enabled, .txt otherwise."`) explains the format switch on hover. Indexed widget keys (`transcript_{i}`, `download_{txt,srt}_{i}`) avoid collisions
- `_display_transcription` is a plain function but is invoked through `st.fragment(_display_transcription)()` at the call site (not a `@st.fragment` decorator) so transcript edits and downloads rerun only this section instead of the whole script (which would otherwise re-evaluate all four input tabs). Wrapping at the call site rather than decorating keeps the function directly unit-testable — the real `st.fragment` wrapper returns `None` without running the body when there is no script-run context (bare test mode)

### Lazy Container Execution

`st.tabs` and `st.expander` can skip computing hidden content via `on_change="rerun"` plus the container's `.open` property, and the general best-practices guidance recommends it for expensive work behind tabs and expanders. This API is present at the declared `streamlit>=1.58` floor (verified against a 1.58.0 install: `st.tabs` already accepts `key` and `on_change`, and `TabContainer.open` exists), so using it does not require raising the floor. **That guidance does not apply to the Advanced options expander.** Its three widgets are not expensive work — they are widget *declarations* whose return values (`decode_independently`, `time_range_input`, `keyterms`) are consumed unconditionally further down the script. Gating the expander body would leave those names undefined whenever the expander is collapsed, i.e. by default. Widget declarations are cheap; their values are load-bearing.

The tabs **are** gated, but at a finer granularity than wrapping the whole body. `st.tabs(..., on_change="rerun", key="input_tabs")` enables each tab's `.open` flag, and the YouTube and URL tabs guard only their **fetch** on it (`if youtube_tab.open and ...`), never the `st.text_input` that feeds it.

That split is load-bearing. Wrapping the whole `with` body in `if tab.open:` would stop rendering the text input while the tab is hidden, and Streamlit removes state for widgets absent from a run (`SessionState._remove_stale_widgets`) — so the typed URL would be silently cleared on every tab switch. Widget declarations are cheap and must always run; only the side effect gets gated.

Note that `on_change="rerun"` does not itself skip anything: `TabContainer.__enter__`/`__exit__` are plain passthroughs (Python cannot skip a `with` body), so the flag only enables `.open` tracking and makes tab switches trigger a rerun. Every `if tab.open:` guard is hand-written.

Three accepted consequences. Switching tabs now costs a server rerun. `youtube_audio` / `url_audio` are `None` while another tab is active, so a pasted URL alone doesn't enable **Transcribe** unless its tab is showing (upload and record sources stay sticky across tabs, so the button's enabled state is source-dependent). And because a tab switch is now a rerun, it can **interrupt an in-flight transcription** — previously tab switching was purely client-side and could not; this is why `_handle_transcription` publishes results progressively rather than once at the end. `key="input_tabs"` holds the active tab's *label* in session state; `AppTest` has no tab-selection API, so the tests drive tab switches by writing that key.

### Audio Formats

Audio: aac, aiff, ogg, mp3, opus, wav, flac, m4a

Video: mp4, avi, mkv, mov, wmv, flv, webm, mpeg, 3gpp

### Upload Limit

- Per-file cap of **500 MB**, set in `.streamlit/config.toml` via `server.maxUploadSize`
- Enforced browser-side by Streamlit; with `accept_multiple_files=True` the cap is per-file, not aggregate
- Server restart required after changing this setting

### Theme

- Streamlit's built-in light and dark themes — `.streamlit/config.toml` carries **no `[theme]` section at all**, so the app inherits the stock palette and typography
- Deliberately unconfigured rather than re-specified: every `theme.*` option defaults to `None` (the stock look lives in the frontend, not in config defaults), so writing out the default hex values would pin the app to one Streamlit version's styling instead of tracking upstream restyles
- The light/dark switcher in the app settings menu stays available. The "single-mode `[theme]` locks the app to one mode" hazard applies only to *custom* themes; with no theme configured, Light / Dark / system setting are all offered
- Native theming only (no custom CSS/HTML). If a custom theme is reintroduced, define both `[theme.light]` and `[theme.dark]` to keep the switcher, and note that font changes require a server restart

### Error Handling

- `RuntimeError` caught explicitly for transcription failures
- Unexpected exceptions shown with `st.exception()`

### Testing

Mocked at the boundary (`mlx_whisper`, `yt_dlp`, `urlopen`, `st`). Shared fixtures (`mock_mlx`, `mock_st`, `mock_uploaded_file`) and helpers (`_make_file`, `_stub_urlopen`, `_stub_ytdlp`, `_make_transcription`, `_expected_transcribe_kwargs`, `_handle_transcription_kwargs`, `_ui_state`) factor the common setup; kwarg-forwarding cases are `@pytest.mark.parametrize`d. An autouse `_clear_caches` fixture clears the `@st.cache_data` wrappers before each test so cached results don't leak between cases. It also calls **`st.cache_data.clear()`**, which is not redundant: the per-wrapper `.clear()` calls only reach caches created by the *imported* `streamlit_app` module, while `AppTest` re-executes the script as a separate module with its own cache store that otherwise survives the whole pytest session. Without the global clear, a second test reusing a URL gets a stale hit — which would let a "fetch was skipped" assertion pass even with the gate removed.

- `_transcribe` — defaults, kwarg forwarding (language, task, initial_prompt, no_verbatim, condition_on_previous_text, clip_timestamps), temp-file cleanup, empty-text guard; `no_verbatim=True` flips `word_timestamps` and `hallucination_silence_threshold`; `clip_timestamps` defaults to `"0"` (full file) and accepts custom ranges (e.g., `"30,90"`)
- `_handle_transcription` — session-state storage, per-file error handling (RuntimeError + unexpected), kwarg forwarding, multi-file batches, partial-failure scenarios; progressive publishing is pinned by two cases — a `BaseException` raised mid-batch (standing in for `RerunException`) leaves the already-finished files in session state, and a batch where every file fails clears the previous batch's results
- `_transcription_kwargs` — UI-flag → `_handle_transcription` kwargs mapping; `translate=True` ↔ `task="translate"`, `decode_independently=True` ↔ `condition_on_previous_text=False`; passthrough of `language`, `include_subtitles`, `initial_prompt`, `no_verbatim`, `clip_timestamps`
- `_display_transcription` — filename subheader, editable text area, right-aligned **Download** button in `st.columns([3, 1])`; label is always `Download` with a `:material/download:` icon and `width="stretch"`; filename, MIME type, and widget key are derived from `include_subtitles` (`.txt` vs `.srt`); `help=` tooltip preserved in both `.txt` and `.srt` paths; edits to the text area flow through to the download payload; multi-file stacked rendering with indexed keys; filenames with Markdown-special characters are escaped in the subheader. Tested as a plain function (the `st.fragment` wrap is applied only at the call site, so the body runs directly under the mocked `st`)
- `_RemoteAudio` / `_fetch_youtube_audio` / `_fetch_url_audio` — adapter round-trip; YouTube fetch with mocked `yt_dlp` (bytes + filename, `extract_info` call args, safe-download options `format=bestaudio/best`, `noplaylist`, `restrictfilenames`, `quiet`); URL fetch with mocked `urlopen` (bytes + filename, `timeout=60`, query-string stripping, percent-decoded filename, empty-path fallback to `download`, oversized-response `RuntimeError`)
- Formatting helpers — `_format_language` (`None` → `"Detect"`, title-casing of lowercase codes), `_format_timestamp` (zero, minutes/seconds, hours, comma decimal marker), `_format_srt` (single-segment, multi-segment cue separator, `-->` escaping to `->` to keep SRT structure intact), `_escape_markdown` (plain passthrough, underscores, brackets, all special chars, backslash), `_validate_time_range` (blank/single/multi/whitespace/decimal/trailing-start valid cases → `None`; non-numeric, negative, end ≤ start, out-of-order, trailing/empty token → error message). Odd counts are valid — a trailing unpaired value is a start that runs to the end of the file, matching Whisper's `clip_timestamps`
- Tab-gated fetches (`AppTest`) — both remote tabs are parametrized over hidden/active: a URL typed while another tab is active performs no fetch, the same URL with its own tab active fetches exactly once, and the typed value is retained either way (proving the widget is not gated, only the fetch). A third case asserts an active remote tab enables **Transcribe**. The active tab is set by writing `session_state["input_tabs"]` (AppTest can't click tabs), and the network entry points are patched on `urllib.request` / `yt_dlp` rather than on `streamlit_app`, because AppTest re-executes the script each run and rebinds its imports. Both gates are mutation-checked: deleting either `tab.open` guard fails the corresponding hidden-tab case
- Module UI (`streamlit.testing.v1.AppTest`) — runs the real script (not mocked `st`, loaded via an absolute path so it's cwd-independent) to cover the module-level UI the mocked tests can't reach: clean render + page title, the four Material-icon tab labels, the **Transcribe** button's `:material/graphic_eq:` icon and disabled-without-audio state, the time-range input's inline `st.error` on invalid input (and no error on valid input, asserted end-to-end through the rendered UI), and (with `st.session_state["transcription"]` seeded) the fragment-rendered subheader + text area + **Download** button carrying its `:material/download:` icon; the empty-state case renders no download button. A plain unit test asserts the `PAGE_CONFIG` constant (`set_page_config` args aren't introspectable via AppTest)

## Resources

- [mlx-whisper](https://pypi.org/project/mlx-whisper/)

## License

MIT (see `LICENSE`); declared via `license = "MIT"` / `license-files = ["LICENSE"]` in `pyproject.toml`.
