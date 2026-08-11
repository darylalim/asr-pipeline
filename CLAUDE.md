# CLAUDE.md

Streamlit application for transcription and translation using OpenAI Whisper on Apple Silicon with MLX.

## Setup

```bash
uv sync
uv run streamlit run streamlit_app.py
```

Python floor is **3.12** (`requires-python`), and `[tool.ty.environment] python-version = "3.12"` pins `ty` to it — so a 3.13-only stdlib symbol runs fine in the local 3.13 `.venv`, passes `pytest`, and fails `ty check` with an error pointing at `pyproject.toml`. There is no `.python-version`, so a fresh `uv sync` takes uv's default interpreter — locally that means 3.13. CI pins `UV_PYTHON: "3.12"` instead, so the floor is exercised there and only there.

## Commands

- **Lint**: `uv run ruff check .`
- **Format**: `uv run ruff format .`
- **Typecheck**: `uv run ty check`
- **Test**: `uv run pytest`

When working with Python, invoke the relevant `/astral:<skill>` for uv, ty, and ruff to ensure best practices are followed.

When changing the Streamlit UI (tabs, widgets, theme, layout, caching, fragments), invoke the `developing-with-streamlit` skill to stay version-correct against the locked Streamlit version (currently 1.61.1; the floor stays `>=1.58`).

## Automation

- **Hooks** (`.claude/settings.json`, checked in): a `PostToolUse` hook runs `ruff format` + `ruff check --fix` on edited `*.py` files and `ruff format` alone on edited `*.md` files (`ruff check` is a no-op on Markdown — it reports "No Python files found" and exits 0, so only the formatter is worth running there) — both branches sit behind a `case "$f" in "$PWD"/*)` containment guard, since the hook fires on *any* edit Claude makes and `cd "$CLAUDE_PROJECT_DIR"` pins only ruff's **config** discovery, not its **target**: without the guard an edit to `~/.claude/**` or a scratchpad file is reformatted with this project's `line-length = 100`; a `Stop` hook gates on `uv lock --check` + `ruff check` + `ruff format --check` + `ty check` + `pytest`, feeding failures back for repair (guarded against re-engage loops via `stop_hook_active`, and entered via `cd "${CLAUDE_PROJECT_DIR:-.}"` so an unset variable falls back to the cwd rather than `cd ""`-ing into a silent `exit 0` that skips the whole gate). Personal overrides live in the gitignored `.claude/settings.local.json`
- **The Stop hook deliberately duplicates the `PostToolUse` hook's `ruff` work.** `PostToolUse` ends in `> /dev/null 2>&1 || true`, so if `uv run ruff` ever breaks (venv rebuilt, `$CLAUDE_PROJECT_DIR` unset, or a path that the containment guard rejects) it silently stops formatting with no signal. Running `ruff` in the `Stop` gate too costs ~50 ms (measured: 3.21 s for the full gate vs. 3.20 s for `ty` + `pytest` alone) and converts that silent failure into a blocking one. Do not "de-duplicate" it
- The two `ruff` steps in the `Stop` hook take `-q`; `ty check` deliberately does **not**. `ruff -q` suppresses only the success chatter and still prints full diagnostics on failure, but `ty check -q` collapses a real type error to the bare line `Found 1 diagnostic` — which, as the `reason` fed back for repair, names neither the file nor the problem
- **CI** (`.github/workflows/ci.yml`): runs `ruff check` + `ruff format --check` + `ty check` + `pytest` on push to `main` and on PRs. Pinned to a **`macos-14` (Apple Silicon) runner**, and the job sets `UV_PYTHON: "3.12"` — the only place the declared floor is actually exercised, since the local `.venv` runs 3.13. There is no `uv python install` step: `uv sync --locked` provisions the interpreter, and a bare `uv python install` would fetch uv's newest default instead of the pin. `enable-cache: true` on `setup-uv` is load-bearing rather than cosmetic — `mlx-whisper` depends on **`torch`**, 504 MB installed, so a cold cache dominates the run
- **Every action is pinned to a full commit SHA**, with the version in a trailing comment (`actions/checkout@3d3c42e… # v7.0.1`). Tags are mutable, so a `@v7` that is force-moved or compromised runs arbitrary code holding the workflow's token — and the `release` job below carries `contents: write`. Use a SHA for any action added later, and bump the trailing comment with it
- **Dependabot** (`.github/dependabot.yml`) keeps those pins from going stale — a frozen SHA never surfaces the security fix it is missing. Weekly `github-actions` updates only (**not** `uv`/Python deps), grouped into a single PR and prefixed `ci:` to match the commit convention. It rewrites the SHA *and* its trailing `# vX.Y.Z` comment together, which is the property that makes SHA pinning maintainable. `directory: "/"` is required for this ecosystem — Dependabot scans `.github/workflows` on its own and does not take that path. Version updates need no repo setting; the config file's presence on `main` is what enables them
- **`macos-14` is required by the backend, not by wheel availability** — the older "`mlx-whisper` ships no Linux wheels" rationale is obsolete. `mlx_whisper` is a pure-Python `py3-none-any` wheel and `mlx` ships `manylinux_2_35` wheels that are already in `uv.lock`, so `uv sync --locked` resolves *and installs* fine on Linux. But `mlx` depends on its compute backend unconditionally only on Darwin (`mlx-metal`); on Linux the backend is opt-in (`mlx-cpu` / `mlx-cuda-*`, gated behind the `cpu` / `cuda` extras) and the lockfile carries neither. A Linux runner therefore goes green through `uv sync` and dies at `import mlx_whisper` several steps later. Switching to `ubuntu-latest` is possible by adding `mlx[cpu]`, but it would test a backend the app never ships on
- **Release** is the `release` job *inside* `ci.yml` (there is no separate release workflow; `draft-release.yml` was merged in). It reads `project.version`, then tags the commit and **publishes** `v<version>` with auto-generated notes. Runs on plain `ubuntu-latest` (unlike `check`) because it never calls `uv sync`, so the Darwin-only `mlx` backend never has to install
- ⚠️ **Bumping `version` in `pyproject.toml` publishes a public GitHub Release** — it tags `main`, marks the release Latest, and notifies watchers. A publish is not cleanly reversible: deleting it leaves anyone who already fetched the tag on a version that no longer exists. An incidental bump in an unrelated `pyproject.toml` edit therefore ships a release, so treat `version` as the release trigger it is. To stage one for review instead, drop `--draft=false` from the `gh release edit` line and publish by hand
- The release job's guard is the **release's existence**, not a `paths:` filter — it runs on every push to `main` and no-ops when `v<version>` is already published. A `paths:` filter misses a bump that arrives inside a merge commit and cannot be re-run after a failure; the existence check is idempotent and re-runnable. It resolves three states: `published` → exit, `draft` → publish it (a previous run died between the two `gh` calls; skipping would strand that draft forever), `missing` → create the draft, then publish. Anything else fails loudly rather than publishing a release whose state is unknown
- **`needs: check` is the point of the merge.** As a standalone workflow the release ran independently of CI, so a version bump drafted a release even when `ruff`/`ty`/`pytest` were red. Nothing may publish ahead of a green `check`
- Two details in the release job that look incidental and are not: `permissions: contents: write` is set **per job**, so the write scope never reaches the `check` job that actually runs third-party code; and `cancel-in-progress` is `${{ github.event_name == 'pull_request' }}` rather than plain `true`, because a blanket cancel can kill a `main` run mid-publish, between creating the draft and flipping it live
- **Creating the release as a draft first is load-bearing, not ceremony.** GitHub withholds the git tag while a release is a draft and creates it on publish, so a failure during note generation leaves no tag pointing at an unreleased commit

**The `Stop` hook mirrors every CI step.** `uv lock --check` is what stands in for `uv sync --locked`: `uv.lock` pins the project's own version, so bumping `version` in `pyproject.toml` needs `uv lock` in the same commit or CI fails. That used to be the one failure with no local signal.

**`uv lock --check` must run *first*, ahead of every `uv run` step — the ordering is load-bearing, not cosmetic.** `uv run` syncs the project environment before running its command, which **rewrites `uv.lock` in place**. So the gate's own first `uv run ruff` silently repairs the drift, and a `uv lock --check` placed after it can never fail. Measured both ways: with the check third in the chain, a bumped `version` passed the gate green and left `uv.lock` dirty in the working tree; moved to first, it blocks with `The lockfile at 'uv.lock' needs to be updated`. The step costs ~0.02 s and the five-step gate still measures 3.17–3.25 s. The same property is why the local drift is easy to miss in the first place — any stray `uv run` fixes the lockfile on disk without staging it.

The gate mirrors CI's **steps**, not its **interpreter**: the hook runs against the local `.venv` (3.13) while CI pins `UV_PYTHON: "3.12"`. So a 3.13-only stdlib symbol passes the `Stop` hook's `pytest` and fails CI's. `ty check` is what closes that gap locally — `[tool.ty.environment]` targets 3.12 no matter which interpreter is running it — which is a reason not to drop it from the gate on the grounds that `pytest` already exercises the code.

Two `ruff` behaviors the gate now covers, worth keeping in mind rather than rediscovering:

- `ruff format` covers **Markdown** (0.16 reformats Python inside fenced code blocks in `.md`), so CI's `ruff format --check .` gates `CLAUDE.md` and `README.md`, not just the two `.py` files — `ruff format --check .` reports **4 files**
- selecting all of `E` enables `E501`, so an over-long line is a `ruff check` **error**, not something the formatter rewraps. But `E501` exempts a line holding a single whitespace-free "word" — a 124-character `x = '<120 chars>'` passes `ruff check` and is caught only by `ruff format --check`, which is why the gate runs both

## Code Style

- snake_case for functions/variables, PascalCase for classes
- Type annotations on all parameters and returns in `streamlit_app.py`; `tests/test_app.py` is deliberately unannotated (fixtures and mocks) and no `ANN` rules are selected
- `RuntimeError` for known transcription failures (no custom exception class)
- Import sorting via ruff with combine-as-imports
- `line-length = 100` (ruff's default is 88), `select = ["E", "F", "I", "W"]`

## Dependencies

- `mlx-whisper` — speech recognition on Apple Silicon
- `streamlit` — web UI (floor `>=1.58`; developed and tested against the locked 1.61.1). The floor is a deliberate support boundary, **not** the API minimum — the app renders and all tests pass on 1.57.0, which already has `st.tabs(key=, on_change=)` and `MutableTabContainer.open`
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
- `README.md` — user-facing docs, and the repo's largest concentration of hand-duplicated facts. It restates the accepted-format list, the 500 MB per-file cap, the Python floor, the `macos-14`/`mlx`-backend CI rationale, `ASR_MODEL_REPO` and its ~1.5 GB first-run weight download, the four dev commands, the `.srt`/`.txt` download switch, and every Advanced-options control's description. So changing `AUDIO_FORMATS`/`VIDEO_FORMATS`, `server.maxUploadSize`, `requires-python`, `runs-on`, `ASR_MODEL_REPO`, or any control's `help` text means a README edit in the same commit. Nothing checks this — the runner rationale drifted silently and took `53b1f8d` to correct
- `docs/screenshot.png`, `docs/screenshot-result.png` — embedded in README with descriptive alt text naming the tabs and controls. Reshoot both, and update the alt text, on any UI or theme change; nothing checks them. See **Screenshots** below for the capture recipe

### Screenshots

`docs/screenshot.png` / `docs/screenshot-result.png` are **896 px wide** and script-captured rather than hand-shot, so the pair stays mutually consistent. The recipe, in case it needs re-deriving:

- **Playwright driving the installed Chrome** — `uv run --no-project --with playwright`, then `p.chromium.launch(channel="chrome")`. The default `chromium` launch fails asking for `playwright install` (the cached build predates the current release); `channel="chrome"` needs no download. The two obvious alternatives are both dead ends: `screencapture` needs a Screen Recording grant, and the Chrome MCP `computer` tool returns a downscaled JPEG — wrong for a PNG asset
- **Geometry**: viewport 1200×1600, `device_scale_factor=2`, `color_scheme="dark"`. Clip to the `.stMainBlockContainer` padding box (704 CSS px) plus ~3% per side, then downscale to 896 px wide (Lanczos) to match the README's existing image width
- **Never shoot below a ~700 px viewport.** The `_labeled_toggle` rows no longer stack there (they moved to `_control_row()`, which holds to 420 px), but the **Transcribe** and **Download** rows still use `st.columns([3, 1])` and do: measured on the running app, the button goes from 168 px to the full content width somewhere between 800 px and 640 px, which reframes the whole shot
- Hide `[data-testid="stToolbar"]`: the Deploy button and hamburger are browser chrome, not app UI
- Filling **Keyterms** leaves the multiselect popover open, and while open the combobox marks the rest of the app `aria-hidden` — dismiss it (Escape, then click elsewhere) or `get_by_role("button", name="Transcribe")` will not resolve
- Streamlit's `centered` container is `max-width: 736px` (**px**) while its type scale is in **rem**, so the two do not scale together. A capture taken at a non-default root font size reads proportionally larger; text that looks smaller than an older screenshot at the same canvas width is the standard 16 px root, not a regression

### Model

Direct `mlx_whisper.transcribe()` call with `ASR_MODEL_REPO = "mlx-community/whisper-large-v3-turbo"`. MLX accelerates natively on Apple Silicon. Results are cached via `@st.cache_data(show_spinner=False, max_entries=20)`; the spinner is disabled because per-file progress is rendered by the `st.status` wrapper in `_handle_transcription`.

`_transcribe` also pins `no_speech_threshold=0.6`, `logprob_threshold=-1.0`, and `compression_ratio_threshold=2.4`. These are **mlx_whisper's own defaults**, restated to keep the decode policy visible at the call site — not tuned values. Deleting one is a no-op that lets the library's default track upstream; changing one is a real behavior change, and `test_transcribe_calls_mlx_with_correct_params` asserts all three.

Model weights are **not** wrapped in `st.cache_resource`. `mlx_whisper` already holds them on a process-level class attribute (`ModelHolder.get_model` in `mlx_whisper/transcribe.py`), so the repo is loaded once per server process; adding `st.cache_resource` would layer a second cache over an existing one.

### Fetch caching

`_fetch_youtube_audio` and `_fetch_url_audio` are `@st.cache_data(max_entries=5, ttl="1h")`. These return **raw audio bytes**, so the two caches together are the app's dominant memory cost: 2 caches × 5 entries, and only the URL path is size-capped (`MAX_URL_DOWNLOAD_BYTES` = 500 MB), putting the bounded side alone at ~2.5 GB and the pair well past that. `max_entries` alone never evicts an entry that nothing displaces, which is what `ttl` addresses. Both also keep `show_spinner` set to a download message rather than `False` — unlike `_transcribe`, nothing wraps these in an `st.status`, so the cache spinner is the only progress signal on a slow download. That is also why the deferred fetch runs inside `with youtube_slot:` / `with url_slot:` rather than just writing to the container afterwards: the spinner renders at the cursor, so a bare call would strand it below the controls instead of in the tab.

Be precise about what `ttl` does, though: Streamlit's `ttl_cache.py` expires **lazily** — expired entries read as absent, but are "only actually removed by a write, by `expire()`, or by a length/size query." So `ttl` bounds staleness and lets a later fetch reclaim the slot; it is *not* a background reaper, and a fully idle server can still hold expired payloads until something touches the cache. It shrinks the window, not the worst case.

**Known gap:** `_fetch_youtube_audio` has no size cap at all — the `downloaded.read_bytes()` at the end of the function slurps whatever `yt-dlp` wrote. A multi-hour livestream VOD is loaded whole into one `bytes` object and cached. The URL path's cap does not apply here; capping this would mean `yt-dlp`'s `max_filesize` option, since the download has already happened by the time the bytes are read.

`_transcribe` needs no `ttl` for the same reason it needs no size bound beyond `max_entries=20` — it caches result dicts (text + segments), not media.

### Input Modes

- `st.set_page_config(**PAGE_CONFIG)` is the first Streamlit call (before `st.title`); `PAGE_CONFIG` is a module constant (browser tab title `Whisper Transcribe`, a `:material/graphic_eq:` page icon, `centered` layout) so its values stay unit-testable
- **Upload** / **Record** / **YouTube** / **URL** tabs (`st.tabs`, each label prefixed with a Material Symbol icon — `upload`, `mic`, `smart_display`, `link`) — Upload accepts multiple files (`accept_multiple_files=True`) with one `st.audio` preview per file; the uploader's `type` is `AUDIO_FORMATS + VIDEO_FORMATS`, which Streamlit lists in the dropzone (no hand-maintained duplicate — but that line truncates, which is why the list is deliberately short; see **Accepted formats**); Record takes a single recording with its own preview; YouTube takes a URL (gated by a `youtube.com` / `youtu.be` regex and stripped of whitespace), downloads the best audio stream via `yt-dlp` (cached — see **Fetch caching** below — plus `restrictfilenames=True`, `noplaylist=True`), and shows an `st.audio` preview of the bytes; URL takes any `http(s)` audio/video file URL (gated by an `https?://` regex; YouTube URLs short-circuit to an `st.info` redirecting to the YouTube tab), downloads via `urllib.request.urlopen` with a 60-second timeout and a 500 MB cap (`MAX_URL_DOWNLOAD_BYTES`, cached), derives the filename from the URL path (percent-decoded, fallback `download`), and shows an `st.audio` preview
- Below the tabs, controls are grouped by intent (input → output → advanced) for visual hierarchy. Always-visible, in order: **Primary language** selector (input), then the output group — **Translate to English** toggle, **Include subtitles** toggle, **No verbatim** toggle (enables `word_timestamps=True` + `hallucination_silence_threshold=2.0` to skip hallucinations on non-speech audio like music outros). The three power-user controls live in a collapsed **`st.expander("Advanced options", icon=":material/tune:")`** (progressive disclosure — defaults still apply when closed; **do not** add `on_change="rerun"` + `if expander.open:` lazy gating here, see below): **Decode segments independently** toggle (sets `condition_on_previous_text=False` so each 30 s window decodes without prior-window context — robust on noisy audio at the cost of cross-boundary fluency), **Time range** text input (forwarded as `clip_timestamps`; comma-separated `start,end` pairs in seconds, e.g. `30,90` or `0,60,120,180`; blank → `"0"` for the full file; validated by `_validate_time_range` — malformed input disables the Transcribe button until corrected, with the `st.error` rendered outside the Advanced options expander (above the button) so the disabled reason stays visible even when the expander holding the input is collapsed), and **Keyterms** chip input (`st.multiselect` with `accept_new_options=True`, max 50 chips, joined with `, ` and forwarded as `initial_prompt`). Below everything, a right-aligned, full-width (`width="stretch"`) **Transcribe** button with a `:material/graphic_eq:` icon. `Include subtitles` stays in the always-visible output group (never the expander) because it has a user-visible side effect — it flips the download between `.srt` and `.txt`
- Every labelled control row shares one layout, defined once in **`_control_row()`**: `st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center")`, label first via `st.markdown(label, help=...)`, control second with `label_visibility="collapsed"`. Build rows through `_labeled_toggle(label, help_text)` or `_field_label(label, help_text)` (label row only, for a widget that renders itself below it) — **never a bare `st.toggle`**, and help text hangs off the label markdown, not the widget's own `help=`. The Primary language row uses `_control_row()` directly, since a selectbox is not a toggle. Neither helper has test coverage, so a bare widget breaks the grid silently
- **A horizontal container, not `st.columns([3, 1])` — the ratio was never load-bearing for these rows, only the flush-right edge was.** `horizontal_alignment="distribute"` is space-between with two children and plain left-alignment with one, which is why `_field_label` can share the identical row without a discarded empty column. The payoff is that columns stack vertically on a narrow viewport and a horizontal container does not; measured against an A/B app, `st.columns([3, 1])` breaks between 800 px and 640 px while `_control_row()` stays side-by-side down to 420 px
- `SELECT_WIDTH = 168` exists because **`st.selectbox` has no `width="content"`** — its default is `"stretch"`, which fills the whole horizontal container. 168 px is what `st.columns([3, 1])` yields and is measured, not guessed: the `centered` layout's main block has a 704 px content box and `st.columns` puts a 32 px gap between the two columns, so `(704 - 32) / 4 = 168`. It must stay equal to the width the **Transcribe** and **Download** buttons get, since those two rows still use `st.columns([3, 1])` (a button *does* need a width ratio) — verified in the running app: selectbox and button both land at `x=784, right=952, w=168`
- The Transcribe button dispatches in priority order: uploaded files → recording → YouTube audio → URL audio. Each non-upload source is wrapped in a single-element list. YouTube and URL sources share a `_RemoteAudio` adapter exposing `.name` (a safe filename, including extension when available) and `.read()` so they flow through `_handle_transcription` without changes. UI flags are routed through `_transcription_kwargs`, which centralizes the `translate → task` mapping and the `decode_independently → condition_on_previous_text` inversion so a script-level negation can't silently disappear. Subtitles controls both the text area's initial content (SRT-formatted segments when on, plain text when off) and the format the **Download** button serves (`.srt` vs `.txt`); the text area is always editable
- `_transcribe` writes audio bytes to a temp file, calls `mlx_whisper.transcribe()`, and caches results (`language=None` → Whisper auto-detects)
- `_handle_transcription` wraps the batch in `st.status(...)` (label updates to `Transcribing {name} ({i}/{total})...` per file, transitions to `complete` at the end), transcribes each upload, and stores the resulting list of `{result, file_stem, filename, include_subtitles}` dicts in `st.session_state["transcription"]`. The list is published **before** the loop and re-published after each file, never only at the end: Streamlit interrupts a running script at the next ForwardMsg (`status.update()` is one) by raising `RerunException`, a `BaseException` that the `except Exception` handler deliberately does not catch. A single post-loop assignment would therefore throw away every file already transcribed whenever any rerun lands mid-batch — a tab switch, a toggle, any widget. The pre-loop publish additionally clears a previous batch when nothing in this one succeeds. Alongside that publish it bumps `st.session_state["batch_id"]`, which namespaces `_display_transcription`'s widget keys — see below for why a batch-invariant key is a correctness bug, not a cosmetic one. The `file_stem` includes the source extension (e.g., `interview_mp3_transcript`) to disambiguate downloads when two uploads share a stem. Per-file errors are reported inline via `st.error` and don't stop the rest of the batch
- `_display_transcription` renders one stacked section per stored result: `st.subheader(filename)` (the filename is Markdown-escaped via `_escape_markdown`, since `st.subheader` renders the Markdown label subset — otherwise `_`, `*`, brackets, or `:` directives in names like YouTube titles would mis-render) + an editable text area (plain text or SRT segments per `include_subtitles`) + a right-aligned, full-width (`width="stretch"`) **Download** button with a `:material/download:` icon that captures the text area's edited content. The button sits in the right column of an `st.columns([3, 1])` split — same ratio as the Transcribe button — so the two share an edge and width. The label is always `Download` regardless of format; the file extension (`.srt` when subtitles are on, `.txt` otherwise) is set via the filename + MIME type args, and a `help=` tooltip (`"Downloads as .srt when subtitles are enabled, .txt otherwise."`) explains the format switch on hover. Widget keys are namespaced by batch *and* index (`transcript_b{batch}_{i}`, `download_{txt,srt}_b{batch}_{i}`), reading `batch_id` from session state. The `{i}` avoids collisions across files in one batch; the `b{batch}` is load-bearing for correctness. **A keyed `st.text_area` restores its session-state value and ignores the `value` argument**, so a batch-invariant `transcript_{i}` renders the *previous* batch's text under the new batch's `st.subheader` — and since the **Download** button's payload is the text area's return value, it serves that stale text under the new `file_stem`. This needs no user edit to trigger: transcribe A, then B, and B's text never appears. Flipping `include_subtitles` does not rescue it either, because that changes only the download key (`_txt` → `_srt`) while `transcript_*` is untouched, so the SRT cues never render
- `_display_transcription` is a plain function but is invoked through `st.fragment(_display_transcription)()` at the call site (not a `@st.fragment` decorator) so transcript edits and downloads rerun only this section instead of the whole script (which would otherwise re-evaluate all four input tabs). Wrapping at the call site rather than decorating keeps the function directly unit-testable — the real `st.fragment` wrapper returns `None` without running the body when there is no script-run context (bare test mode)

### Lazy Container Execution

`st.tabs` and `st.expander` can skip computing hidden content via `on_change="rerun"` plus the container's `.open` property, and the general best-practices guidance recommends it for expensive work behind tabs and expanders. This API is present at the declared `streamlit>=1.58` floor (verified against a 1.58.0 install: `st.tabs` already accepts `key` and `on_change`, and `TabContainer.open` exists), so using it does not require raising the floor. **That guidance does not apply to the Advanced options expander.** Its three widgets are not expensive work — they are widget *declarations* whose return values (`decode_independently`, `time_range_input`, `keyterms`) are consumed unconditionally further down the script. Gating the expander body would leave those names undefined whenever the expander is collapsed, i.e. by default. Widget declarations are cheap; their values are load-bearing.

The tabs **are** gated, but at a finer granularity than wrapping the whole body. `st.tabs(..., on_change="rerun", key="input_tabs")` enables each tab's `.open` flag, and the YouTube and URL tabs guard only their **fetch** on it (`if youtube_tab.open and ...`), never the `st.text_input` that feeds it.

**Those fetches do not run in the tab body — only their preview slot lives there.** Each tab declares its `st.text_input` and then a bare `youtube_slot = st.container()` / `url_slot = st.container()`; the gated fetch itself sits further down the script, below every control, and writes back into that reserved container. Streamlit paints top to bottom and marks not-yet-redrawn elements stale, so a fetch at its old position greyed out the language selector, all three toggles, and Advanced options for the length of a download that can run to 500 MB on a 60-second timeout. Reserving the slot is what lets the fetch move without the preview moving with it — verified through `AppTest`'s element tree, where the audio element still resolves as `Tab > Block > audio` despite running lexically after the expander. The **Transcribe** button is the one thing that still waits, and unavoidably: its `disabled` state reads `audio_sources`, which is exactly what the fetch produces.

That split is load-bearing. Wrapping the whole `with` body in `if tab.open:` would stop rendering the text input while the tab is hidden, and Streamlit removes state for widgets absent from a run (`SessionState._remove_stale_widgets`) — so the typed URL would be silently cleared on every tab switch. Widget declarations are cheap and must always run; only the side effect gets gated.

Note that `on_change="rerun"` does not itself skip anything: `TabContainer.__enter__`/`__exit__` are plain passthroughs (Python cannot skip a `with` body), so the flag only enables `.open` tracking and makes tab switches trigger a rerun. Every `if tab.open:` guard is hand-written.

Three accepted consequences. Switching tabs now costs a server rerun. `youtube_audio` / `url_audio` are `None` while another tab is active, so a pasted URL alone doesn't enable **Transcribe** unless its tab is showing (upload and record sources stay sticky across tabs, so the button's enabled state is source-dependent). And because a tab switch is now a rerun, it can **interrupt an in-flight transcription** — previously tab switching was purely client-side and could not; this is why `_handle_transcription` publishes results progressively rather than once at the end. `key="input_tabs"` holds the active tab's *label* in session state; `AppTest` has no tab-selection API, so the tests drive tab switches by writing that key.

### Accepted formats

Audio: mp3, m4a, wav, opus

Video: mp4, mov, webm, mkv

**The list is short on purpose, and the order is meaningful.** Both tuples run most-likely-first.
`ffmpeg` decodes far more than these eight — the tuples are a *product* decision about what the
uploader advertises and accepts, not a decoder limit.

The forcing constraint is that `st.file_uploader` renders `500MB per file • MP3, M4A, …` inside a
single span styled `white-space: nowrap; overflow: hidden; text-overflow: ellipsis`
(`static/js/FileUploader.*.js`), and no `st.file_uploader` parameter suppresses or reformats it —
the two halves are fed by `max_upload_size` (or `server.maxUploadSize`) and by `type` respectively,
so the only lever on the format list's *length* is shortening `type` itself. Measured
in a running app at the `centered` layout's max width: the span gets **571px**, the size prefix
eats **92px**, leaving **~479px** for the format list at Source Sans 14px (~31–40px per entry).
The former 17-format list needed 546px and truncated mid-word; the current eight need ~265px.
`test_format_list_fits_the_dropzone_hint` guards this with a 60-character proxy for that budget.

Two rejected alternatives, both worth not re-deriving:

- **An `st.caption` restating the full list below the uploader.** Works, but adds a permanent line
  of chrome under the dropzone to compensate for a truncated label. Rejected on UI grounds
- **Shortening `type` to the MIME shortcuts (`type=["audio", "video"]`).** Renders as just
  `audio, video` — the frontend formatter strips the `/*` and skips the uppercasing it applies to
  bare extensions — and always fits. But `enforce_filename_restriction`
  (`streamlit/elements/lib/file_uploader_utils.py`) **skips server-side validation entirely** when
  any entry contains a `/`, because the backend can't tell whether a file was meant to match a
  MIME pattern or an extension. Bare extensions are what keep that bypass guard active

Dropping a format rejects it browser-side *and* server-side, so this is a real narrowing, not a
label tweak. It applies only to the Upload tab: the YouTube and URL paths never consult these
tuples, so a `.flac` URL still transcribes fine.

Formats removed when the list was cut to eight, in case one needs restoring: `aac` (near-redundant
— standalone `.aac` is rare, AAC audio almost always arrives inside `.m4a`), `flac`, `ogg`
(Telegram voice notes, where WhatsApp uses `.opus`), `aiff`, `avi`, `wmv`, `flv`, `mpeg`, `3gpp`.

If a newline ever seems like the fix for a long format string somewhere: Streamlit's markdown
pipeline ships no `remark-breaks`, so a single `\n` collapses to a space rather than breaking.

### Upload Limit

- Per-file cap of **500 MB**, set in `.streamlit/config.toml` via `server.maxUploadSize`
- Enforced browser-side *and* server-side — the upload route rejects an oversized body three ways (a `Content-Length` fast-fail, a cap on the streaming read, and an exact post-parse size check), all HTTP 413, so the cap is not client-bypassable. Each file is a separate PUT, so with `accept_multiple_files=True` the cap is per-file, not aggregate
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

Mocked at the boundary (`mlx_whisper`, `yt_dlp`, `urlopen`, `st`). Shared fixtures (`mock_mlx`, `mock_st`, `mock_uploaded_file`) and helpers (`_make_file`, `_stub_urlopen`, `_stub_ytdlp`, `_make_transcription`, `_expected_transcribe_kwargs`, `_handle_transcription_kwargs`, `_ui_state`) factor the common setup; kwarg-forwarding cases are `@pytest.mark.parametrize`d. An autouse `_clear_caches` fixture clears the `@st.cache_data` wrappers before each test so cached results don't leak between cases. It also calls **`st.cache_data.clear()`**, which is not redundant: the per-wrapper `.clear()` calls only reach caches created by the *imported* `streamlit_app` module, while `AppTest` re-executes the script as a separate module with its own cache store that otherwise survives the whole pytest session. No assertion depends on it today — removing the global clear leaves the suite green, because the parametrize order runs `skipped_when_hidden` before `fetched_when_active`, so nothing has populated the AppTest-module cache by the time the hidden-tab case asserts. It keeps those cases order-*independent*: if that order ever flipped (reordering, `-k` selection, a randomizer plugin), the hidden-tab case would read a stale hit and its `call_count == 0` assertion would pass even with the `tab.open` guard removed.

The subsumption runs the other way too, which is the part that is easy to get backwards: `st.cache_data.clear()` is `CacheDataAPI.clear()` → `DataCaches.clear_all()` (`streamlit/runtime/caching/cache_data_api.py`), and `_data_caches` is a module-level singleton shared by every module in the process — so the global call already covers the three per-wrapper `.clear()` calls above. Those are kept as a readable inventory of which caches exist, not because the global clear misses them. **A new `@st.cache_data` function therefore needs no new line in this fixture.**

`mock_st` is **not** a bare `MagicMock`: it stubs `columns` to return a list sized from the spec (so `label_col, input_col = st.columns([3, 1])` unpacks) and `text_area` to echo its `value` argument straight back (which is the entire mechanism behind the download-reflects-edits assertions). Adding an `st.*` call whose return value is unpacked or fed onward inside a function under test means teaching this fixture about it first — otherwise the failure surfaces as a `MagicMock` unpacking error rather than a readable assertion.

`[tool.pytest.ini_options] pythonpath = ["."]` is what makes `from streamlit_app import ...` resolve — the project is not installed into the venv and declares no build backend, so `pytest` only works from the repo root (hence the `cd "$CLAUDE_PROJECT_DIR"` in the Stop hook).

- Format constants — `AUDIO_FORMATS` / `VIDEO_FORMATS` pinned exactly (order included, since it decides what survives if the dropzone label is ever cut), plus `test_format_list_fits_the_dropzone_hint`, a 60-character ceiling standing in for the dropzone's ~479px budget (see **Accepted formats**)
- `_transcribe` — defaults, kwarg forwarding (language, task, initial_prompt, no_verbatim, condition_on_previous_text, clip_timestamps), temp-file cleanup, empty-text guard; `no_verbatim=True` flips `word_timestamps` and `hallucination_silence_threshold`; `clip_timestamps` defaults to `"0"` (full file) and accepts custom ranges (e.g., `"30,90"`)
- `_handle_transcription` — session-state storage, per-file error handling (RuntimeError + unexpected), kwarg forwarding, multi-file batches, partial-failure scenarios; progressive publishing is pinned by two cases — a `BaseException` raised mid-batch (standing in for `RerunException`) leaves the already-finished files in session state, and a batch where every file fails clears the previous batch's results; a third asserts `batch_id` advances on every batch **including an all-failed one**, since a stale text area would otherwise outlive the results it was rendered from
- `_transcription_kwargs` — UI-flag → `_handle_transcription` kwargs mapping; `translate=True` ↔ `task="translate"`, `decode_independently=True` ↔ `condition_on_previous_text=False`; passthrough of `language`, `include_subtitles`, `initial_prompt`, `no_verbatim`, `clip_timestamps`
- `_display_transcription` — filename subheader, editable text area, right-aligned **Download** button in `st.columns([3, 1])`; label is always `Download` with a `:material/download:` icon and `width="stretch"`; filename, MIME type, and widget key are derived from `include_subtitles` (`.txt` vs `.srt`); `help=` tooltip preserved in both `.txt` and `.srt` paths; edits to the text area flow through to the download payload; multi-file stacked rendering (per-file widget counts and subheaders — the `transcript_b{batch}_{i}` / `download_{ext}_b{batch}_{i}` keys are pinned only at index 0 and batch 0, so a key that lost its `{i}` would still pass); a dedicated case renders at `batch_id = 7` to pin the namespace itself; filenames with Markdown-special characters are escaped in the subheader. Tested as a plain function (the `st.fragment` wrap is applied only at the call site, so the body runs directly under the mocked `st`)
- `_RemoteAudio` / `_fetch_youtube_audio` / `_fetch_url_audio` — adapter round-trip; YouTube fetch with mocked `yt_dlp` (bytes + filename, `extract_info` call args, safe-download options `format=bestaudio/best`, `noplaylist`, `restrictfilenames`, `quiet`); URL fetch with mocked `urlopen` (bytes + filename, `timeout=60`, query-string stripping, percent-decoded filename, empty-path fallback to `download`, oversized-response `RuntimeError`)
- Formatting helpers — `_format_language` (`None` → `"Detect"`, title-casing of lowercase codes), `_format_timestamp` (zero, minutes/seconds, hours, comma decimal marker), `_format_srt` (single-segment, multi-segment cue separator, `-->` escaping to `->` to keep SRT structure intact), `_escape_markdown` (plain passthrough, underscores, brackets, all special chars, backslash), `_validate_time_range` (blank/single/multi/whitespace/decimal/trailing-start valid cases → `None`; non-numeric, negative, end ≤ start, out-of-order, trailing/empty token → error message). Odd counts are valid — a trailing unpaired value is a start that runs to the end of the file, matching Whisper's `clip_timestamps`
- Tab-gated fetches (`AppTest`) — both remote tabs are parametrized over hidden/active: a URL typed while another tab is active performs no fetch, the same URL with its own tab active fetches exactly once, and the typed value is retained either way (proving the widget is not gated, only the fetch). A third case asserts an active remote tab enables **Transcribe**. The active tab is set by writing `session_state["input_tabs"]` (AppTest can't click tabs), and the network entry points are patched on `urllib.request` / `yt_dlp` rather than on `streamlit_app`, because AppTest re-executes the script each run and rebinds its imports. Both gates are mutation-checked: deleting either `tab.open` guard fails the corresponding hidden-tab case
- Module UI (`streamlit.testing.v1.AppTest`) — runs the real script (not mocked `st`, loaded via an absolute path so it's cwd-independent) to cover the module-level UI the mocked tests can't reach: clean render + page title, the four Material-icon tab labels, the **Transcribe** button's `:material/graphic_eq:` icon and disabled-without-audio state, the time-range input's inline `st.error` on invalid input (and no error on valid input, asserted end-to-end through the rendered UI), and (with `st.session_state["transcription"]` seeded) the fragment-rendered subheader + text area + **Download** button carrying its `:material/download:` icon; the empty-state case renders no download button. Two cases publish **twice** with different data (`_publish` mirrors `_handle_transcription`'s results-plus-`batch_id` write) to pin the batch-namespaced keys: the second batch's text must replace the first's, edits must still stick *within* a batch, and the same must hold when `include_subtitles` flips between the two. Staleness is a two-render property — no single-render test can see it, which is why the original bug survived the suite. Mutation-checked: reverting the keys to `transcript_{i}` fails both. A plain unit test asserts the `PAGE_CONFIG` constant (`set_page_config` args aren't introspectable via AppTest)

## Resources

- [mlx-whisper](https://pypi.org/project/mlx-whisper/)

## License

MIT (see `LICENSE`); declared via `license = "MIT"` / `license-files = ["LICENSE"]` in `pyproject.toml`.
