Problem Title: Configurable Remote Download Timeouts and Retries

Problem Brief
Implement opt-in controls so remote URL downloads can fail fast or persist, by allowing callers to set timeout and retry counts. Users should be able to choose conservative defaults or longer, more resilient transfers without editing library code.

Agent Instructions
1. Add CLI arguments and `entry_point` parameters for `download_timeout` (float seconds) and `download_retries` (int, default 3). Thread these through to the download helpers.
2. Update `download_url_to_tempfile` and `fetch_image_from_url` to honor the timeout and retry options, using exponential backoff (base 0.5s, capped at 4s) and failing with the last `requests.RequestException` once retries are exhausted.
3. Preserve existing behaviour when options are omitted. Timeout default should remain 30s for images and 60s for videos.
4. Document the new options in the README usage section and ensure exiting error messages remain unchanged apart from retry attempt logging (use `logging.debug`).
5. Add regression tests covering: (a) CLI wiring (invoking `test.sh new` should simulate retries via monkeypatch); (b) ensuring retries occur the configured number of times; (c) verifying fallback to defaults when options are untouched.

Test Assumptions
- Tests expect new CLI flags `--download-timeout` and `--download-retries`, plus corresponding keyword arguments on `transparent_background.Remover.entry_point` and the internal helper functions.
- The helper `transparent_background.utils.download_url_to_tempfile(url, suffix=None, timeout=60.0, retries=3)` should exist after the change; tests will monkeypatch `requests` to count calls.
- `transparent_background.utils.parse_args` is expected to include the new CLI options.
- `transparent_background.Remover.entry_point` must pass `timeout` and `retries` keyword arguments to `URLImageLoader` and `URLVideoLoader` when invoked with remote URLs.
