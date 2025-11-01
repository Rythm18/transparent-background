Problem title
Remote HTTP/HTTPS input support for images and videos in transparent-background CLI/entrypoint

Problem brief
We need to allow users to pass an HTTP/HTTPS URL as the --source for the library’s primary entry point. Today, processing requires a local file, webcam ID, or directory. With this feature, the tool will accept a remote URL that points to an image or a video, resolve its type, fetch it, and then process it in the same way as a local source. For images, the URL should be streamed and opened directly as a PIL.Image without saving a permanent local copy. For videos, the URL should be downloaded to a temporary file and fed through the existing VideoLoader so that the rest of the pipeline works unchanged. The destination path handling and naming should mirror the current file-based behavior. This keeps the user experience seamless while unlocking cloud and web inputs for automation and demos.

Agent Instructions
Prompt the agent with high-level build plans and acceptance criteria
- Build plan:
  - Add URL-aware helpers in transparent_background/utils.py: is_url, fetch_image_from_url, download_url_to_tempfile, get_format_from_url, and URLImageLoader/URLVideoLoader that mirror ImageLoader/VideoLoader contracts.
  - Modify transparent_background/Remover.py entry_point to detect URLs. Infer media type from URL extension or response content-type. Route to URLImageLoader or URLVideoLoader accordingly. Keep all existing logic for progress, naming, format overrides, and webcam/local path handling intact.
  - Add requests>=2.28.0 to setup.py install_requires. Do not add or change CLI flags; this is a transparent enhancement.
  - Ensure temp files for remote videos are cleaned up, and that the code tolerates servers without HEAD support by falling back to GET.
- Acceptance criteria:
  - Given an HTTP URL to a JPEG/PNG, the image is fetched, processed, and saved with the same naming as local files (unless --dest/--format override).
  - Given an HTTP URL to a video (e.g., .mp4), the video is downloaded to a temporary file, processed frame-by-frame, and saved to the destination.
  - If the URL has no extension but returns image/* or video/* content-type, the correct loader is chosen.
  - Local files, directories, and webcam inputs continue to work unchanged.
  - The helper functions are importable from transparent_background.utils and are unit-testable without downloading model weights.
  - Reasonable network errors (timeouts, 4xx/5xx) raise clear exceptions without crashing the entire process.

Test Assumptions
- Public helpers exposed in transparent_background/utils.py:
  - def is_url(s: str) -> bool
  - def fetch_image_from_url(url: str, timeout: float = 30.0) -> PIL.Image.Image
  - def download_url_to_tempfile(url: str, suffix: Optional[str] = None, timeout: float = 60.0) -> str
  - def get_format_from_url(url: str, content_type: Optional[str] = None) -> str  # returns 'Image', 'Video', or ''
  - class URLImageLoader: __init__(self, url: str); __iter__/__next__ as ImageLoader; __len__ -> int
  - class URLVideoLoader: __init__(self, url: str); delegates to VideoLoader with a temp file; __len__ -> int
- Entry point behavior remains exported via transparent_background.Remover.entry_point(out_type, mode, device, ckpt, source, dest, jit, threshold, resize, save_format=None, reverse=False, flet_progress=None, flet_page=None, preview=None, preview_out=None, options=None). Tests focus on the helper functions to avoid model downloads.
- tests/new/test_remote_input_utils.py:
  - Spins up a local http.server serving samples/aeroplane.jpg
  - Asserts is_url returns True for HTTP(S) URLs and False for local paths
  - Verifies fetch_image_from_url returns a valid RGB PIL image with non-zero dimensions
  - Verifies download_url_to_tempfile writes a non-empty file to disk
  - Verifies get_format_from_url returns 'Image' for JPEG URLs (or content-type=image/jpeg)
