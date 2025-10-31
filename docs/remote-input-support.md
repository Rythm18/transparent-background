Problem title
Remote HTTP/HTTPS input support for images and videos in transparent-background CLI/entrypoint

Problem brief
We need to allow users to pass an HTTP/HTTPS URL as the --source for the library’s primary entry point. Today, processing requires a local file, webcam ID, or directory. With this feature, the tool will accept a remote URL that points to an image or a video, resolve its type, fetch it, and then process it in the same way as a local source. For images, the URL should be streamed and opened directly as a PIL.Image without saving a permanent local copy. For videos, the URL should be downloaded to a temporary file and fed through the existing VideoLoader so that the rest of the pipeline works unchanged. The destination path handling and naming should mirror the current file-based behavior. This keeps the user experience seamless while unlocking cloud and web inputs for automation and demos.

This capability is especially useful for automation pipelines (e.g., server-side processing of user-submitted links), quick demos, and integrating the library into notebooks and CI where files may be stored in object storage or CDN backends. The goal is to introduce it as a non-breaking enhancement with minimal API surface changes—no new flags required—so that any existing scripts can simply pass a URL instead of a path and receive the same outputs. The implementation should be robust, handling servers that do not support HEAD requests, URLs without extensions, and reasonable network errors (timeouts, 4xx/5xx). Performance considerations include streaming for images, chunked downloads for videos, and cleaning up temporary files.

Agent Instructions
Build plan
- Introduce small, testable utilities in transparent_background/utils.py:
  - is_url(s: str) -> bool: simple scheme check for http/https.
  - fetch_image_from_url(url: str, timeout: float = 30.0) -> PIL.Image.Image: stream bytes via requests, validate status, open as RGB image.
  - download_url_to_tempfile(url: str, suffix: Optional[str] = None, timeout: float = 60.0) -> str: download to a NamedTemporaryFile(delete=False), infer suffix from URL path or content-type if not given; return the temp file path.
  - Optionally: get_format_from_url(url: str, content_type: Optional[str] = None) -> Literal['Image','Video',''] that inspects extension first, then content-type.
  - URLImageLoader and URLVideoLoader classes that match the iterator contracts of ImageLoader/VideoLoader and expose __len__ like their local counterparts.
- Wire entry_point in transparent_background/Remover.py to detect URLs with is_url(source). For URLs:
  - Determine media type using get_format_from_url; default to Image if unknown.
  - Image: instantiate URLImageLoader(source). Video: instantiate URLVideoLoader(source).
  - Choose save_dir like the current file path branch (default to CWD unless --dest provided).
  - Preserve all existing logic: type validations (e.g., rgba not for Video), progress displays, naming, extension overrides via --format, and writing.
- Dependencies: add requests>=2.28.0 to setup.py install_requires.
- Do not change any CLI flags. The feature is transparent for existing users—URLs are now valid values for --source in both CLI and API entry_point.

Additional considerations
- Timeout and reliability: Use small HEAD timeouts where helpful, but fall back gracefully when HEAD is not supported; always allow GET to succeed when possible. Keep defaults conservative (e.g., 30–60s) and chunk downloads to control memory usage.
- Type inference: Prefer extension from URL path; if absent, use the content-type response header (image/* → Image, video/* → Video). If still unknown, default to Image.
- Naming: For images, derive the output filename from the URL path’s basename. If the basename is empty, use a sensible default like "remote.jpg"; extension overrides remain possible via --format.
- Cleanup: Ensure remote video temporary files are removed at the end of iteration; handle exceptions without leaving large files behind.
- Backwards compatibility: Leave existing loaders and control flow untouched for file/directory/webcam sources; new logic applies only when the source is an HTTP(S) URL.

Acceptance criteria
- Passing a public HTTP URL to a JPEG or PNG processes and saves to the destination with the same naming behavior as local files.
- Passing an HTTP URL to a video file (e.g., .mp4) downloads to a temp file, processes frames, and writes a video output.
- If the URL lacks an extension, but the server returns an image/* or video/* content-type, the correct loader is chosen.
- All code paths are non-breaking for existing local file and directory usage.
- New utility functions are importable from transparent_background.utils and covered by tests without requiring model weights.
- Graceful errors for 4xx/5xx responses and unreachable URLs.

Test Assumptions
- New testable public helpers in transparent_background/utils.py:
  - def is_url(s: str) -> bool
  - def fetch_image_from_url(url: str, timeout: float = 30.0) -> PIL.Image.Image
  - def download_url_to_tempfile(url: str, suffix: Optional[str] = None, timeout: float = 60.0) -> str
  - def get_format_from_url(url: str, content_type: Optional[str] = None) -> str  # returns 'Image', 'Video', or ''
  - class URLImageLoader: __init__(self, url: str); __iter__/__next__ as ImageLoader; __len__ -> int
  - class URLVideoLoader: __init__(self, url: str); delegates to VideoLoader with a temp file; __len__ -> int
- Entry point behavior remains exported via transparent_background.Remover.entry_point(out_type, mode, device, ckpt, source, dest, jit, threshold, resize, save_format=None, reverse=False, flet_progress=None, flet_page=None, preview=None, preview_out=None, options=None) but tests will target the helper functions to avoid model downloads.
- tests/new/test_remote_input_utils.py will:
  - spin up a local http.server that serves the repository’s samples/aeroplane.jpg
  - assert is_url on the constructed URL is True
  - fetch_image_from_url returns a valid PIL image in RGB mode, with width and height > 0
  - download_url_to_tempfile fetches to an existing path whose file size > 0
  - get_format_from_url infers 'Image' from the URL extension
- A trivial base test exists to keep the base pipeline green.
