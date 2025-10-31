import os
import threading
import socket
import contextlib
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urljoin

import pytest

from transparent_background.utils import is_url

# We will import these only after feature is implemented.
# For base code, tests still reference their names via getattr to ensure failure is explicit.
import transparent_background.utils as tbu


@contextlib.contextmanager
def run_http_server(directory, host="127.0.0.1", port=0):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            # Python >= 3.7 supports directory arg
            super().__init__(*args, directory=directory, **kwargs)

    httpd = HTTPServer((host, port), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd
    finally:
        httpd.shutdown()
        thread.join(timeout=2)


def test_is_url_simple():
    assert is_url("http://example.com/foo.jpg")
    assert is_url("https://example.com/bar.mp4")
    assert not is_url("/local/path/file.jpg")


def test_remote_image_helpers_work(tmp_path):
    samples_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "samples")
    image_name = "aeroplane.jpg"
    image_path = os.path.join(samples_dir, image_name)
    assert os.path.isfile(image_path), "Sample image missing"

    with run_http_server(samples_dir) as srv:
        host, port = srv.server_address
        base_url = f"http://{host}:{port}/"
        url = urljoin(base_url, image_name)

        # Helper must exist
        fetch_fn = getattr(tbu, "fetch_image_from_url", None)
        assert callable(fetch_fn), "fetch_image_from_url not found"

        img = fetch_fn(url, timeout=10.0)
        # Validate a real image
        assert hasattr(img, "size") and img.size[0] > 0 and img.size[1] > 0
        assert getattr(img, "mode", "RGB") == "RGB"

        # Download helper
        dl_fn = getattr(tbu, "download_url_to_tempfile", None)
        assert callable(dl_fn), "download_url_to_tempfile not found"

        temp_path = dl_fn(url, timeout=10.0)
        assert os.path.isfile(temp_path)
        assert os.path.getsize(temp_path) > 0

        # Format inference
        fmt_fn = getattr(tbu, "get_format_from_url", None)
        assert callable(fmt_fn), "get_format_from_url not found"
        assert fmt_fn(url, content_type="image/jpeg") == "Image"
