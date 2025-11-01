import os
import threading
import socket
import contextlib
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urljoin
import shutil

import pytest
import requests
import numpy as np

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


def test_urlimageloader_len_and_name():
    samples_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "samples")
    image_name = "aeroplane.jpg"
    with run_http_server(samples_dir) as srv:
        host, port = srv.server_address
        base_url = f"http://{host}:{port}/"
        url = urljoin(base_url, image_name)

        URLImageLoader = getattr(tbu, "URLImageLoader", None)
        assert URLImageLoader is not None, "URLImageLoader not found"
        loader = URLImageLoader(url)
        assert len(loader) == 1
        img, name = next(iter(loader))
        assert name.endswith(image_name)
        assert hasattr(img, "size") and img.size[0] > 0 and img.size[1] > 0


def test_fetch_and_download_404_raises():
    samples_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "samples")
    with run_http_server(samples_dir) as srv:
        host, port = srv.server_address
        base_url = f"http://{host}:{port}/"
        bad_url = urljoin(base_url, "no_such_file.jpg")

        fetch_fn = getattr(tbu, "fetch_image_from_url", None)
        dl_fn = getattr(tbu, "download_url_to_tempfile", None)
        assert callable(fetch_fn) and callable(dl_fn)

        with pytest.raises(requests.HTTPError):
            fetch_fn(bad_url, timeout=5.0)
        with pytest.raises(requests.HTTPError):
            dl_fn(bad_url, timeout=5.0)


def test_entry_point_image_url_writes_output(tmp_path, monkeypatch):
    samples_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "samples")
    image_name = "aeroplane.jpg"
    with run_http_server(samples_dir) as srv:
        host, port = srv.server_address
        base_url = f"http://{host}:{port}/"
        url = urljoin(base_url, image_name)

        # Replace heavy Remover with a lightweight stub
        from PIL import Image as PILImage
        class DummyRemover:
            def __init__(self, *args, **kwargs):
                pass
            def process(self, img, type="rgba", threshold=None, reverse=False):
                # Return the image unchanged
                if isinstance(img, PILImage.Image):
                    return img
                return PILImage.fromarray(np.array(img))

        import transparent_background.Remover as rem_mod
        monkeypatch.setattr(rem_mod, "Remover", DummyRemover)

        out_dir = tmp_path
        rem_mod.entry_point(
            out_type="map",
            mode="base",
            device=None,
            ckpt=None,
            source=url,
            dest=str(out_dir),
            jit=False,
            threshold=None,
            resize="static",
            save_format=None,
            reverse=False,
            flet_progress=None,
            flet_page=None,
            preview=None,
            preview_out=None,
            options=None,
        )

        # Expect an output file with derived name
        expected = os.path.join(str(out_dir), "aeroplane_map.jpg")
        assert os.path.isfile(expected)


def test_entry_point_image_url_no_ext_content_type_and_default_dest(tmp_path, monkeypatch):
    # Prepare a dir with a file named 'content' (no extension) serving image bytes
    serve_dir = tmp_path / "serve"
    serve_dir.mkdir()
    samples_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "samples")
    src = os.path.join(samples_dir, "aeroplane.jpg")
    target = serve_dir / "content"
    shutil.copy(src, target)

    with run_http_server(str(serve_dir)) as srv:
        host, port = srv.server_address
        base_url = f"http://{host}:{port}/"
        url = urljoin(base_url, "content")

        # Force HEAD to report image content-type
        class H:
            headers = {"content-type": "image/jpeg"}
        monkeypatch.setattr(requests, "head", lambda *a, **k: H())

        # Stub Remover
        from PIL import Image as PILImage
        class DummyRemover:
            def __init__(self, *args, **kwargs):
                pass
            def process(self, img, type="rgba", threshold=None, reverse=False):
                if isinstance(img, PILImage.Image):
                    return img
                return PILImage.fromarray(np.array(img))
        import transparent_background.Remover as rem_mod
        monkeypatch.setattr(rem_mod, "Remover", DummyRemover)

        # Run without dest: output should go to CWD
        monkeypatch.chdir(str(tmp_path))
        rem_mod.entry_point(
            out_type="map",
            mode="base",
            device=None,
            ckpt=None,
            source=url,
            dest=None,
            jit=False,
            threshold=None,
            resize="static",
            save_format="jpg",
            reverse=False,
            flet_progress=None,
            flet_page=None,
            preview=None,
            preview_out=None,
            options=None,
        )
        assert os.path.isfile(os.path.join(str(tmp_path), "content_map.jpg"))


def test_urlvideoloader_len_with_mocked_download(tmp_path, monkeypatch):
    # Avoid downloading a real video; ensure loader constructs and reports length
    import transparent_background.utils as utils_mod
    URLVideoLoader = getattr(utils_mod, "URLVideoLoader", None)
    assert URLVideoLoader is not None, "URLVideoLoader not found"

    # mock download_url_to_tempfile to create a tiny fake mp4 file
    def fake_dl(url, suffix=None, timeout=60.0):
        import tempfile
        p = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        p.write(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom")
        p.flush()
        p.close()
        return p.name

    monkeypatch.setattr(utils_mod, "download_url_to_tempfile", fake_dl)

    loader = URLVideoLoader("http://example.com/foo.mp4")
    assert len(loader) == 1


def test_entry_point_video_url_writes_output_and_cleans_temp(tmp_path, monkeypatch):
    import transparent_background.Remover as rem_mod
    import transparent_background.utils as utils_mod

    # Mock download to known temp path
    fake_temp = str(tmp_path / "tmpvid.mp4")
    def fake_dl(url, suffix=None, timeout=60.0):
        with open(fake_temp, "wb") as f:
            f.write(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom")
        return fake_temp
    monkeypatch.setattr(utils_mod, "download_url_to_tempfile", fake_dl)

    # Fake VideoLoader yielding two frames
    from PIL import Image as PILImage
    class FakeVideoLoader:
        def __init__(self, root):
            self.frames = [PILImage.new("RGB", (64, 64), (255, 0, 0)), PILImage.new("RGB", (64, 64), (0, 255, 0))]
            self.size = 1
            self.fps = 24
        def __iter__(self):
            self.idx = 0
            return self
        def __next__(self):
            if self.idx >= len(self.frames):
                raise StopIteration
            fr = self.frames[self.idx]
            self.idx += 1
            return fr, "video.mp4"
        def __len__(self):
            return self.size
    monkeypatch.setattr(utils_mod, "VideoLoader", FakeVideoLoader)

    # Dummy remover
    class DummyRemover:
        def __init__(self, *a, **k):
            pass
        def process(self, img, type="map", threshold=None, reverse=False):
            return img
    monkeypatch.setattr(rem_mod, "Remover", DummyRemover)

    # Stub cv2 writer and cvtColor
    written_paths = []
    class FakeWriter:
        def __init__(self, path, fourcc, fps, size):
            written_paths.append(path)
        def set(self, *a, **k):
            return True
        def write(self, frame):
            return True
        def release(self):
            return True
    monkeypatch.setattr(rem_mod.cv2, "VideoWriter", FakeWriter)
    monkeypatch.setattr(rem_mod.cv2, "cvtColor", lambda arr, code: np.zeros((64,64,3), dtype=np.uint8))

    # Run with cwd as tmp_path, no dest provided
    monkeypatch.chdir(str(tmp_path))
    url = "http://example.com/aeroplane.mp4"
    rem_mod.entry_point(
        out_type="map",
        mode="base",
        device=None,
        ckpt=None,
        source=url,
        dest=None,
        jit=False,
        threshold=None,
        resize="static",
        save_format=None,
        reverse=False,
        flet_progress=None,
        flet_page=None,
        preview=None,
        preview_out=None,
        options=None,
    )

    assert any(os.path.basename(p) == "aeroplane_map.mp4" for p in written_paths)
    assert not os.path.exists(fake_temp)
