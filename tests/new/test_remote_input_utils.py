import os
import threading
import socket
import contextlib
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urljoin
import shutil

import pytest
import requests

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
    # Create a local image to serve
    serve_dir = tmp_path / "serve1"
    serve_dir.mkdir()
    from PIL import Image as PILImage
    img_local = PILImage.new("RGB", (32, 32), (123, 222, 111))
    img_name = "img.jpg"
    img_local.save(str(serve_dir / img_name), format="JPEG")

    with run_http_server(str(serve_dir)) as srv:
        host, port = srv.server_address
        base_url = f"http://{host}:{port}/"
        url = urljoin(base_url, img_name)

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


def test_urlimageloader_len_and_name(tmp_path):
    serve_dir = tmp_path / "serve2"
    serve_dir.mkdir()
    from PIL import Image as PILImage
    img_local = PILImage.new("RGB", (16, 16), (10, 20, 30))
    img_name = "pic.jpg"
    img_local.save(str(serve_dir / img_name), format="JPEG")

    with run_http_server(str(serve_dir)) as srv:
        host, port = srv.server_address
        base_url = f"http://{host}:{port}/"
        url = urljoin(base_url, img_name)

        URLImageLoader = getattr(tbu, "URLImageLoader", None)
        assert URLImageLoader is not None, "URLImageLoader not found"
        loader = URLImageLoader(url)
        assert len(loader) == 1
        img, name = next(iter(loader))
        assert name.endswith(img_name)
        assert hasattr(img, "size") and img.size[0] > 0 and img.size[1] > 0


def test_fetch_and_download_404_raises(tmp_path):
    # Serve empty directory; any path will 404
    serve_dir = tmp_path / "serve404"
    serve_dir.mkdir()
    with run_http_server(str(serve_dir)) as srv:
        host, port = srv.server_address
        base_url = f"http://{host}:{port}/"
        bad_url = urljoin(base_url, "no_such_file.jpg")

        fetch_fn = getattr(tbu, "fetch_image_from_url", None)
        dl_fn = getattr(tbu, "download_url_to_tempfile", None)
        assert callable(fetch_fn) and callable(dl_fn)

        with pytest.raises(requests.RequestException):
            fetch_fn(bad_url, timeout=5.0)
        with pytest.raises(requests.RequestException):
            dl_fn(bad_url, timeout=5.0)


def test_entry_point_image_url_writes_output(tmp_path, monkeypatch):
    # Create server with image
    serve_dir = tmp_path / "serve3"
    serve_dir.mkdir()
    from PIL import Image as PILImage
    img_local = PILImage.new("RGB", (24, 24), (1, 2, 3))
    image_name = "photo.jpg"
    img_local.save(str(serve_dir / image_name), format="JPEG")

    with run_http_server(str(serve_dir)) as srv:
        host, port = srv.server_address
        base_url = f"http://{host}:{port}/"
        url = urljoin(base_url, image_name)

        # Replace heavy Remover with a lightweight stub
        class DummyRemover:
            def __init__(self, *args, **kwargs):
                pass
            def process(self, img, type="rgba", threshold=None, reverse=False):
                # Return the image unchanged
                if isinstance(img, PILImage.Image):
                    return img
                return img

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
        expected = os.path.join(str(out_dir), "photo_map.jpg")
        assert os.path.isfile(expected)


def test_entry_point_image_url_no_ext_content_type_and_default_dest(tmp_path, monkeypatch):
    # Prepare a dir with a file named 'content' (no extension) serving image bytes
    serve_dir = tmp_path / "serve"
    serve_dir.mkdir()
    from PIL import Image as PILImage
    img_local = PILImage.new("RGB", (20, 20), (9, 8, 7))
    (serve_dir / "content").write_bytes(b"")
    # write as no extension by saving to temp and copying bytes
    tmp_img = tmp_path / "tmp.jpg"
    img_local.save(str(tmp_img), format="JPEG")
    shutil.copy(str(tmp_img), str(serve_dir / "content"))

    with run_http_server(str(serve_dir)) as srv:
        host, port = srv.server_address
        base_url = f"http://{host}:{port}/"
        url = urljoin(base_url, "content")

        # Force HEAD to report image content-type
        class H:
            headers = {"content-type": "image/jpeg"}
        monkeypatch.setattr(requests, "head", lambda *a, **k: H())

        # Stub Remover
        class DummyRemover:
            def __init__(self, *args, **kwargs):
                pass
            def process(self, img, type="rgba", threshold=None, reverse=False):
                if isinstance(img, PILImage.Image):
                    return img
                return img
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

    # Also monkeypatch VideoLoader to control length
    class FakeVideoLoader:
        def __init__(self, root):
            self.size = 2
            self.frames = [(None, "v.mp4"), (None, "v.mp4")]
        def __iter__(self):
            self.i = 0
            return self
        def __next__(self):
            if self.i >= len(self.frames):
                raise StopIteration
            fr = self.frames[self.i]
            self.i += 1
            return fr
        def __len__(self):
            return self.size
    monkeypatch.setattr(utils_mod, "VideoLoader", FakeVideoLoader)

    loader = URLVideoLoader("http://example.com/foo.mp4")
    assert len(loader) == 2


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
    monkeypatch.setattr(rem_mod.cv2, "cvtColor", lambda arr, code: arr)

    # Avoid numpy dependency inside entry_point
    import types as _types
    monkeypatch.setattr(rem_mod, "np", _types.SimpleNamespace(array=lambda x: x))

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


def test_entry_point_video_url_no_ext_content_type(tmp_path, monkeypatch):
    import transparent_background.Remover as rem_mod
    import transparent_background.utils as utils_mod

    # Mock HEAD to report video content-type when URL lacks extension
    class H:
        headers = {"content-type": "video/mp4"}
    monkeypatch.setattr(requests, "head", lambda *a, **k: H())

    fake_temp = str(tmp_path / "tmpvid2.mp4")
    def fake_dl(url, suffix=None, timeout=60.0):
        with open(fake_temp, "wb") as f:
            f.write(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom")
        return fake_temp
    monkeypatch.setattr(utils_mod, "download_url_to_tempfile", fake_dl)

    # Fake VideoLoader
    from PIL import Image as PILImage
    class FakeVideoLoader:
        def __init__(self, root):
            self.frames = [PILImage.new("RGB", (8, 8), (0,0,0))]
            self.size = 1
            self.fps = 24
        def __iter__(self):
            self.i = 0
            return self
        def __next__(self):
            if self.i >= len(self.frames):
                raise StopIteration
            fr = self.frames[self.i]
            self.i += 1
            return fr, "video"
        def __len__(self):
            return self.size
    monkeypatch.setattr(utils_mod, "VideoLoader", FakeVideoLoader)

    class DummyRemover:
        def __init__(self, *a, **k):
            pass
        def process(self, img, type="map", threshold=None, reverse=False):
            return img
    monkeypatch.setattr(rem_mod, "Remover", DummyRemover)

    class FakeWriter:
        def __init__(self, path, fourcc, fps, size):
            self.path = path
        def set(self, *a, **k):
            return True
        def write(self, frame):
            return True
        def release(self):
            return True
    written = []
    def fake_writer(path, fourcc, fps, size):
        written.append(path)
        return FakeWriter(path, fourcc, fps, size)
    monkeypatch.setattr(rem_mod.cv2, "VideoWriter", fake_writer)
    monkeypatch.setattr(rem_mod.cv2, "cvtColor", lambda arr, code: arr)
    import types as _types
    monkeypatch.setattr(rem_mod, "np", _types.SimpleNamespace(array=lambda x: x))

    monkeypatch.chdir(str(tmp_path))
    url = "http://example.com/vid"  # no extension
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

    assert any(os.path.basename(p).endswith("_map.mp4") for p in written)
    assert not os.path.exists(fake_temp)


def test_entry_point_video_url_rgba_raises(tmp_path, monkeypatch):
    import transparent_background.Remover as rem_mod
    # Stub Remover to avoid heavy init
    class DummyRemover:
        def __init__(self, *a, **k):
            pass
    monkeypatch.setattr(rem_mod, "Remover", DummyRemover)

    # rgba with Video must raise before processing
    with pytest.raises(AttributeError):
        rem_mod.entry_point(
            out_type="rgba",
            mode="base",
            device=None,
            ckpt=None,
            source="http://example.com/foo.mp4",
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


def test_entry_point_image_url_no_ext_no_head_defaults_image(tmp_path, monkeypatch):
    # Prepare server serving 'mystery' without extension and HEAD failing
    serve_dir = tmp_path / "serve4"
    serve_dir.mkdir()
    from PIL import Image as PILImage
    img_local = PILImage.new("RGB", (10, 10), (4,5,6))
    tmp_img = tmp_path / "m.jpg"
    img_local.save(str(tmp_img), format="JPEG")
    shutil.copy(str(tmp_img), str(serve_dir / "mystery"))

    with run_http_server(str(serve_dir)) as srv:
        host, port = srv.server_address
        base_url = f"http://{host}:{port}/"
        url = urljoin(base_url, "mystery")

        # Fail HEAD
        def head_fail(*a, **k):
            raise requests.RequestException("HEAD fail")
        monkeypatch.setattr(requests, "head", head_fail)

        class DummyRemover:
            def __init__(self, *a, **k):
                pass
            def process(self, img, type="map", threshold=None, reverse=False):
                return img
        import transparent_background.Remover as rem_mod
        monkeypatch.setattr(rem_mod, "Remover", DummyRemover)

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
        assert os.path.isfile(os.path.join(str(tmp_path), "mystery_map.jpg"))
