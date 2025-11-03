import os
import importlib
import types as _types
from io import BytesIO

import pytest

try:
    import requests  # optional dependency for remote support
except ImportError:  # pragma: no cover - base image may lack requests
    requests = None

import transparent_background.utils as tbu


def require_callable(attr_name):
    fn = getattr(tbu, attr_name, None)
    assert callable(fn), f"{attr_name} not found"
    return fn


def require_requests():
    if requests is None:
        pytest.fail("requests dependency required for download tests")
    return requests


def default_retries():
    return getattr(tbu, "_DEFAULT_RETRIES", 3)


def test_parse_args_accepts_download_options(monkeypatch):
    monkeypatch.setattr(
        tbu.sys,
        "argv",
        [
            "prog",
            "--source",
            "input.jpg",
            "--download-timeout",
            "12.5",
            "--download-retries",
            "4",
        ],
    )
    args = tbu.parse_args()
    assert args.download_timeout == pytest.approx(12.5)
    assert args.download_retries == 4


def test_download_url_to_tempfile_retries_with_backoff(tmp_path, monkeypatch):
    req = require_requests()
    dl_fn = require_callable("download_url_to_tempfile")

    head_resp = _types.SimpleNamespace(headers={"content-type": "video/mp4"})
    monkeypatch.setattr(req, "head", lambda *a, **k: head_resp)

    attempts = []
    sleeps = []

    def fake_sleep(duration):
        sleeps.append(round(duration, 3))

    monkeypatch.setattr(tbu.time, "sleep", fake_sleep)

    class FailingThenSuccess:
        def __init__(self):
            self.calls = 0
        def __call__(self, url, stream=True, timeout=None):
            self.calls += 1
            attempts.append(timeout)
            if self.calls < 3:
                raise req.Timeout("temporary failure")
            return _types.SimpleNamespace(
                __enter__=lambda s: s,
                __exit__=lambda s, exc_type, exc, tb: False,
                raise_for_status=lambda s: None,
                iter_content=lambda s, chunk_size=8192: [b"data"],
            )

    getter = FailingThenSuccess()
    monkeypatch.setattr(req, "get", getter)

    path = dl_fn("http://example.com/video.mp4", timeout=5.0, retries=3)
    try:
        assert os.path.isfile(path)
        assert attempts == [5.0, 5.0, 5.0]
        assert sleeps[:2] == [0.5, 1.0]
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_download_url_to_tempfile_backoff_cap(tmp_path, monkeypatch):
    req = require_requests()
    dl_fn = require_callable("download_url_to_tempfile")

    monkeypatch.setattr(req, "head", lambda *a, **k: _types.SimpleNamespace(headers={}))

    sleeps = []
    monkeypatch.setattr(tbu.time, "sleep", lambda duration: sleeps.append(round(duration, 3)))

    class AlwaysFail:
        def __call__(self, url, stream=True, timeout=None):
            raise req.ConnectionError("fail")

    monkeypatch.setattr(req, "get", AlwaysFail())

    with pytest.raises(req.ConnectionError):
        dl_fn("http://example.com/vid.mp4", timeout=2.0, retries=6)

    assert sleeps[:5] == [0.5, 1.0, 2.0, 4.0, 4.0]


def test_download_url_to_tempfile_raises_last_exception(monkeypatch):
    req = require_requests()
    dl_fn = require_callable("download_url_to_tempfile")

    monkeypatch.setattr(req, "head", lambda *a, **k: _types.SimpleNamespace(headers={}))

    class Failing:
        def __init__(self):
            self.calls = 0
        def __call__(self, url, stream=True, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise req.Timeout("first")
            raise req.ConnectionError("second")

    monkeypatch.setattr(req, "get", Failing())

    with pytest.raises(req.ConnectionError) as exc:
        dl_fn("http://example.com/fail", timeout=3.0, retries=2)
    assert "second" in str(exc.value)


def test_fetch_image_timeout_overrides_and_defaults(tmp_path, monkeypatch):
    req = require_requests()
    fetch_fn = require_callable("fetch_image_from_url")

    payload = BytesIO()
    from PIL import Image as PILImage

    PILImage.new("RGB", (10, 10), (1, 2, 3)).save(payload, format="PNG")
    payload_bytes = payload.getvalue()

    recorded = []

    def fake_get(url, stream=True, timeout=None):
        recorded.append(timeout)
        return _types.SimpleNamespace(
            __enter__=lambda s: s,
            __exit__=lambda s, exc_type, exc, tb: False,
            raise_for_status=lambda s: None,
            content=payload_bytes,
        )

    monkeypatch.setattr(req, "get", fake_get)

    img = fetch_fn("http://example.com/img.png", timeout=12.0, retries=1)
    assert img.size == (10, 10)

    fetch_fn("http://example.com/default.png")
    assert recorded[0] == 12.0
    assert recorded[1] == 30.0  # default


def test_entry_point_passes_download_options(tmp_path, monkeypatch):
    require_requests()
    url = "http://example.com/photo.jpg"

    captured = {}

    class FakeLoader:
        def __init__(self, passed_url, *, timeout, retries):
            captured["url"] = passed_url
            captured["timeout"] = timeout
            captured["retries"] = retries
            from PIL import Image as PILImage
            self.frames = [(PILImage.new("RGB", (5, 5), (9, 9, 9)), "photo.jpg")]
            self.size = 1
        def __iter__(self):
            self.i = 0
            return self
        def __next__(self):
            if self.i >= len(self.frames):
                raise StopIteration
            frame = self.frames[self.i]
            self.i += 1
            return frame
        def __len__(self):
            return self.size

    monkeypatch.setattr(tbu, "URLImageLoader", FakeLoader)
    monkeypatch.setattr(tbu, "URLVideoLoader", FakeLoader)

    import transparent_background.Remover as rem_mod

    class DummyRemover:
        def __init__(self, *args, **kwargs):
            pass
        def process(self, img, type="map", threshold=None, reverse=False):
            return img

    monkeypatch.setattr(rem_mod, "Remover", DummyRemover)

    dest = tmp_path / "out"
    dest.mkdir()

    rem_mod.entry_point(
        out_type="map",
        mode="base",
        device=None,
        ckpt=None,
        source=url,
        dest=str(dest),
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
        download_timeout=4.5,
        download_retries=2,
    )

    assert captured["url"] == url
    assert captured["timeout"] == pytest.approx(4.5)
    assert captured["retries"] == 2


def test_urlvideo_loader_defaults_timeout_and_retries(tmp_path, monkeypatch):
    require_requests()
    import transparent_background.utils as utils_mod

    recorded = {}

    def fake_dl(url, suffix=None, timeout=None, retries=None):
        recorded["timeout"] = timeout
        recorded["retries"] = retries
        temp_path = tmp_path / "temp.mp4"
        temp_path.write_bytes(b"data")
        return str(temp_path)

    monkeypatch.setattr(utils_mod, "download_url_to_tempfile", fake_dl)

    class FakeVideoLoader:
        def __init__(self, path):
            self.frames = [(b"frame", "video.mp4")]
            self.size = 1
            self.fps = 30
            self.cap = _types.SimpleNamespace(get=lambda prop: 1)
        def __iter__(self):
            self.i = 0
            return self
        def __next__(self):
            if self.i >= len(self.frames):
                raise StopIteration
            value = self.frames[self.i]
            self.i += 1
            return value
        def __len__(self):
            return self.size

    monkeypatch.setattr(utils_mod, "VideoLoader", FakeVideoLoader)

    list(utils_mod.URLVideoLoader("http://example.com/foo.mp4"))

    assert recorded["timeout"] == pytest.approx(60.0)
    assert recorded["retries"] == default_retries()
