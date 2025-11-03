import os
import re
import sys
import cv2
import yaml
import time
import torch
import hashlib
import argparse
import logging
import mimetypes
import tempfile

from io import BytesIO
from urllib.parse import urlparse

import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform

import numpy as np

from PIL import Image
from threading import Thread
from easydict import EasyDict

VID_EXTS = ('mp4', 'avi', 'h264', 'mkv', 'mov', 'flv', 'wmv', 'webm', 'ts', 'm4v', 'vob', '3gp', '3g2', 'rm', 'rmvb', 'ogv', 'ogg', 'drc', 'gif', 'gifv', 'mng', 'avi', 'mov', 'qt', 'wmv', 'yuv', 'rm', 'rmvb', 'asf', 'amv', 'mp4', 'm4p', 'm4v', 'mpg', 'mp2', 'mpeg', 'mpe', 'mpv', 'mpg', 'mpeg', 'm2v', 'm4v', 'svi', '3gp', '3g2', 'mxf', 'roq', 'nsv', 'flv', 'f4v', 'f4p', 'f4a', 'f4b')
IMG_EXTS = ('jpg', 'jpeg', 'bmp', 'png', 'ppm', 'pgm', 'pbm', 'pnm', 'webp', 'sr', 'ras', 'tiff', 'tif', 'exr', 'hdr', 'pic', 'dib', 'jpe', 'jp2', 'j2k', 'jpf', 'jpx', 'jpm', 'mj2', 'jxr', 'hdp', 'wdp', 'cur', 'ico', 'ani', 'icns', 'bpg', 'jp2', 'j2k', 'jpf', 'jpx', 'jpm', 'mj2', 'jxr', 'hdp', 'wdp', 'cur', 'ico', 'ani', 'icns', 'bpg')

_DEFAULT_IMAGE_TIMEOUT = 30.0
_DEFAULT_VIDEO_TIMEOUT = 60.0
_DEFAULT_RETRIES = 3
_BACKOFF_BASE = 0.5
_BACKOFF_CAP = 4.0

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',    '-s',  type=str,                   help="Path to the source. Single image, video, directory of images, directory of videos is supported.")
    parser.add_argument('--dest',      '-d',  type=str, default=None,     help="Path to destination. Results will be stored in current directory if not specified.")
    parser.add_argument('--type',      '-t',  type=str, default='rgba',   help="Specify output type. If not specified, output results will make the background transparent. Please refer to the documentation for other types.")
    parser.add_argument('--reverse',  '-R',  action='store_true',        help="Output will be reverse and foreground will be removed instead of background if specified.")
    parser.add_argument('--format',    '-f',  type=str, default=None,     help="Specify output format. If not specified, it will be saved with the format of input.")
    parser.add_argument('--resize',    '-r',  type=str, default='static', help="Specify resizing method. If not specified, static resize will be used. Choose from (static|dynamic).")
    parser.add_argument('--jit',       '-j',  action='store_true',        help="Speed up inference speed by using torchscript, but decreases output quality.")
    parser.add_argument('--device',    '-D',  type=str, default=None,     help="Designate device. If not specified, it will find available device.")
    parser.add_argument('--mode',      '-m',  type=str, default='base',   help="choose between base and fast mode. Also, use base-nightly for nightly release checkpoint.")
    parser.add_argument('--ckpt',      '-c',  type=str, default=None,     help="Designate checkpoint. If not specified, it will download or load pre-downloaded default checkpoint.")
    parser.add_argument('--threshold', '-th', type=str, default=None,     help="Designate threshold. If specified, it will output hard prediction above threshold. If not specified, it will output soft prediction.")
    parser.add_argument('--download-timeout', type=float, default=None,    help="Maximum seconds each remote download attempt should wait before timing out.")
    parser.add_argument('--download-retries', type=int, default=_DEFAULT_RETRIES, help="Total number of attempts for remote downloads (including the first attempt).")
    return parser.parse_args()

def get_backend():
    if torch.cuda.is_available():
        return "cuda:0"
    elif torch.backends.mps.is_available():
        return "mps:0"
    else:
        return "cpu"
    
def load_config(config_dir, easy=True):
    cfg = yaml.load(open(config_dir), yaml.FullLoader)
    if easy is True:
        cfg = EasyDict(cfg)
    return cfg

def get_format(source):
    img_count = len([i for i in source if i.lower().endswith(IMG_EXTS)])
    vid_count = len([i for i in source if i.lower().endswith(VID_EXTS)])
    
    if img_count * vid_count != 0:
        return ''
    elif img_count != 0:
        return 'Image'
    elif vid_count != 0:
        return 'Video'
    else:
        return ''

def sort(x):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(x, key=alphanum_key)

def download_and_unzip(filename, url, dest, unzip=True, **kwargs):
    if not os.path.isdir(dest):
        os.makedirs(dest, exist_ok=True)
    
    if os.path.isfile(os.path.join(dest, filename)) is False:
        os.system("wget -O {} {}".format(os.path.join(dest, filename), url))
    elif 'md5' in kwargs.keys() and kwargs['md5'] != hashlib.md5(open(os.path.join(dest, filename), 'rb').read()).hexdigest():
        os.system("wget -O {} {}".format(os.path.join(dest, filename), url))
        
    if unzip:
        os.system("unzip -o {} -d {}".format(os.path.join(dest, filename), dest))
        os.system("rm {}".format(os.path.join(dest, filename)))


def _import_requests():
    try:
        import requests  # type: ignore
    except ImportError as exc:  # pragma: no cover - handled by tests via skip
        raise ImportError(
            "The 'requests' package is required for remote URL support. Install transparent-background with the appropriate extras or add requests to your environment."
        ) from exc
    return requests


def _normalise_retries(retries):
    if retries is None:
        return _DEFAULT_RETRIES
    try:
        retries_int = int(retries)
    except (TypeError, ValueError):
        retries_int = _DEFAULT_RETRIES
    return max(1, retries_int)


def _call_with_retries(operation, *, retries, exception_type, sleep_fn=None):
    attempts = _normalise_retries(retries)
    if sleep_fn is None:
        sleep_fn = time.sleep
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except exception_type as exc:  # pragma: no cover - exercised in tests
            last_error = exc
            if attempt == attempts:
                break
            delay = min(_BACKOFF_BASE * (2 ** (attempt - 1)), _BACKOFF_CAP)
            logger.debug("Retrying download attempt %d/%d after error: %s", attempt, attempts, exc)
            sleep_fn(delay)
    assert last_error is not None
    raise last_error


def is_url(value):
    if not isinstance(value, str):
        return False
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _infer_suffix(url, content_type=None):
    if content_type:
        ctype = content_type.split(";")[0].strip().lower()
        guessed = mimetypes.guess_extension(ctype)
        if guessed == ".jpe":
            guessed = ".jpg"
        if guessed:
            return guessed
    parsed = urlparse(url)
    path = parsed.path or ""
    _, ext = os.path.splitext(path)
    if ext:
        return ext
    return ""


def _yield_name(url, default_name):
    parsed = urlparse(url)
    name = os.path.basename(parsed.path)
    return name or default_name


def _call_raise_for_status(response):
    method = getattr(response, "raise_for_status", None)
    if callable(method):
        try:
            method()
        except TypeError:
            method(response)


def _iter_chunks(response, chunk_iter):
    try:
        iterator = chunk_iter(chunk_size=8192)
    except TypeError:
        iterator = chunk_iter(response, chunk_size=8192)
    for chunk in iterator:
        yield chunk


def get_format_from_url(url, content_type=None, *, timeout=_DEFAULT_VIDEO_TIMEOUT, retries=_DEFAULT_RETRIES):
    if content_type is None:
        requests = _import_requests()

        def _head_request():
            return requests.head(url, allow_redirects=True, timeout=timeout)

        try:
            response = _call_with_retries(
                _head_request,
                retries=retries,
                exception_type=requests.RequestException,
            )
            headers = getattr(response, "headers", {}) or {}
            content_type = headers.get("content-type") if isinstance(headers, dict) else None
        except requests.RequestException:
            content_type = None

    if content_type:
        ctype = content_type.split(";")[0].strip().lower()
        if ctype.startswith("video/"):
            return "Video"
        if ctype.startswith("image/"):
            return "Image"

    ext = os.path.splitext(urlparse(url).path)[1].lower().lstrip(".")
    if ext in VID_EXTS:
        return "Video"
    if ext in IMG_EXTS:
        return "Image"
    return ""


def download_url_to_tempfile(url, suffix=None, timeout=None, retries=None):
    requests = _import_requests()
    effective_timeout = timeout if timeout is not None else _DEFAULT_VIDEO_TIMEOUT
    attempts = _normalise_retries(retries)

    def _head_request():
        return requests.head(url, allow_redirects=True, timeout=effective_timeout)

    try:
        head_response = _call_with_retries(
            _head_request,
            retries=attempts,
            exception_type=requests.RequestException,
        )
        headers = getattr(head_response, "headers", {}) or {}
        content_type = headers.get("content-type") if isinstance(headers, dict) else None
    except requests.RequestException:
        content_type = None

    inferred_suffix = suffix or _infer_suffix(url, content_type)
    if inferred_suffix and not inferred_suffix.startswith('.'):
        inferred_suffix = f".{inferred_suffix}"

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=inferred_suffix or "")
    temp_path = temp_file.name
    temp_file.close()

    def _get_request():
        return requests.get(url, stream=True, timeout=effective_timeout)

    response = None
    try:
        response = _call_with_retries(
            _get_request,
            retries=attempts,
            exception_type=requests.RequestException,
        )
        _call_raise_for_status(response)
        chunk_iter = getattr(response, "iter_content", None)
        with open(temp_path, "wb") as out_file:
            if callable(chunk_iter):
                for chunk in _iter_chunks(response, chunk_iter):
                    if not chunk:
                        continue
                    out_file.write(chunk)
            else:
                data = getattr(response, "content", b"")
                out_file.write(data)
    except Exception:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        raise
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    return temp_path


def fetch_image_from_url(url, timeout=None, retries=None):
    requests = _import_requests()
    effective_timeout = timeout if timeout is not None else _DEFAULT_IMAGE_TIMEOUT
    attempts = _normalise_retries(retries)

    def _get_request():
        return requests.get(url, stream=False, timeout=effective_timeout)

    response = _call_with_retries(
        _get_request,
        retries=attempts,
        exception_type=requests.RequestException,
    )
    try:
        _call_raise_for_status(response)
        data = getattr(response, "content", None)
        if data is None:
            data = response.text.encode()
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            close()

    image = Image.open(BytesIO(data))
    return image.convert("RGB")
        
class dynamic_resize:
    def __init__(self, L=1280): 
        self.L = L
                    
    def __call__(self, img):
        size = list(img.size)
        if (size[0] >= size[1]) and size[1] > self.L: 
            size[0] = size[0] / (size[1] / self.L)
            size[1] = self.L
        elif (size[1] > size[0]) and size[0] > self.L:
            size[1] = size[1] / (size[0] / self.L)
            size[0] = self.L
        size = (int(round(size[0] / 32)) * 32, int(round(size[1] / 32)) * 32)
    
        return img.resize(size, Image.BILINEAR)

class dynamic_resize_a(ImageOnlyTransform):
    def __init__(self, L=1280, always_apply=False, p=1.0):
        super(dynamic_resize_a, self).__init__(always_apply, p)
        self.L = L

    def apply(self, img, **params):
        size = list(img.shape[:2])
        if (size[0] >= size[1]) and size[1] > self.L: 
            size[0] = size[0] / (size[1] / self.L)
            size[1] = self.L
        elif (size[1] > size[0]) and size[0] > self.L:
            size[1] = size[1] / (size[0] / self.L)
            size[0] = self.L
        size = (int(round(size[0] / 32)) * 32, int(round(size[1] / 32)) * 32)

        return A.resize(img, height=size[0], width=size[1])

    def get_transform_init_args_names(self):
        return ("L",)
    
class static_resize:
    def __init__(self, size=[1024, 1024]): 
        self.size = size
                    
    def __call__(self, img):
        return img.resize(self.size, Image.BILINEAR)    

class normalize:
    def __init__(self, mean=None, std=None, div=255):
        self.mean = mean if mean is not None else 0.0
        self.std = std if std is not None else 1.0
        self.div = div
        
    def __call__(self, img):
        img /= self.div
        img -= self.mean
        img /= self.std
            
        return img
    
class tonumpy:
    def __init__(self):
        pass

    def __call__(self, img):
        img = np.array(img, dtype=np.float32)
        return img
    
class totensor:
    def __init__(self):
        pass

    def __call__(self, img):
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).float()
        
        return img

class ImageLoader:
    def __init__(self, root):
        if os.path.isdir(root):
            self.images = [os.path.join(root, f) for f in os.listdir(root) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            self.images = sort(self.images)
        elif os.path.isfile(root):
            self.images = [root]
        self.size = len(self.images)

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index == self.size:
            raise StopIteration
        
        img = Image.open(self.images[self.index]).convert('RGB')
        name = os.path.split(self.images[self.index])[-1]
        # name = os.path.splitext(name)[0]
            
        self.index += 1
        return img, name

    def __len__(self):
        return self.size
    
class VideoLoader:
    def __init__(self, root):
        if os.path.isdir(root):
            self.videos = [os.path.join(root, f) for f in os.listdir(root) if f.lower().endswith(('.mp4', '.avi', 'mov'))]
        elif os.path.isfile(root):
            self.videos = [root]
        self.size = len(self.videos)

    def __iter__(self):
        self.index = 0
        self.cap = None
        self.fps = None
        return self

    def __next__(self):
        if self.index == self.size:
            raise StopIteration
        
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.videos[self.index])
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        ret, frame = self.cap.read()
        name = os.path.split(self.videos[self.index])[-1]
        # name = os.path.splitext(name)[0]
        if ret is False:
            self.cap.release()
            self.cap = None
            img = None
            self.index += 1
        
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame).convert('RGB')
            
        return img, name
    
    def __len__(self):
        return self.size
    
class URLImageLoader:
    def __init__(self, url, *, timeout=None, retries=None):
        self.url = url
        self.timeout = timeout if timeout is not None else _DEFAULT_IMAGE_TIMEOUT
        self.retries = _normalise_retries(retries)
        self._frames = None
        self._index = 0

    def _ensure_loaded(self):
        if self._frames is None:
            image = fetch_image_from_url(self.url, timeout=self.timeout, retries=self.retries)
            name = _yield_name(self.url, "remote.jpg")
            self._frames = [(image, name)]

    def __iter__(self):
        self._ensure_loaded()
        self._index = 0
        return self

    def __next__(self):
        self._ensure_loaded()
        if self._index >= len(self._frames):
            raise StopIteration
        frame = self._frames[self._index]
        self._index += 1
        return frame

    def __len__(self):
        return 1

    def close(self):
        pass


class URLVideoLoader:
    def __init__(self, url, *, timeout=None, retries=None, suffix=None):
        self.url = url
        self.timeout = timeout if timeout is not None else _DEFAULT_VIDEO_TIMEOUT
        self.retries = _normalise_retries(retries)
        self.suffix = suffix
        self._temp_path = None
        self._inner = None
        self._iter = None
        self._closed = False

    def _ensure_inner(self):
        if self._inner is None:
            self._temp_path = download_url_to_tempfile(
                self.url,
                suffix=self.suffix,
                timeout=self.timeout,
                retries=self.retries,
            )
            self._inner = VideoLoader(self._temp_path)
        return self._inner

    def __iter__(self):
        inner = self._ensure_inner()
        self._iter = iter(inner)
        return self

    def __next__(self):
        inner = self._ensure_inner()
        if self._iter is None:
            self._iter = iter(inner)
        try:
            frame = next(self._iter)
        except StopIteration:
            self.close()
            raise
        return frame

    def __len__(self):
        inner = self._ensure_inner()
        return len(inner)

    def __getattr__(self, item):
        inner = self._ensure_inner()
        return getattr(inner, item)

    def close(self):
        if self._closed:
            return
        self._closed = True

        if self._inner is not None and hasattr(self._inner, "close"):
            try:
                self._inner.close()
            except Exception:
                pass

        cap = getattr(self._inner, "cap", None)
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

        if self._temp_path and os.path.exists(self._temp_path):
            try:
                os.remove(self._temp_path)
            except OSError:
                pass

        self._inner = None

    def __del__(self):
        self.close()


class WebcamLoader:
    def __init__(self, ID):
        self.ID = int(ID)
        self.cap = cv2.VideoCapture(self.ID)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.imgs = []
        self.imgs.append(self.cap.read()[1])
        self.thread = Thread(target=self.update, daemon=True)
        self.thread.start()
        
    def update(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret is True:
                self.imgs.append(frame)
            else:
                break
        
    def __iter__(self):
        return self

    def __next__(self):
        if len(self.imgs) > 0:
            frame = self.imgs[-1]
        else:
            frame = Image.fromarray(np.zeros((480, 640, 3)).astype(np.uint8))
            
        if self.thread.is_alive() is False or cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            raise StopIteration
        
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame).convert('RGB')
        
        del self.imgs[:-1]
        return frame, None

    def __len__(self):
        return 0
