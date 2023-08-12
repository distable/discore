from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image, ImageFile, UnidentifiedImageError
from src.lib.printlib import trace, trace_decorator

ImageFile.LOAD_TRUNCATED_IMAGES = True

def pil2cv(img: Image) -> np.ndarray:
    try:
        return np.asarray(img)
    except Exception as e:
        print(f'Error converting PIL to CV: {e}')
        return np.ones((img.height, img.width, 3))


def cv2pil(img: np.ndarray) -> Image:
    return Image.fromarray(img)


def ensure_extension(path: str | Path, ext):
    path = Path(path)
    if path.suffix != ext:
        path = path.with_suffix(ext)
    return path

def save_jpg(pil, path, with_async=False):
    save_img(pil, path, with_async=with_async, img_format='JPEG')

def save_png(pil, path, with_async=False):
    save_img(pil, path, with_async=with_async, img_format='PNG')

def save_img(pil, path, with_async=False, img_format='PNG'):
    pil = load_pil(pil)
    if pil is None:
        return

    if img_format[0] == '.':
        img_format = {'.jpg': 'JPEG', '.png': 'PNG'}.get(img_format, None)
        if img_format is None:
            raise ValueError(f'Unknown format: {img_format}')

    with trace(f'save_img({Path(path).relative_to(Path.cwd())}, async={with_async}, {pil})'):
        path = Path(path)
        if img_format == 'PNG':
            path = ensure_extension(path, '.png')
        elif img_format == 'JPEG':
            path = ensure_extension(path, '.jpg')

        if with_async:
            save_async(path, pil)
        else:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            pil.save(path, format="PNG", quality=90)


def save_async(path, pil, format='PNG') -> None:
    if isinstance(path, Path):
        path = path.as_posix()

    # Use threaded lambda to save image
    def write(im) -> None:
        try:
            if isinstance(im, np.ndarray):
                im = cv2pil(im)
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            im.save(path, format='PNG')
        except:
            pass

    import threading
    t = threading.Thread(target=write, args=(pil,))
    t.start()


def save_jpg(pil, path, quality=90):
    path = ensure_extension(path, '.jpg')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(path, format='JPEG', quality=quality)


def save_npy(path, nparray):
    path = ensure_extension(path, '.npy')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), nparray)

def load_npy(path):
    path = ensure_extension(path, '.npy')
    path = Path(path)
    if not path.is_file():
        return None
    return np.load(str(path))


def save_json(data, path):
    import json
    path = Path(path).with_suffix('.json')

    if isinstance(data, dict) or isinstance(data, list):
        # data = json.dumps(data, indent=4, sort_keys=True)
        data = json.dumps(data)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.as_posix(), 'w') as w:
        w.write(data)


@trace_decorator
def load_json(path, default='required'):
    import json
    is_required = default == 'required'

    path = Path(path).with_suffix('.json')
    if not path.is_file():
        if is_required:
            raise FileNotFoundError(f'File not found: {path}')
        else:
            return default

    try:
        with open(path.as_posix(), 'r') as r:
            return json.load(r)
    except Exception as e:
        if is_required:
            raise e
        else:
            return default


def load_pil(path: Image.Image | Path | str, size=None):
    ret = None

    if isinstance(path, Image.Image): ret = path
    if isinstance(path, Path): ret = Image.open(path.as_posix())
    if isinstance(path, str) and Path(path).is_file(): ret = Image.open(path)
    if isinstance(path, str) and path.startswith('#'): ret = Image.new('RGB', size or (1, 1), color=path)
    if isinstance(path, np.ndarray): ret = cv2pil(path)

    if ret is None:
        raise ValueError(f'Unknown type of path: {type(path)}')

    ret = ret.convert('RGB')
    if size is not None:
        ret = ret.resize(size, Image.LANCZOS)

    return ret

def load_pilarr(pil, size=None):
    pil = load_pil(pil, size)
    pil = pil.convert('RGB')
    return np.array(pil)

def load_cv2(pil, size=None):
    ret = None

    has_size = isinstance(size, tuple) and None not in size
    is_web_url = isinstance(pil, str) and pil.startswith('http')

    if isinstance(pil, np.ndarray): ret = pil
    elif isinstance(pil, Image.Image): ret = pil2cv(pil)
    elif isinstance(pil, Path):
        try:
            ret = pil2cv(Image.open(pil.as_posix()))
        except UnidentifiedImageError:
            return None
    elif is_web_url:
        try:
            ret = pil2cv(Image.open(requests.get(pil, stream=True).raw))
        except UnidentifiedImageError:
            return np.zeros((size[1], size[0], 3), dtype=np.uint8)
            pass
    elif isinstance(pil, str) and Path(pil).is_file(): ret = cv2.imread(pil)
    elif isinstance(pil, str) and pil.startswith('#'):
        rgb = Image.new('RGB', size or (1, 1), color=pil)
        rgb = rgb.convert('RGB')
        ret = np.asarray(rgb)
    elif pil == 'black':
        ret = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    elif pil == 'white':
        ret = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255
    elif isinstance(pil, str) and pil.startswith('#'):
        # color string like 'black', etc.
        rgb = Image.new('RGB', size or (1, 1), color=pil)
        rgb = rgb.convert('RGB')
        ret = np.asarray(rgb)
    elif has_size:  # load_cv2 always tries to return some sort of img array no matter what
        ret = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255
        ret[:, :, 0] = 255

    if has_size and ret.shape[:2] != size:
        ret = cv2.resize(ret, size)

    return ret

def load_torch(path_or_cv2):
    import torch
    img = load_cv2(path_or_cv2)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img

def load_txt(path):
    path = Path(path)
    if not path.is_file():
        return None
    with open(path.as_posix(), 'r') as r:
        return r.read().strip()
def save_txt(path, txt):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.as_posix(), 'w') as w:
        w.write(txt)

def crop_or_pad(image, width, height, bg=(0, 0, 0), anchor=(0.5, 0.5)):
    h_img, w_img, _ = image.shape

    if width == w_img and height == h_img:
        return image  # No need to crop or pad if dimensions are already equal

    # Calculate the padding sizes
    pad_left = 0
    pad_right = 0
    pad_top = 0
    pad_bottom = 0

    if width > w_img:
        pad_left = int((width - w_img) * anchor[0])
        pad_right = width - w_img - pad_left
    elif width < w_img:
        crop_left = int((w_img - width) * anchor[0])
        crop_right = w_img - width - crop_left
        image = image[:, crop_left:w_img - crop_right]

    if height > h_img:
        pad_top = int((height - h_img) * anchor[1])
        pad_bottom = height - h_img - pad_top
    elif height < h_img:
        crop_top = int((h_img - height) * anchor[1])
        crop_bottom = h_img - height - crop_top
        image = image[crop_top:h_img - crop_bottom, :]

    # Pad the image
    image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0, 1))

    return image

# def crop_or_pad(ret, w, h, param):
#     """
#     Crop or pad h,w,c image to fit w and h
#     """
#
#
#     pass
#
#     # with trace(f"res_frame_cv2: crop"):
#
#     #     w = ret.shape[1]
#     #     h = ret.shape[0]
#     #     ow = w
#     #     oh = h
#     #     if w > h:
#     #         # Too wide, crop width
#     #         cropped_span = ow - self.w
#     #         if cropped_span > 0:
#     #             ret = ret[0:oh, cropped_span // 2:ow - cropped_span // 2]
#     #         else:
#     #             # We have to pad with black borders
#     #             w = self.w
#     #             h = self.h
#     #             padded = np.zeros((h, w, 3), dtype=np.uint8)
#     #             padded[:, (w - ow) // 2:(w - ow) // 2 + ow] = ret[0:oh, 0:ow]
#     #             ret = padded
#     #     else:
#     #         # Too tall, crop height
#     #         cropped_span = oh - self.h
#     #         if cropped_span > 0:
#     #             ret = ret[cropped_span // 2:oh - cropped_span // 2, 0:ow]
#     #         else:
#     #             # We have to pad with black borders
#     #             w = self.w
#     #             h = self.h
#     #             padded = np.zeros((h, w, 3), dtype=np.uint8)
#     #             padded[(h - oh) // 2:(h - oh) // 2 + oh, :] = ret[0:oh, 0:ow]
#     #             ret = padded
#     # return None
