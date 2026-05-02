import base64
import io

import torch
import torchvision.transforms.functional as TF
from PIL import Image

import config

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)
_RESIZE_TO     = 256


def preprocess_image(file_bytes: bytes):
    """Return (tensor, preview_pil, original_pil) from raw radiograph bytes.

    tensor      — (1, 3, 224, 224) float32, ImageNet-normalised
    preview_pil — 224×224 RGB PIL image for display
    original    — original PIL image (pre-crop)
    """
    original = Image.open(io.BytesIO(file_bytes))

    if original.mode == "RGBA":
        bg = Image.new("RGB", original.size, (255, 255, 255))
        bg.paste(original, mask=original.split()[-1])
        original = bg
    elif original.mode != "RGB":
        original = original.convert("RGB")

    resized = original.resize((_RESIZE_TO, _RESIZE_TO), Image.LANCZOS)

    offset = (_RESIZE_TO - config.INPUT_SIZE) // 2
    preview = resized.crop((offset, offset,
                            offset + config.INPUT_SIZE,
                            offset + config.INPUT_SIZE))

    tensor = TF.to_tensor(preview)
    tensor = TF.normalize(tensor,
                          mean=list(_IMAGENET_MEAN),
                          std=list(_IMAGENET_STD))
    tensor = tensor.unsqueeze(0)

    return tensor, preview, original


def pil_to_data_url(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/png" if fmt.upper() == "PNG" else f"image/{fmt.lower()}"
    return f"data:{mime};base64,{encoded}"
