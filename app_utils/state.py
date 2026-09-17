from dataclasses import dataclass
import numpy as np
from PIL import Image
import app_utils.config as config
from fractions import Fraction
import os


@dataclass(eq=False)
class Lenticules:
    top: np.ndarray
    bottom: np.ndarray
    shift: float
    tilt: float

    def __init__(self, top, bottom):
        self.top = top
        self.bottom = bottom
        self.shift = 0
        self.tilt = 0

    def __hash__(self) -> int:
        return hash(
            (
                round(float(self.shift), 6),
                round(float(self.tilt), 6),
            )
        )

    @property
    def shifted_top(self) -> np.ndarray:
        return self.top + self.shift

    @property
    def shifted_bottom(self) -> np.ndarray:
        return self.bottom + self.shift + self.tilt


@dataclass(eq=False)
class ImageState:
    name: str
    image_data: Image.Image
    preview_path: str
    raw_width: int
    lenticules: Lenticules
    selected: bool
    left: float
    right: float

    def __init__(self, name, image_data, raw_width, lenticules_top, lenticules_bottom):
        self.name = name
        self.image_data = image_data
        self.raw_width = raw_width
        self.lenticules = Lenticules(top=lenticules_top, bottom=lenticules_bottom)
        self.selected = False
        self.left = 0
        self.right = 1
        self.cache_webp()

    def __hash__(self) -> int:
        return hash(
            (
                round(float(self.left), 6),
                round(float(self.right), 6),
                self.selected,
                hash(self.lenticules),
            )
        )

    @property
    def is_cropped(self) -> bool:
        return not (self.left == 0 and self.right == 1)

    def set_selected(self, is_selected: bool):
        self.selected = is_selected
        if self.is_cropped:
            self.cache_webp()

    @property
    def crop_box(self) -> tuple[int, int, int, int]:
        w, h = self.image_data.size
        return (int(np.floor(self.left * w)), 0, int(np.ceil(self.right * w)), h)

    @property
    def crop_mask(self) -> Image.Image:
        mask = Image.new("1", self.image_data.size, color=1)
        mask.paste(0, self.crop_box)
        return mask

    @property
    def crop_slider_limits(self) -> tuple[float, float, int, int]:
        train_strip_section = config.TRAIN_STRIP_SIZE / self.raw_width
        left_crop_max = np.clip(
            np.round(self.right - train_strip_section - 0.001, 3), 0.001, 0.999
        )
        right_crop_min = np.clip(
            np.round(self.left + train_strip_section + 0.001, 3), 0.001, 0.999
        )
        frac = Fraction(left_crop_max / (1 - right_crop_min)).limit_denominator(64)
        return left_crop_max, right_crop_min, frac.numerator, frac.denominator

    @property
    def overlay(self) -> Image.Image:
        background = self.image_data.convert("RGBA")
        if self.is_cropped:
            foreground = self.image_data.convert("L").convert("RGBA")
            alpha_mask = self.crop_mask
            foreground.putalpha(alpha_mask)
            return Image.alpha_composite(background, foreground)
        else:
            return background

    def cache_webp(self):
        filename = os.path.splitext(os.path.basename(self.name))[0]
        path = os.path.join(config.CACHE_DIR, f"{filename}.webp")

        if self.selected and self.is_cropped:
            self.image_data.crop(self.crop_box).save(path, format="WEBP")
        else:
            self.image_data.save(path, format="WEBP")

        self.preview_path = path

    def shift_tilt_lenticules(self, shift, tilt) -> tuple[np.ndarray, np.ndarray]:
        self.lenticules.shift = shift
        self.lenticules.tilt = tilt
        return self.lenticules.shifted_top, self.lenticules.shifted_bottom


GalleryState = dict[str, ImageState]
