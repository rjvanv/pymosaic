import pathlib

import numpy as np
from PIL import Image

# relative imports
from . import utils


TILE_DIRECTORY = pathlib.Path(__file__).parent.joinpath("tile_directories")


class Tiles:
    @staticmethod
    def _process_tile(image, tile_size=8, mode="L"):
        box = utils.greatest_square((image.height, image.width))
        image = image.crop(box).resize((tile_size, tile_size), Image.ANTIALIAS).convert(mode)
        return utils.collapse(np.asarray(np.asarray(image, dtype="int32")))

    def __init__(self, data, tile_size=8, mode="L"):
        self.data = data
        self.tile_size = tile_size
        self.mode = mode

    @classmethod
    def from_images(cls, images, tile_size=8, mode="L"):
        data = np.fromiter(map(lambda x: cls._process_tile(x, tile_size, mode), images), dtype="int32")
        return cls(data, tile_size, mode)

    @classmethod
    @utils.log_execution_time
    def load(cls, dir_path, tile_size=8, mode="L", glob="*"):
        if isinstance(dir_path, cls):
            return dir_path
        data = []
        for fn in pathlib.Path(dir_path).glob(glob):
            with Image.open(fn) as img:
                data.append(cls._process_tile(img, tile_size, mode))
        return cls(np.asarray(data), tile_size, mode)

    def __del__(self):
        del self.data
