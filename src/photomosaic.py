import numpy as np
from PIL import Image

from . import utils


class Photomosaic:
    def __init__(self, data, tile_size=8):
        data = utils.crop_data(data, tile_size)
        blocks = utils.image_to_blocks(data, tile_size=tile_size, ensure_tiles=False)
        self._rows = data.shape[0] // tile_size
        self.data = utils.collapse(blocks, target_dimensions=2)
        self.height, self.width = data.shape[:2]
        self.color_channels = (0, *data.shape[2:3])[-1]
        self.tile_size = tile_size

    @classmethod
    @utils.log_execution_time
    def load(cls, filename, tile_size=8, mode="L", scale=1.0):
        with Image.open(filename) as img:
            new_size = tuple(np.math.ceil(dim * scale) for dim in img.size)
            img = img.resize(new_size, Image.ANTIALIAS)
            box = utils.greatest_rectangle((img.height, img.width), tile_size)
            img = img.crop(box).convert(mode)
            data = np.asarray(img, dtype="int32")
        return cls(data, tile_size)

    @utils.log_execution_time
    def transform(self, tiles, cmp=utils.square_sum):
        self.data = tiles.data[np.argmin(cmp(tiles.data, self.data), axis=1)]
        return self

    @utils.log_execution_time
    def to_image(self):
        blocks = utils.uncollapse_images(self.data, color_channels=self.color_channels)
        return utils.blocks_to_image(blocks, self._rows)

    def save(self, filename, *args, **kwargs):
        return self.to_image().save(filename, *args, **kwargs)

    def __del__(self):
        del self.data
