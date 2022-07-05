import pathlib
import functools
import subprocess

from . import utils
from . import photomosaic
from . import tiles
from . import comparisons
from . import gif_utils

DEFAULTS = {"mode": "L", "tile_size": 8}


def iter_tile_directories(tile_directories=None):
    tile_directories = tile_directories or utils.SETTINGS.tile_directories
    for directory in filter(lambda x: x.is_dir(), pathlib.Path(tile_directories).iterdir()):
        yield directory


def get_default_tile_directory():
    for directory in iter_tile_directories():
        return directory
    raise ValueError(f"Please specify a tile_directory argument or export TILE_DIRECTORIES")


def make_default_outfile(filename):
    return pathlib.Path(filename).parent.joinpath("mosaic-of-" + pathlib.Path(filename).name)


def _convert_frame(filename, *args, **kwargs):
    frame_kwargs = {**kwargs, "outfile": filename, "file_format": "GIF"}
    image_to_mosaic(filename, *args, **frame_kwargs)


def image_to_mosaic(filename, tile_directory, cmp, outfile=None, file_format=None, **kwargs):
    mosaic = photomosaic.Photomosaic.load(filename, **kwargs)
    mosaic_tiles = tiles.Tiles.load(tile_directory, **utils.without(kwargs, "scale", "outfile"))
    mosaic = mosaic.transform(mosaic_tiles, cmp=cmp)
    if outfile:
        mosaic.save(outfile, format=file_format)
    return mosaic


def gif_to_mosaic(
    filename,
    tile_directory,
    cmp,
    outfile=None,
    max_colors=None,
    max_workers=utils.SETTINGS.max_workers,
    progress_class=utils.SETTINGS.progress_class,
    **kwargs,
):
    keep_frames = outfile is None
    mode = kwargs.get("mode", "L")
    max_colors = {"L": 2}.get(mode, 32) if max_colors is None else int(max_colors)
    mosaic_tiles = tiles.Tiles.load(tile_directory)
    with gif_utils.GifProcessor(filename, max_colors=max_colors, keep_frames=keep_frames) as gif:
        gif.split().map(
            functools.partial(_convert_frame, tile_directory=mosaic_tiles, cmp=cmp, **kwargs),
            progress_bar=progress_class,
            max_workers=max_workers,
        )
        if outfile:
            gif.save(outfile)
    return gif


def to_mosaic(
    filename,
    tile_directory=None,
    cmp="square_sum",
    max_colors=None,
    max_workers=utils.SETTINGS.max_workers,
    progress_class=utils.SETTINGS.progress_class,
    **kwargs,
):
    if tile_directory is None:
        tile_directory = get_default_tile_directory()
    cmp = cmp if callable(cmp) else getattr(comparisons.COMPARISON_REGISTRY, str(cmp))
    if str(filename).lower().endswith("gif"):
        result = gif_to_mosaic(
            filename,
            tile_directory,
            cmp,
            max_colors=max_colors,
            max_workers=max_workers,
            progress_class=progress_class,
            **kwargs,
        )
    else:
        result = image_to_mosaic(filename, tile_directory, cmp, **kwargs)
    return result


if __name__ == "__main__":
    import click

    def click_convert(f):
        return lambda _ctx, _param, value: f(value)

    def optional_convert(f):
        return click_convert(lambda x: x if x is None else f(x))

    def show_file(filename, cmd_template="firefox {}"):
        cmd = cmd_template.format(filename).split()
        return subprocess.run(cmd, check=True)

    @click.command()
    @click.argument("filename")
    @click.option("--tile-directory", default=utils.SETTINGS.default_tile_directory)
    @click.option("--tile-size", default=8, type=int)
    @click.option("--scale", default=1.0, type=float)
    @click.option("--mode", default="L", type=click.Choice(["L", "RGB", "RGBA"]))
    @click.option("--max-colors", default=None, callback=optional_convert(int))
    @click.option("--cmp", default="cython_square_sum", type=click.Choice(dir(comparisons.COMPARISON_REGISTRY)))
    @click.option("-o", "--outfile", default=None)
    @click.option("--show", is_flag=True)
    def main(filename, show=True, **kwargs):
        kwargs["outfile"] = kwargs.get("outfile") or make_default_outfile(filename)
        to_mosaic(filename, **kwargs)
        if show:
            show_file(kwargs["outfile"])

    main()
