import functools
import itertools
import logging
import pathlib
import sys
import time
import os
import subprocess
from concurrent import futures
import attr

import numpy as np
from PIL import Image

LOGGER = logging.getLogger("photomosaic")


def log_execution_time(f):
    @functools.wraps(f)
    def inner(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        LOGGER.debug(f"{getattr(f, '__name__', 'function')} took {time.time()-start} to complete.")
        return result

    return inner


def without(obj, *keys):
    obj_copy = obj.copy()
    for field in keys:
        obj_copy.pop(field, None)
    return obj_copy


def image_to_blocks(image, tile_size, ensure_tiles=True):
    if ensure_tiles:
        data = crop_data(np.asarray(image), tile_size)
    else:
        data = np.asarray(image)
    rows = data.shape[0] // tile_size
    cols = data.shape[1] // tile_size
    matrix = [np.split(row, cols, axis=1) for row in np.split(data, rows)]
    return np.asarray([col for row in matrix for col in row])


def blocks_to_image(data, rows):
    data = np.asarray(data) if isinstance(data, list) else data
    pix_map = np.concatenate([np.concatenate(row, axis=1) for row in np.split(data, rows)])
    return Image.fromarray(np.asarray(pix_map, dtype="uint8"))


def greatest_square(image_shape):
    dim = min(image_shape)
    return greatest_rectangle(image_shape, (dim, dim))


def greatest_rectangle(image_shape, tile_shape):
    tile_shape = (tile_shape, tile_shape) if isinstance(tile_shape, int) else tile_shape
    rectangle = []
    for dim, tile_size in zip(image_shape, tile_shape):
        remainder = dim % tile_size
        crop_before = remainder // 2
        crop_after = dim - (remainder - crop_before)
        rectangle.extend([crop_before, crop_after])
    upper, lower, left, right = rectangle
    return left, upper, right, lower


def crop_data(data, tile_size):
    left, upper, right, lower = greatest_rectangle(data.shape, tile_size)
    return data[upper:lower, left:right]


def collapse(data, target_dimensions=1):
    lead_dims = range(-1, -1 * target_dimensions, -1)
    tail_dim = np.multiply.reduce(data.shape[target_dimensions - 1 :])
    return data.reshape(*lead_dims, tail_dim)


def uncollapse_images(data, color_channels=None):
    tile_size = (data.shape[-1] / (color_channels or 1)) ** 0.5
    target_shape = np.fromiter(filter(bool, (-1, tile_size, tile_size, color_channels)), dtype=int)
    return data.reshape(*target_shape)


def square_sum(tile_data, image_data):
    return np.sum((tile_data - image_data[:, None]) ** 2, axis=2)


def resize(image, factor=1.0, resample=Image.ANTIALIAS):
    size = map(lambda x: int(x * factor), image.size)
    return image.resize(size, resample=resample)


@attr.s
class Settings:
    defaults = attr.ib(factory=dict)
    store = attr.ib(factory=dict)
    initializers = attr.ib(factory=dict)
    converters = attr.ib(factory=dict)

    def is_truthy(self, key):
        item = getattr(self, key)
        return item.upper() in ["1", "Y", "YES", "TRUE"] if isinstance(item, str) else bool(item)

    def __getattr__(self, item):
        key = item.upper()
        result = self.store.get(key) or os.environ.get(key) or self.defaults.get(key)
        if result is None and key in self.initializers:
            result = self.initializers[key]()
            self.store[key] = result
        return self.converters[key](result) if key in self.converters else result

    def register(self, f=None, loc="initializers", *, name):
        if f is None:
            return functools.partial(self.register, loc=loc, name=name)
        getattr(self, loc)[name.upper()] = f
        return f


SETTINGS = Settings(
    defaults={
        "TILE_DIRECTORIES": pathlib.Path(__file__).parent.joinpath("tile_directories"),
        "DEFAULT_TILE_DIRECTORY": pathlib.Path(__file__).parent.joinpath("tile_directories") / "letters",
        "MAX_WORKERS": 10,
        "EXECUTOR_CLASS": futures.ProcessPoolExecutor,
        "SHOW_COMMAND_TEMPLATE": "firefox {}",
        "GIFSICLE_COMMAND": "gifsicle",
    }
)

SETTINGS.register(int, loc="converters", name="MAX_WORKERS")


@SETTINGS.register(loc="converters", name="GIFSICLE_ENABLED")
def is_truthy(value):
    return value.upper() in ["1", "TRUE", "Y", "YES"] if isinstance(value, str) else bool(value)


def _test_command(*cmds):
    try:
        subprocess.run(
            list(cmds),
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


@SETTINGS.register(name="GIFSICLE_ENABLED")
def is_gifsicle_enabled():
    command = SETTINGS.gifsicle_command
    return _test_command(command, "--version")


def rm_tree(fp):
    fp = pathlib.Path(fp)
    if fp.is_dir():
        for child in fp.iterdir():
            rm_tree(child)
        fp.rmdir()
    else:
        fp.unlink(missing_ok=True)


def is_sized(obj):
    try:
        len(obj)
        return True
    except (ValueError, TypeError):
        return False


@attr.s
class ProgressBar:
    total = attr.ib(default=0)
    title = attr.ib(default=None)
    length = attr.ib(default=50)
    bar = attr.ib(default="\u25A0")
    console = attr.ib(default=sys.stdout)
    done = attr.ib(default=0, init=False)
    start_time = attr.ib(default=0, init=False)
    spinner = attr.ib(
        default=None,
        converter=lambda x: None if x is None else itertools.cycle(x) if is_sized(x) else x,
    )

    def __enter__(self, total=None):
        self.total = self.total if total is None else total
        self.done = 0
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.console.write("\n")
        self.console.flush()

    def _format_progress(self):
        fraction_done = self.done / self.total
        done = int(fraction_done * self.length)
        remaining = self.length - done
        digits = len(str(self.total))
        progress = f"|{self.bar * done}{' ' * remaining}|"
        done_out_of_total = f"{self.done: <{digits}}/{self.total}"
        percent_done = f"{fraction_done * 100:>5.2f}%"
        return f"{progress}{done_out_of_total} [{percent_done}]"

    def _format_spinner(self):
        return str(next(self.spinner)) if self.spinner else ""

    def _format_estimate(self):
        elapsed = time.time() - self.start_time
        if elapsed == 0.0 or self.done < 2:
            return f" in {elapsed:4.2f}s (?/s, eta: ?:?s)"
        rate = self.done / self.total
        return f" in {elapsed:4.2f}s ({rate:4.2f}/s, eta: {(self.total / rate) - elapsed:4.2f}s)    "

    def __call__(self):
        self.done += 1
        self.console.write(f"{self._format_progress()}{self._format_spinner()}{self._format_estimate()}\r")
        self.console.flush()


class SynchronousExecutor(futures.Executor):
    def __init__(self, *_args, **_kwargs):
        self.args = _args
        self.kwargs = _kwargs

    @staticmethod
    def submit(fn, /, *args, **kwargs):
        result = futures.Future()

        def do_result(self, _=None):
            try:
                self.set_result(fn(*args, **kwargs))
            except Exception as exc:
                self.set_exception(exc)

        setattr(result, "result", do_result.__get__(result, result.__class__))
        return result

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


EXECUTORS = {
    "THREADPOOLEXECUTOR": futures.ThreadPoolExecutor,
    "PROCESSPOOLEXECUTOR": futures.ProcessPoolExecutor,
    "SYNCHRONOUSEXECUTOR": SynchronousExecutor,
}


@SETTINGS.register(name="PROGRESS_CLASS")
def get_progress_bar():
    try:
        import alive_progress

        return alive_progress.alive_bar
    except ImportError:
        return ProgressBar


@SETTINGS.register(loc="converters", name="EXECUTOR_CLASS")
def get_executor_class(value):
    if isinstance(value, futures.Executor):
        return value
    return EXECUTORS.get(str(value).upper(), SynchronousExecutor)


if __name__ == "__main__":
    TOTAL = 28
    import random

    with ProgressBar(TOTAL) as bar:
        for _ in range(TOTAL):
            time.sleep(random.random())
            bar()
