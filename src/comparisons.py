import functools
import math
import attr
import numpy as np


MAX_BATCH_SIZE = 4000


def cython_square_sum(tile_data, image_data):
    raise NotImplemented


@attr.s
class ComparisonRegistry:
    registry = attr.ib(factory=dict)
    function_loaders = attr.ib(factory=list)

    def get(self, item, default=None):
        return getattr(self, item, default)

    def __getattr__(self, name):
        try:
            return self.registry[name]
        except KeyError:
            # maybe it's not registered yet
            self.load()
            if name in self.registry:
                return self.registry[name]
            # nope, doesn't exist
            raise AttributeError(f"{type(self).__name__} has not attribute {name}")

    def register_function(self, f=None, *, name=None):
        if f is None:
            return functools.partial(self.register_func, name=name)
        self.registry[name or getattr(f, "__name__", f"cmp_{len(self.registry)}")] = f
        return f

    def register_function_loader(self, f):
        self.function_loaders.append(f)
        return f

    def load(self):
        while self.function_loaders:
            try:
                self.registry.update(self.function_loaders.pop()())
            except (ModuleNotFoundError, ImportError):
                continue

    def __dir__(self):
        self.load()
        return list(self.registry)


COMPARISON_REGISTRY = ComparisonRegistry()


@COMPARISON_REGISTRY.register_function
def square_sum(tile_data, image_data):
    return np.sum((tile_data[:, None] - image_data) ** 2, axis=2).T


@COMPARISON_REGISTRY.register_function
def manhattan_distance(tile_data, image_data):
    return np.sum((tile_data[:, None] - image_data) ** 2, axis=2).T


@COMPARISON_REGISTRY.register_function_loader
def load_cython_functions():
    import matrix_math

    global cython_square_sum

    def cython_square_sum(tile_data, image_data):
        out = np.zeros((image_data.shape[0], tile_data.shape[0]), image_data.dtype)
        return matrix_math.euclidean_distance_matrix_1d(image_data, tile_data, out)

    return {"cython_square_sum": cython_square_sum}


@COMPARISON_REGISTRY.register_function
def pure_python_square_sum(tile_data, image_data):
    return list(
        map(
            lambda image_tile: list(
                map(lambda tile: sum((pair[0] - pair[1]) ** 2 for pair in zip(tile, image_tile)), tile_data)
            ),
            image_data,
        )
    )


def batched_compare(tile_data, image_data, cmp=square_sum, batch_size=MAX_BATCH_SIZE):
    if image_data.shape[0] < batch_size:
        return square_sum(tile_data, image_data)
    image_data = np.array_split(image_data, math.ceil(image_data.shape[0] / batch_size))
    return np.row_stack(list(map(functools.partial(cmp, tile_data), image_data)))


COMPARISON_REGISTRY.register_function(functools.partial(batched_compare, cmp=square_sum), name="batched_square_sum")


def load_data():
    import pathlib

    files = pathlib.Path("sample_data")
    return map(np.load, map(str, files.glob("*")))
