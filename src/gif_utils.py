import pathlib
import subprocess
import attr
from PIL import Image

from . import utils

DEFAULT_COLORS = 32


@attr.s
class GifProcessor:
    filepath = attr.ib(converter=pathlib.Path)
    frame_directory = attr.ib(
        default=attr.Factory(lambda self: self.filepath.parent / self.filepath.stem, takes_self=True),
        converter=pathlib.Path,
    )
    duration = attr.ib(default=10)
    max_colors = attr.ib(default=None, converter=lambda x: int(x or 0) or DEFAULT_COLORS)
    keep_frames = attr.ib(default=False)

    def cleanup(self, keep_frames=None):
        keep_frames = self.keep_frames if keep_frames is None else keep_frames
        if not keep_frames:
            utils.rm_tree(self.frame_directory)

    def __del__(self):
        self.cleanup()

    def _stitch_with_gifsicle(self, outfile=None):
        subprocess.run(
            [
                utils.SETTINGS.gifsicle_command,
                "-O2",
                *sorted(self.frame_directory.glob("*")),
                "--loop=0",
                "-o",
                f"{outfile or self.filepath}",
                f"--colors={self.max_colors}",
            ],
            check=True,
        )

    def _split_with_gifsicle(self):
        subprocess.run(
            [
                utils.SETTINGS.gifsicle_command,
                "--explode",
                "--unoptimize",
                str(self.filepath),
                "-o",
                f"{self.frame_directory}/frames",
            ],
            check=True,
        )

    def split(self):
        self.frame_directory.mkdir(exist_ok=True, parents=True)
        tool = "gifsicle" if utils.SETTINGS.gifsicle_enabled else "pil"
        getattr(self, f"_split_with_{tool}")()
        return self

    def stitch(self, outfile=None):
        tool = "gifsicle" if utils.SETTINGS.gifsicle_enabled else "pil"
        getattr(self, f"_stitch_with_{tool}")(outfile=outfile)
        return self

    def _split_with_pil(self):
        image = Image.open(self.filepath)
        for frame in range(image.n_frames):
            image.seek(frame)
            image.save(self.frame_directory / f"frame.{frame:03}.png")

    def _stitch_with_pil(self, outfile=None):
        outfile = outfile or self.filepath
        img, *imgs = [Image.open(f) for f in sorted(self.frame_directory.glob("*"))]
        img.save(
            fp=outfile,
            format="GIF",
            append_images=imgs,
            save_all=True,
            duration=self.duration,
            loop=0,
            optimize=True,
            quality=75,
        )

    @classmethod
    def from_frames(cls, frame_directory, outfile=None):
        filename = outfile or f"{pathlib.Path(frame_directory).absolute()}.gif"
        return cls(filename, frame_directory).stitch()

    def map(self, f, progress_bar=None, executor_class=None, **kwargs):
        executor_class = executor_class or utils.SETTINGS.executor_class
        with executor_class(**kwargs) as executor:
            files = sorted(self.frame_directory.glob("*"))
            results = executor.map(f, files)
            if progress_bar:
                with progress_bar(len(files)) as bar:
                    for _ in results:
                        bar()
            else:
                list(results)
        return self

    def __enter__(self, keep_frames=None):
        self.keep_frames = self.keep_frames if keep_frames is None else keep_frames
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    save = stitch
