import pathlib
from distutils.core import setup
from distutils.extension import Extension


USE_CYTHON = "auto"
if USE_CYTHON:
    try:
        from Cython.Distutils import build_ext
    except (ImportError, ModuleNotFoundError):
        if USE_CYTHON == "auto":
            USE_CYTHON = False
        else:
            raise

cmdclass = {}
ext_modules = []

if USE_CYTHON:
    ext_modules.extend(
        [
            Extension("photomosaic.matrix_math", ["cython_utils/matrix_math.pyx"]),
        ]
    )
    cmdclass.update({"build_ext": build_ext})
else:
    ext_modules.extend([Extension("photomosaic.matrix_math", ["cython_utils/matrix_math.c"])])

setup(
    name="photomosaic",
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    packages=["photomosaic"],
    package_dir={"photomosaic": "src"},
    include_package_data=True,
    install_requires=list(
        filter(lambda x: x and not x.startswith("#"), pathlib.Path("requirements.txt").read_text().split("\n"))
    ),
)
