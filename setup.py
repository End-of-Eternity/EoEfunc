from setuptools import setup
from distutils.util import convert_path

# stolen from https://github.com/Irrational-Encoding-Wizardry/vsutil/blob/master/setup.py
meta = {}
exec(open(convert_path("EoEfunc/_metadata.py")).read(), meta)


setup(
    name="EoEfunc",
    version=meta["__version__"],
    packages=["EoEfunc"],
    package_data={"EoEfunc": ["py.typed"]},
    author="EoE",
    description="A load of useless garbage written by me (EoE)",
    install_requires=["vapoursynth"],
    python_requires=">=3.8",
)
