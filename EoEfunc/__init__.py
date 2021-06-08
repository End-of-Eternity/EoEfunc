import functools
from typing import Callable

from . import format, frequencies, denoise, rescaling, misc, util, scripting, vsnumpy  # noqa

try:
    from ._metadata import __version__, __author__
except ImportError:
    __version__ = __author__ = "Unknown"

# submodule aliases
fmt = format
freq = frequencies
rs = rescaling
dn = denoise
vsnp = vsnumpy

# individual exports
debug_output = scripting.debug_output
setup_env = scripting.setup_environment

# region Deprecated


def __deprecate(old_name: str, func: Callable):
    @functools.wraps(func)
    def deprecated(*args, **kwargs) -> Callable:
        print(
            f"Warning: {old_name} is a deprecated alias for {func.__module__}.{func.__name__}, and"
            " will be removed soon. If you're running this, please let the script writer know"
        )
        return func(*args, **kwargs)

    # deprecated.__name__ = old_name
    globals().update({old_name: deprecated})


globals().update(_slocation=[0.0, 4, 0.35, 16, 0.4, 512, 1.0, 512])

__deprecate("setupEnv", scripting.setup_environment)
__deprecate("setFormat", format.set_format)
__deprecate("getFormat", format.get_format)
__deprecate("rescale", rescaling.rescale)

__deprecate("CMDegrain", denoise.CMDegrain)
__deprecate("_prefilter", denoise.prefilter3)

__deprecate("lowF", frequencies.extract_frequency)
__deprecate("lowFreq", frequencies.extract_frequency)
__deprecate("FreqMerge", frequencies.insert_frequency)

__deprecate("ContraSharpening", misc.ContraSharpening)
__deprecate("sizedGrain", misc.sized_grain)

# endregion Deprecated
