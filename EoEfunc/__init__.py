import vapoursynth as vs

if vs.__api_version__.api_major < 4:
    vs.VideoFormat = vs.Format
else:
    vs.YCOCG = None

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
