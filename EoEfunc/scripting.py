"""
Some utility functions mainly used internally for Judas scripts

Functions
---------
- setup_environment
- debug_output
"""
from typing import Any, Dict, List, Optional, Union
from contextlib import suppress

import vapoursynth as vs

core = vs.core


def setup_environment(env: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Utility function for setting variables when using the Start-BatchJudasEncode script

    Will fall back to DEFAULTS where Start-BatchJudasEncode parameters are missing

    Parameters
    ----------
    env: Dict[str, Any]
        Dictionary containing variables, described below:

        max_cache_size or DEFAULT_CACHE: int
            Max amount of system memory for Vapoursynth to use (in GiB)
        num_threads or DEFAULT_THREADS: int
            Number of threads for Vapoursynth to use
        GPU: Union[int, bool]
            Hardware acceleration device index to set. Falls back to 0
            False for no hardware acceleration
        src_path or DEFAULT_PATH: str
            Path to input video to be indexed

    Returns
    -------
    Dict[str, Any]
        Dictionary containing processed variables, described below:

        core: vs.Core
            Vapoursynth Core object with max cache size and thread count set
        GPU: int
            Hardware acceleration device index
        src_path: str
            Path to input video to be indexed
        debug: bool
            Set to true if no src_path was set (since we can assume we are not running from
            Start-BatchJudasEncode)

    Raises
    ------
    ValueError
        Thrown when no src_path or DEFAULT_PATH is found
    """
    import sys
    import __main__

    print(file=sys.stderr)

    def decode(value: Union[bytes, str]) -> str:
        if vs.__api_version__.api_major >= 4:
            return value
        else:
            return value.decode("utf-8")

    env = env or __main__.globals()

    if not ("core" in env and isinstance(env["core"], vs._CoreProxy)):
        env["core"] = vs.core

    if "max_cache_size" in env:
        env["core"].max_cache_size = int(decode(env["max_cache_size"])) * 1024
        print(
            f"Vapoursynth: INFO --> Allocated max {env['core'].max_cache_size/1024}GiB of RAM",
            file=sys.stderr,
        )
    else:
        if "DEFAULT_CACHE" in env:
            print(
                "Vapoursynth: WARNING --> No max cache size set, allocating max"
                f" {env['DEFAULT_CACHE']}GiB of RAM",
                file=sys.stderr,
            )
            env["core"].max_cache_size = env["DEFAULT_CACHE"] * 1024
        else:
            print(
                "Vapoursynth: WARNING --> No default cache size set, using Vapoursynth default",
                file=sys.stderr,
            )

    if "num_threads" in env:
        env["core"].num_threads = int(decode(env["num_threads"]))
        print(f"Vapoursynth: INFO --> Using {env['core'].num_threads} threads", file=sys.stderr)
    else:
        if "DEFAULT_THREADS" in env:
            print(
                f"Vapoursynth: WARNING --> No thread count set, using {env['DEFAULT_THREADS']}"
                " threads by default",
                file=sys.stderr,
            )
            env["core"].num_threads = env["DEFAULT_THREADS"]
        else:
            print(
                "Vapoursynth: ERROR --> No default thread count set, using Vapoursynth default",
                file=sys.stderr,
            )

    env["use_hwaccel"] = True
    if "GPU" in env:
        decoded_GPU = decode(env["GPU"])
        if decoded_GPU.lower() in ["false", "no"]:
            env["use_hwaccel"] = False
        else:
            env["GPU"] = 0
            with suppress(ValueError):
                env["GPU"] = int(decoded_GPU)
            print(f"Vapoursynth: INFO --> Using GPU {env['GPU']}", file=sys.stderr)
    else:
        env["GPU"] = 0
        print("Vapoursynth: WARNING --> No GPU specified, using GPU 0 by default", file=sys.stderr)

    env["debug"] = "src_path" not in env
    if env["debug"]:
        if "DEFAULT_PATH" in env:
            env["src_path"] = env["DEFAULT_PATH"]
            print(
                "Vapoursynth: WARNING --> No input video specified, using default"
                f" {env['src_path']}",
                file=sys.stderr,
            )
        else:
            raise ValueError("No DEFAULT_PATH set, and no src_path was specified.")
    else:
        env["src_path"] = decode(env["src_path"])

    print(f"Vapoursynth: INFO --> Beginning job - (\"{env['src_path']}\")", file=sys.stderr)
    return env


def debug_output(
    clip: Union[vs.VideoNode, str],
    text: Optional[str] = None,
    alignment: Optional[int] = None,
    fft: bool = False,
    split_planes: Union[bool, int, List[int]] = False,
    frame_props: Union[int, bool] = False,
    frame_num: Union[int, bool] = False,
    chroma_doubler: Optional[Union[str, bool]] = None,
) -> vs.VideoNode:
    """Utility function to output a Vapoursynth clip with text information

    The function also allows you to specify:
        - Whether to split the planes
        - Whether to calculate the Fourier Transformation
        - Whether to upscale subsampled formats

    The function will set the resulting clip to the next available output, except for 0. This is to
    avoid accidentally leaving text on a clip to be encoded via vspipe, which reads output 0.

    Parameters
    ----------
    clip: Union[vs.VideoNode, str]
        Input clip, or path to a source
    text: Optional[str] = None
        Clip name, for text.Text
    alignment: Optional[int] = None
        text.Text alignment
    fft: bool = False
        Whether to output the Fourier Transformation
    split_planes: Union[bool, int, List[int]] = False
        Whether to split the planes. If an int, or int list, only output specified plane numbers
    frame_props: Union[int, bool] = False
        Whether to draw frame properties on output. If an int, use as text alignment
    frame_num: Union[int, bool] = False
        Whether to draw frame number on output. If an int, use as text alignment
    chroma_doubler: Optional[Union[str, bool]] = None
        Whether to scale a subsampled clip to 444. If a string, use as a preset

    Returns
    -------
    vs.VideoNode
        The final clip that is outputted. This can safely be ignored, and is only returned in case
        you wish to do anything further with the clip

    Raises
    ------
    ValueError
        Raised when `chroma_doubler` string preset is unrecognised

    Dependancies
    ------------
    vsutil
        For split_planes
    """
    from .misc import FFTSpectrum
    from .format import set_format, _to_444Presets
    from vsutil import split

    if isinstance(clip, str):
        clip = core.lsmas.LWLibavSource(clip)

    def _output(clip: vs.VideoNode, text: str):
        nonlocal alignment, frame_num, frame_props
        output_indexes = [0] + list(vs.get_outputs().keys())
        index = min(set(range(max(output_indexes) + 2)) - set(output_indexes))
        textout = text
        if fft:
            textout = f"FFT - {text}" if text else text
            clip = FFTSpectrum(clip)
        frame_props = 7 if frame_props is True else frame_props
        frame_num = 9 if frame_num is True else frame_num
        alignment = (
            alignment or 7
            if 7 not in [frame_num, frame_props]
            else 9
            if 9 not in [frame_num, frame_props]
            else 8
        )
        if frame_props:
            clip = core.text.FrameProps(
                clip, alignment=None if frame_props is True else frame_props
            )
        if frame_num:
            clip = core.text.FrameNum(clip, alignment=None if frame_num is True else frame_num)
        clip = core.text.Text(clip, textout, alignment) if text else clip
        clip.set_output(index)
        return clip

    if isinstance(chroma_doubler, str):
        chroma_doubler = chroma_doubler.lower()

    if chroma_doubler and (clip.format.subsampling_h == clip.format.subsampling_w == 0):
        if chroma_doubler == "point":
            clip = set_format(clip, "444p", core.resize.Point)
        elif chroma_doubler == "bicubic":
            clip = set_format(clip, "444p", core.resize.Bicubic)
        elif chroma_doubler == "spline":
            clip = set_format(clip, "444p", core.resize.Spline36)
        elif chroma_doubler in _to_444Presets:
            clip = set_format(clip, chroma_doubler)
        else:
            raise ValueError(f"Unsupported chroma_doubler: {chroma_doubler}")

    if split_planes is not False:
        if chroma_doubler is None and (clip.format.subsampling_h == clip.format.subsampling_w != 0):
            clip = set_format(clip, "444p", core.resize.Point)
        planes = [0, 1, 2]
        if not isinstance(split_planes, bool):
            planes = split_planes if isinstance(split_planes, list) else [split_planes]
        clip_planes = split(clip)
        for i in planes:
            _output(clip_planes[i], f"{text} - Plane {i}" if text else text)
    else:
        _output(clip, text)
