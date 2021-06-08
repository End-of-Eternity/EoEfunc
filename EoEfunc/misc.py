from typing import Any, Callable, Dict, List, Union
import vapoursynth as vs

core = vs.core


def FFTSpectrum(clip: vs.VideoNode) -> vs.VideoNode:
    from .format import set_format

    return core.fftspectrum.FFTSpectrum(set_format(clip, 8))


def unfuck_edges(
    fucked: vs.VideoNode,
    notFucked: vs.VideoNode,
    left: int = 1,
    right: int = 1,
    top: int = 1,
    bottom: int = 1,
) -> vs.VideoNode:
    from .util import peak

    mask = core.std.BlankClip(
        fucked,
        width=fucked.width - left - right,
        height=fucked.height - top - bottom,
        format=fucked.format.replace(color_family=vs.GRAY, subsampling_w=0, subsampling_h=0),
        length=1,
        color=0,
    )
    mask = core.std.AddBorders(mask, left, right, top, bottom, peak(mask))
    return core.std.MaskedMerge(fucked, notFucked, mask, planes=0)


def ContraSharpening(denoised, original, radius=None, rep=13, planes=None):
    import havsfunc as haf

    cs = haf.ContraSharpening(denoised, original, radius, rep, planes)
    return unfuck_edges(cs, denoised)


def sized_grain(
    clip: vs.VideoNode,
    strength: Union[float, List[float]] = [0.35, 0],
    luma_scaling: int = 12,
    static: bool = True,
    size: float = 1.5,
    subsampledGrain: bool = False,
    scaler: Callable = core.resize.Bicubic,
    scalerargs: Dict[str, Any] = dict(filter_param_a=0, filter_param_b=0),
) -> vs.VideoNode:
    from vsutil.clips import get_y

    if not isinstance(strength, list):
        strength = [strength, 0]
    grain_mask = core.adg.Mask(core.std.PlaneStats(get_y(clip).std.BoxBlur()), luma_scaling)
    grainformat = clip.format
    if not subsampledGrain:
        grainformat = clip.format.replace(subsampling_h=0, subsampling_w=0)
    grainclip = core.std.BlankClip(
        clip,
        round(clip.width / (size * 2)) * 2,
        round(clip.height / (size * 2)) * 2,
        color=[32767] * 3,
        format=grainformat,
    )
    grain = core.grain.Add(grainclip, var=strength[0], uvar=strength[1], constant=static)
    grain = scaler(grain, width=clip.width, height=clip.height, format=clip.format, **scalerargs)
    grain = core.std.MergeDiff(clip, grain)
    grain = core.std.MaskedMerge(clip, grain, grain_mask)
    return grain


FFT = FFTSpectrum
