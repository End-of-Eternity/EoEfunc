import vapoursynth as vs

core = vs.core


def nnedi3(clip: vs.VideoNode, opencl: bool = True, **override) -> vs.VideoNode:
    """
    Standard nnedi3 antialiasing. Stolen from an old lvsfunc version.

    :param clip:     Input clip
    :param opencl:   Use OpenCL (Default: True)
    :param override: nnedi3 parameter overrides

    :return:         Antialiased clip
    """
    from vsutil import get_y

    nnedi3_args = dict(field=0, dh=True, nsize=3, nns=3, qual=1)
    nnedi3_args.update(override)

    aa_func = core.nnedi3cl.NNEDI3CL if opencl else core.nnedi3.nnedi3

    y = get_y(clip)

    aa = aa_func(y.std.Transpose(), **nnedi3_args)
    aa = aa.resize.Spline36(height=clip.width, src_top=0.5)
    aa = aa_func(aa.std.Transpose(), **nnedi3_args)
    aa = aa.resize.Spline36(height=clip.height, src_top=0.5)

    if clip.format.color_family != vs.GRAY:
        aa = core.std.ShufflePlanes([aa, clip], planes=[0, 1, 2], colorfamily=vs.YUV)

    return aa


def nnedi3_clamp(src: vs.VideoNode, aa: vs.VideoNode, strength: float = 1, opencl: bool = True):
    """
    Clamp stronger AAs to nnedi3.

    Just calls lvsfunc.aa.clamp_aa for you.

    :param src: Source clip
    :param aa: Strongly-AA'd clip (eg: eedi3)
    :param strength: Clamping strength (Default: 1)
    :param opencl: Use OpenCL for nnedi3 (Default: True)

    :return: Clip with clamped anti-aliasing
    """
    from lvsfunc.aa import clamp_aa

    return clamp_aa(src, nnedi3(src, opencl), aa, strength)


def minmax_clamp(src: vs.VideoNode, aa: vs.VideoNode, radius: int = 1):
    """
    Clamp aa to surrounding source clip pixel minimum/maximums.

    Quick and easy way to reduce excessive haloing caused by an aa.
    Not line sensitive - won't remove halos, will only clamp overbrightened pixels.

    :param src: Source clip.
    :param aa: AA'd clip.
    :radius: Radius to search for clamping (Default: 1)

    :return: Clip with clamped anti-aliasing
    """
    from vsutil import iterate

    minimum = iterate(src, core.std.Minimum, radius)
    maximum = iterate(src, core.std.Maximum, radius)

    return core.std.Expr([aa, minimum, maximum], "x y max z min")
