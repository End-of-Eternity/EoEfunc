import vapoursynth as vs
from typing import Optional, Callable

core = vs.core


def rescale(
    src: vs.VideoNode,
    width: int,
    height: int,
    kernel: str = "bilinear",
    taps: int = 4,
    b: float = 1 / 3,
    c: float = 1 / 3,
    smooth: float = 0,
    process: Optional[Callable[[vs.VideoNode], vs.VideoNode]] = None,
    mask_detail: bool = False,
    rescale_threshold: float = 0.03,
    get_mask: bool = True,
    **mask_args,
) -> vs.VideoNode:
    from .format import set_format, make_similar, make_similar_mask
    from vsutil import split, join
    import kagefunc as kgf
    import fvsfunc as fvf
    import nnedi3_rpow2

    if not height:
        raise ValueError("rescale: you need to specify a value for the output height")

    only_luma = src.format.num_planes == 1

    planes = split(src)
    src_luma = planes[0]

    planes[0] = descaled = make_similar(
        kgf._descale_luma(set_format(planes[0], "s"), width, height, kernel, taps, b, c), planes[0]
    )

    if smooth:
        smoothed = core.resize.Spline36(src_luma, width, height)
        planes[0] = core.std.Merge(planes[0], smoothed, smooth)

    if process:
        planes[0] = process(planes[0])

    planes[0] = nnedi3_rpow2.nnedi3cl_rpow2(planes[0], rfactor=2)
    planes[0] = core.resize.Spline36(planes[0], src.width, src.height)

    if only_luma and not (mask_detail or get_mask):
        return planes[0]

    if mask_detail or get_mask:
        upscaled = fvf.Resize(descaled, src.width, src.height, kernel=kernel, taps=taps, a1=b, a2=c)
        mask = get_descale_mask(src_luma, upscaled, rescale_threshold, **mask_args)
        if mask_detail:
            planes[0] = core.std.MaskedMerge(planes[0], src_luma, mask)
    scaled = join(planes)
    scaled = core.resize.Point(scaled, format=src.format)
    return (scaled, make_similar_mask(mask, scaled)) if get_mask else scaled


def get_descale_mask(
    src: vs.VideoNode,
    upscaled: vs.VideoNode,
    threshold: float = 0.015,
    maximum: int = 2,
    inflate: int = 2,
    morpho: bool = False,
    msize: int = 5,
    mshape: int = 0,
) -> vs.VideoNode:
    from vsutil import iterate

    mask = core.std.Expr([src, upscaled], f"x y - abs {threshold * 65535} < 0 65535 ?")
    mask = iterate(mask, core.std.Maximum, maximum)
    mask = core.morpho.Close(mask, msize, mshape) if morpho else mask
    mask = iterate(mask, core.std.Inflate, inflate)
    return mask
