import vapoursynth as vs
from typing import Optional, Callable, Tuple
from functools import partial
import warnings

core = vs.core


def rescale(
    src: vs.VideoNode,
    width: int,
    height: int,
    kernel: str = "catmull",
    taps: int = 4,
    b: Optional[float] = None,
    c: Optional[float] = None,
    smooth: float = 0,
    process: Optional[Callable[[vs.VideoNode], vs.VideoNode]] = None,
    mask_detail: bool = False,
    rescale_threshold: float = 0.03,
    get_mask: bool = True,
    src_width: Optional[float] = None,
    src_height: Optional[float] = None,
    src_top: Optional[float] = None,
    src_left: Optional[float] = None,
    **mask_args,
) -> vs.VideoNode:
    from .format import process_as, make_similar_mask
    from vsutil import split, join
    import nnedi3_rpow2

    if not height:
        raise ValueError("rescale: you need to specify a value for the output height")

    only_luma = src.format.num_planes == 1

    planes = split(src)
    src_luma = planes[0]

    descaler, resizer = get_scalers(kernel, taps, b, c)
    if smooth == 1:
        descaler = resizer

    descale_args = dict(
        width=width,
        height=height,
        src_width=src_width,
        src_height=src_height,
        src_top=src_top,
        src_left=src_left,
    )

    planes[0] = descaled = process_as(src_luma, partial(descaler, **descale_args), "s")

    if smooth and smooth != 1:
        smoothed = core.resize.Spline36(src_luma, width, height)
        planes[0] = core.std.Merge(planes[0], smoothed, smooth)

    if process:
        planes[0] = process(planes[0])

    planes[0] = nnedi3_rpow2.nnedi3cl_rpow2(planes[0], rfactor=2)
    planes[0] = core.resize.Spline36(planes[0], src.width, src.height)

    if only_luma and not (mask_detail or get_mask):
        return planes[0]

    if mask_detail or get_mask:
        upscaled = resizer(descaled, src.width, src.height)
        mask = get_descale_mask(src_luma, upscaled, rescale_threshold, **mask_args)
        if mask_detail:
            planes[0] = core.std.MaskedMerge(planes[0], src_luma, mask)
    scaled = join(planes)
    scaled = core.resize.Point(scaled, format=src.format)
    return (scaled, make_similar_mask(mask, scaled)) if get_mask else scaled


def get_bicubic_scalers(b, c):
    return (
        partial(core.descale.Debicubic, b=b, c=c),
        partial(core.resize.Bicubic, filter_param_a=b, filter_param_b=c),
    )


Scaler = Callable[..., vs.VideoNode]

scalers = {
    "bilinear": lambda **_: (core.descale.Debilinear, core.resize.Bilinear),
    "spline16": lambda **_: (core.descale.Despline16, core.resize.Spline16),
    "spline36": lambda **_: (core.descale.Despline36, core.resize.Spline36),
    "spline64": lambda **_: (core.descale.Despline64, core.resize.Spline64),
    "bicubic": lambda b, c, **_: get_bicubic_scalers(b or 0, c or 0.5),
    "catmull": lambda **_: get_bicubic_scalers(0, 0.5),
    "mitchell": lambda **_: get_bicubic_scalers(1 / 3, 1 / 3),
    "c-spline": lambda **_: get_bicubic_scalers(0, 1),
    "lanczos": lambda taps, **_: (
        partial(core.descale.Delanczos, taps=taps),
        partial(core.resize.Lanczos, filter_param_a=taps),
    ),
}


def get_scalers(
    kernel: str, taps: int, b: Optional[float], c: Optional[float]
) -> Tuple[Scaler, Scaler]:
    kernel = "catmull" if kernel == "catmull_rom" else kernel
    if kernel in ["catmull", "mitchell", "c-spline"] and (b is not None or c is not None):
        warnings.warn("b and c are ignored for custom bicubic kernels")
    if kernel[:-1] == "lanczos":
        try:
            taps = int(kernel[-1])
            kernel = "lanczos"
        except ValueError:
            raise ValueError("rescale: invalid lanczos kernel taps")

    if kernel not in scalers:
        raise ValueError("rescale: invalid kernel")

    return scalers[kernel](b=b, c=c, taps=taps)


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
