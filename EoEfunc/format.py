from typing import Callable, List, Optional, Tuple, Union, overload
import vapoursynth as vs

core = vs.core


def get_format(clip: Optional[vs.VideoNode] = None, formatstr: Optional[str] = None) -> vs.Format:
    import re

    if clip is None and formatstr is None:
        raise ValueError("Where's my params :(")
    if clip is not None and not isinstance(clip, vs.VideoNode):
        raise ValueError("Clip isn't a clip")
    if formatstr is not None:
        if isinstance(formatstr, int):
            formatstr = str(formatstr)
        elif not isinstance(formatstr, str):
            raise ValueError("formatstr must be an int or a string")
    else:
        return clip.format

    formatstr = formatstr.lower()
    pattern = (
        r"^(?!$)"  # start of line, and not immediately end
        r"(?:((yuv|ycbcr|ycocg|)"  # yuv/ycbcr/ycocg                            (group 2)
        r"(?:(4[0-4]{2})|)"  # subsampling                                      (group 3)
        r"|rgb|y|gray|)"  # colour sampling (includes groups 2-3)               (group 1)
        r"p?"  # allow specifying planar, but not required
        r"(?:([0-9]{1,2})(?:le)?|"  # bidepth, allow little endian (FFmpeg)     (group 4)
        r"(?:(16|32)"  # n bits for float                                       (group 5)
        r"?(f|float))|"  # n bit float (default 32)                             (group 6)
        r"(?:(?:32)?(s|single))|"  # specifically single precision float        (group 7)
        r"(?:(?:16)?(h|half))|)"  # specifically half precision float           (group 8)
        r"|compatyuy2|compatbgr32)"  # compat formats
        r"$"  # ending
    )
    match = re.search(pattern, formatstr)

    if not match:
        raise ValueError("Invalid formatstr")

    colorFamily = sampleType = bits_per_sample = subsampling_w = subsampling_h = None

    m_colorSampling = match.group(1)
    m_yuv_ycocg = match.group(2)
    m_subsampling = match.group(3)
    m_bits_per_sample = int(match.group(4) if match.group(4) else 0)
    # group 5 could be 16/32, group 6 is "float," which defaults to 32 since user didnt specify,
    # group 7 is single only (32), group 8 is half only (16). If none true, then return 0
    m_float_bits_per_sample = int(
        match.group(5)
        if match.group(5)
        else 32
        if (match.group(6) or match.group(7))
        else 16
        if match.group(8)
        else 0
    )
    if formatstr == "compatbgr32":
        return vs.COMPATBGR32
    elif formatstr == "compatyuy2":
        return vs.COMPATYUY2
    else:
        if m_colorSampling:
            if m_colorSampling == "gray" or m_colorSampling == "y":
                colorFamily = vs.GRAY
            elif m_colorSampling == "rgb":
                colorFamily = vs.RGB
            # yuv and ycocg
            else:
                if m_subsampling:
                    if not m_yuv_ycocg and clip.format.color_family not in [vs.YUV, vs.YCOCG]:
                        raise ValueError("Only yuv or ycocg can be subsampled")
                    if m_subsampling == "444":
                        subsampling_w, subsampling_h = 0, 0
                    elif m_subsampling == "422":
                        subsampling_w, subsampling_h = 1, 0
                    elif m_subsampling == "440":
                        subsampling_w, subsampling_h = 0, 1
                    elif m_subsampling == "420":
                        subsampling_w, subsampling_h = 1, 1
                    elif m_subsampling == "411":
                        subsampling_w, subsampling_h = 2, 0
                    elif m_subsampling == "410":
                        subsampling_w, subsampling_h = 2, 2
                    else:
                        raise ValueError("Unknown subsampling")
                if m_yuv_ycocg == "yuv" or m_yuv_ycocg == "ycbcr":
                    colorFamily = vs.YUV
                elif m_yuv_ycocg == "ycocg":
                    colorFamily = vs.YCOCG
        if m_float_bits_per_sample:
            sampleType = vs.FLOAT
            bits_per_sample = m_float_bits_per_sample
        elif m_bits_per_sample:
            sampleType = vs.INTEGER
            if colorFamily == vs.RGB:
                if m_bits_per_sample % 3:
                    raise ValueError("bitdepth for RGB must be mod 3")
                m_bits_per_sample = m_bits_per_sample // 3
            if m_bits_per_sample < 8:
                raise ValueError("bitdepth too low (8 minimum)")
            elif m_bits_per_sample > 32:
                raise ValueError("bitdepthtoo great (32 maximum)")
            bits_per_sample = m_bits_per_sample

    if clip:
        subsampling_w = 0 if colorFamily in [vs.RGB, vs.GRAY] else subsampling_w
        subsampling_h = 0 if colorFamily in [vs.RGB, vs.GRAY] else subsampling_h
        args = dict(
            color_family=colorFamily,
            sample_type=sampleType,
            bits_per_sample=bits_per_sample,
            subsampling_w=subsampling_w,
            subsampling_h=subsampling_h,
        )
        # remove undefined vars because format.replace bad
        args = {k: v for k, v in args.items() if v is not None}
        return clip.format.replace(**args)
    else:
        # defaults to yuv444p16
        # Chose 444 over 420 since ommitting subsampling should mean no subsampling
        colorFamily = colorFamily or vs.YUV
        sampleType = sampleType or vs.INTEGER
        bits_per_sample = bits_per_sample or 16
        subsampling_w = subsampling_w or 0
        subsampling_h = subsampling_h or 0
        return core.register_format(
            colorFamily, sampleType, bits_per_sample, subsampling_w, subsampling_h
        )


def guess_matrix(clip: vs.VideoNode) -> str:
    # guess the matrix based on either frame size (where unspec.) or from frame props
    if clip.format.color_family == vs.RGB:
        return "rgb"
    if clip.format.color_family == vs.YCOCG:
        return "ycgco"
    matrix = "709" if max(clip.width, clip.height) >= 720 else "170m"
    props = clip.get_frame(0).props
    if "_Matrix" in props and props["_Matrix"] != 2:
        matrix = matrix_enum(props["_Matrix"])
    return matrix


def from_string(formatstr: str) -> vs.Format:
    return get_format(None, formatstr)


def _set_format_internal(
    clip: vs.VideoNode,
    newformat: vs.Format,
    resizer: Optional[Callable] = None,
    chroma_doubler: Union[Callable, str, bool] = False,
    **resizer_args,
) -> vs.VideoNode:

    if clip.format == newformat:
        return clip

    guessmatrix = "709" if max(clip.width, clip.height) >= 720 else "170m"

    # fmt: off
    isfrom = lambda fmt: clip.format.color_family in (fmt if isinstance(fmt, list) else [fmt]) # noqa
    isto = lambda fmt: newformat.color_family in (fmt if isinstance(fmt, list) else [fmt]) # noqa
    # fmt: on

    if (
        isfrom(vs.RGB)
        and isto([vs.GRAY, vs.YUV])
        and ("matrix" not in resizer_args and "matrix_s" not in resizer_args)
    ):
        resizer_args.update(matrix_s=guessmatrix)
    if (
        isfrom([vs.GRAY, vs.YUV])
        and isto(vs.RGB)
        and ("matrix_in" not in resizer_args and "matrix_in_s" not in resizer_args)
    ):
        props = clip.get_frame(0).props
        if "_Matrix" not in props or ("_Matrix" in props and props["_Matrix"] == 2):
            resizer_args.update(matrix_in_s=guessmatrix)

    if resizer is None:
        if not chroma_doubler and (
            clip.format.subsampling_h != newformat.subsampling_h
            or clip.format.subsampling_w != newformat.subsampling_w
        ):
            resizer = core.resize.Bicubic
        else:
            resizer = core.resize.Point

    if chroma_doubler:
        clip = to_444(clip, chroma_doubler if chroma_doubler is not True else None)

    return resizer(clip=clip, format=newformat, **resizer_args)


def set_format(
    clips: Union[vs.VideoNode, List[vs.VideoNode]],
    formatstr: Union[str, int],
    resizer: Optional[Callable] = None,
    chroma_doubler: Union[Callable, str, bool] = False,
    **resizer_args,
) -> vs.VideoNode:

    if clips is None or formatstr is None:
        raise ValueError("setFormat requires both at least one clip and a formatstr")

    if not isinstance(clips, list):
        clips = [clips]

    newformat = get_format(clips[0], formatstr)
    out = [
        _set_format_internal(clip, newformat, resizer, chroma_doubler, **resizer_args)
        for clip in clips
    ]
    return out if len(out) != 1 else out[0]


def get_chroma_offsets(
    clip: vs.VideoNode, chromaloc: Union[str, int, bool, Tuple[float, float]] = True
) -> Tuple[float, float]:
    if isinstance(chromaloc, tuple):
        offsets = chromaloc
    elif chromaloc is False or isinstance(chromaloc, (str, int)):
        offsets = [0, 0]
        if chromaloc in ["left", 0, "top_left", 2, "bottom_left", 0]:
            offsets[0] += -0.5  # left -> (-0.5, Y)
        if chromaloc in ["top_left", 2, "top", 3]:
            offsets[1] += -0.5  # top -> (X, -0.5)
        elif chromaloc in ["bottom_left", 4, "bottom", 5]:
            offsets[1] += 0.5  # bottom -> (X, +0.5)
    else:
        subsampling = (clip.format.subsampling_w, clip.format.subsampling_h)
        if subsampling in [(1, 0), (1, 1)]:
            offsets = (0.5, 0)  # 422 and 420 (MPEG-2+)
        elif subsampling == (0, 1):
            offsets = (0, 0)  # 440
        elif subsampling == (2, 0):
            offsets = (2.5, 0)  # 411
        elif subsampling == (2, 2):
            offsets = (2.5, 1)  # 410
    return offsets


_to_444Presets = ["znedi3", "nnedi3", "nnedi3cl", "eedi3", "eedi3cl"]


def to_444(
    clip: vs.VideoNode,
    chroma_doubler: Optional[Union[Callable, str]] = None,
    chromaloc: Union[str, int, bool, Tuple[float, float]] = True,
    **doubler__args,
) -> vs.VideoNode:
    from vsutil import split, join

    if clip.format.subsampling_h == clip.format.subsampling_w == 0:
        return clip
    if isinstance(chroma_doubler, str):
        chroma_doubler = chroma_doubler.lower()
    if chroma_doubler in [core.znedi3.nnedi3, "znedi3", None]:
        doubler__args = dict(nsize=0, nns=4, qual=2).update(doubler__args)
    elif chroma_doubler in [core.nnedi3.nnedi3, "nnedi3"]:
        doubler__args = dict(nsize=0, nns=3, qual=2).update(doubler__args)
        chroma_doubler = core.nnedi3.nnedi3
    elif chroma_doubler in [core.nnedi3cl.NNEDI3CL, "nnedi3cl"]:
        doubler_args = dict(nsize=0, nns=3, qual=2).update(doubler__args)
        chroma_doubler = core.nnedi3cl.NNEDI3CL
    # eedi3_rpow2 just uses eedi3 defaults. may be worth testing if this is best?
    # it's not like you need anything but nnedi3 for chroma though
    elif chroma_doubler == "eedi3":
        chroma_doubler = core.eedi3m.EEDI3
    elif chroma_doubler == "eedi3cl":
        chroma_doubler = core.eedi3m.EEDI3CL
    elif isinstance(chroma_doubler, str):
        raise ValueError(
            f"Unknown doubler {chroma_doubler}. Valid presets are znedi3, "
            "nnedi3, nnedi3cl, eedi3 and eedi3cl. For alternatives, use "
            "chroma_doubler=Function."
        )

    offsets = get_chroma_offsets(clip, chromaloc)

    def _scale(plane: vs.VideoNode) -> vs.VideoNode:
        ssh = clip.format.subsampling_h
        ssw = clip.format.subsampling_w
        # is clip vertical?
        v = True
        # alternate fields
        hf = wf = 1
        for i in reversed(range(max(ssh, ssw))):
            if i < ssh:
                if not v:
                    plane = plane.std.Transpose()
                    v = not v
                plane = chroma_doubler(plane, field=hf, dh=1, **doubler_args)
                plane = core.resize.Bicubic(
                    plane, plane.width, plane.height, src_top=0, src_left=-0.5
                )
                hf ^= 1
            if i < ssw:
                if v:
                    plane = plane.std.Transpose()
                    v = not v
                plane = chroma_doubler(plane, field=wf, dh=1, **doubler_args)
                plane = core.resize.Bicubic(
                    plane, plane.width, plane.height, src_top=0, src_left=-0.5
                )
                wf ^= 1
        if not v:
            plane = plane.std.Transpose()
        if offsets != [0, 0]:
            plane = core.resize.Bicubic(
                plane, plane.width, plane.height, src_top=offsets[0], src_left=offsets[1]
            )
        return plane

    Y, U, V = split(clip)
    return join([Y, _scale(U), _scale(V)])


def make_similar(
    clips: Union[vs.VideoNode, List[vs.VideoNode]],
    reference: vs.VideoNode,
    resizer: Optional[Callable] = None,
    chroma_doubler: Union[Callable, bool] = False,
    **resizer_args,
) -> Union[vs.VideoNode, List[vs.VideoNode]]:

    if not isinstance(clips, list):
        clips = [clips]

    fmt = get_format(reference)
    out = [
        _set_format_internal(clip, fmt, resizer, chroma_doubler, **resizer_args) for clip in clips
    ]
    return out[0] if len(out) == 1 else out


def make_similar_mask(mask: vs.VideoNode, ref: vs.VideoNode) -> vs.VideoNode:
    if mask.format.color_family != vs.GRAY:
        raise ValueError("makeSimilarMask is intended for GRAY mask clips")
    return _set_format_internal(mask, get_format(ref, "gray"))


@overload
def matrix_enum(matrix: str) -> int:
    ...


@overload
def matrix_enum(matrix: int) -> str:
    ...


def matrix_enum(matrix: Union[str, int]) -> Union[str, int]:
    matrices = {
        0: "rgb",
        1: "709",
        2: "unspec",
        # 3 is reserved
        4: "fcc",
        5: "470bg",
        6: "170m",
        7: "240m",
        8: "ycgco",
        9: "2020ncl",
        10: "2020cl",
        12: "chromancl",
        13: "chromacl",
        14: "ictcp",
    }
    if isinstance(matrix, str):
        matrices = {v: k for k, v in matrices.items()}
    if matrix not in matrices.keys():
        raise ValueError(
            f"Matrix {matrix} is not defined by Vapoursynth. See"
            " http://www.vapoursynth.com/doc/functions/resize.html"
        )
    return matrices[matrix]


# aliases
get = get_format
set = set_format
from_str = from_string
