from functools import partial
import sys
from typing import Callable, Dict, List, Optional, Union
from enum import IntEnum, Enum
import warnings
import vapoursynth as vs

core = vs.core


class GlobalMode(IntEnum):
    IGNORE = 0
    READ = 1
    WRITE = 2
    WRITEONLY = 3


class Pel(IntEnum):
    FULL = 1
    HALF = 2
    QUARTER = 4


class SubPel(IntEnum):
    BILINEAR = 0
    BICUBIC = 1
    WIENER = 2
    NNEDI3 = 3


class Prefilter(IntEnum):
    LIGHT = 0
    MEDIUM = 1
    STRONG = 2
    DFTTEST = 3
    KNLMEANS = 4


class MVMode(Enum):
    INTEGER = 0
    FLOAT_OLD = 1
    FLOAT_NEW = 2

    @property
    def namespace(self):
        if self == MVMode.INTEGER:
            return core.mv
        else:
            return core.mvsf

    @property
    def Super(self):
        return self.namespace.Super

    @property
    def Analyse(self):
        if self == MVMode.INTEGER:
            return self.namespace.Analyse
        else:
            return self.namespace.Analyze

    @property
    def Recalculate(self):
        return self.namespace.Recalculate

    def Degrain(self, radius: Optional[int] = None) -> Callable:
        if radius is None:
            if self == MVMode.FLOAT_NEW:
                return self.namespace.Degrain
            else:
                raise ValueError(f"{self.name}.Degrain needs radius")
        elif (self == MVMode.FLOAT_OLD and 1 <= radius <= 24) or (
            self == MVMode.INTEGER and 1 <= radius <= 3
        ):
            return getattr(self.namespace, f"Degrain{radius}")
        raise ValueError(f"{self.name}.Degrain doesn't support a radius of {radius}")


# accessable outside of denoise.py without being pepegastoopid
global_mv_vectors: Dict[str, Optional[Union[vs.VideoNode, MVMode]]] = {"MVMode": None}


def prefilter_dfttest(clip: vs.VideoNode):
    from vsutil import get_y, scale_value

    peak = 1 if clip.format.sample_type == vs.FLOAT else 2 << clip.format.bits_per_sample
    i = scale_value(16, 8, clip.format.bits_per_sample)
    j = scale_value(75, 8, clip.format.bits_per_sample)
    expr = f"x {i} < {peak} x {j} > 0 {peak} x {i} - {peak} {j} {i} - / * - ? ?"
    return core.std.MaskedMerge(
        clip.dfttest.DFTTest(tbsize=1, slocation=[0.0, 4.0, 0.2, 9.0, 1.0, 15.0]),
        clip,
        get_y(clip).std.Expr(expr=[expr]),
    )


def CMDegrain(
    src: vs.VideoNode,
    tr: int = 2,
    thSAD: int = 200,
    thSADC: Optional[int] = None,
    refine: int = 3,
    highprecision: bool = False,
    contrasharp: Union[bool, vs.VideoNode] = False,
    freq_merge: Optional[Union[bool, vs.VideoNode]] = None,
    planes: Optional[Union[int, List[int]]] = None,
    globalmode: GlobalMode = GlobalMode.IGNORE,
    prefilter: Optional[Union[Prefilter, vs.VideoNode]] = None,
    pel: Optional[Pel] = None,
    subpel: SubPel = SubPel.WIENER,
    truemotion: Optional[bool] = None,
):
    from .misc import ContraSharpening
    from . import format
    from .frequencies import merge_frequency
    from havsfunc import MinBlur, DitherLumaRebuild
    from vsutil import scale_value

    input_clip = src

    if input_clip.format is None:
        raise ValueError("Variable format is horrible, go away")
    elif input_clip.format.color_family not in [vs.GRAY, vs.YUV]:
        raise ValueError("This function is intended for Gray or YUV clips")

    for clip in [prefilter, contrasharp, freq_merge]:
        if isinstance(clip, vs.VideoNode) and clip.format != src.format:
            raise ValueError("All input clip formats must be identical to source")

    if contrasharp and src.format.sample_type == vs.FLOAT:
        raise vs.Error("ContraSharpening is not supported for float input")

    if planes is not None and isinstance(planes, int):
        planes = [planes]

    chroma = True
    if input_clip.format.color_family == vs.GRAY:
        if planes is not None and planes != [0]:
            raise ValueError("Can't denoise secondary planes of a gray input clip")
        planes = [0]
        chroma = False
    elif planes is None:
        planes = [0, 1, 2]

    if planes == [0, 1, 2]:
        mvplane = 4
    elif len(planes) == 1:
        mvplane = planes[0]
    elif planes == [1, 2]:
        mvplane = 3
    else:
        raise ValueError(
            "planes must be either all planes, an individual plane, or both chroma planes."
        )

    if thSADC is None:
        thSADC = thSAD // 2

    if freq_merge is None:
        freq_merge = thSAD >= 200

    isHD = input_clip.width > 1024 or input_clip.height > 576

    if pel is None:
        pel = Pel.FULL if isHD else Pel.HALF
    if pel == Pel.FULL:
        subpel = min(subpel, SubPel.WIENER)

    blksize = max(2 ** (refine + 2), 16 if isHD else 8)
    if truemotion is None:
        truemotion = not isHD

    limit = scale_value(255, 8, input_clip.format.bits_per_sample)

    if refine > 6:
        raise ValueError("refine > 6 is not supported")
    elif highprecision or refine == 6 or tr > 3:
        input_clip = format.set(input_clip, "s")
        mvmode = MVMode.FLOAT_NEW
        if not hasattr(core, "mvsf"):
            raise ImportError(
                "Missing mvsf. Please grab https://github.com/IFeelBloated/vapoursynth-mvtools-sf"
            )
        if not hasattr(core.mvsf, "Degrain"):
            if tr > 24:
                raise ImportError(
                    "If you're mad enough to want to use tr > 24, you're going to need to build the"
                    " master branch of https://github.com/IFeelBloated/vapoursynth-mvtools-sf"
                )
            print(
                "CMDegrain: Using older mvsf. Building"
                " https://github.com/IFeelBloated/vapoursynth-mvtools-sf from master is"
                " recommended",
                file=sys.stderr,
            )
            mvmode = MVMode.FLOAT_OLD
    else:
        if not hasattr(core, "mv"):
            raise ImportError(
                "Missing mvtools. Please grab https://github.com/dubhater/vapoursynth-mvtools"
            )
        mvmode = MVMode.INTEGER

    if globalmode != GlobalMode.IGNORE:
        global global_mv_vectors
        vectors = global_mv_vectors
        if globalmode == GlobalMode.READ and vectors["MVMode"] != mvmode:
            raise ValueError(
                f"Unable to read vectors from global dictionary. Expected mvmode {mvmode}, found"
                f" {vectors['MVMode']}"
            )
        else:
            vectors["MVMode"] = mvmode
    else:
        vectors = {}

    if globalmode != GlobalMode.READ:
        resample_input = True
        if prefilter is None:
            prefilter = input_clip
            resample_input = False
        elif isinstance(prefilter, vs.VideoNode):
            prefilter = format.make_similar(prefilter, input_clip)
        elif prefilter == Prefilter.DFTTEST:
            prefilter = prefilter_dfttest(input_clip)
        elif prefilter == Prefilter.KNLMEANS:
            knlm_args = dict(d=1, a=1, h=7)
            if chroma:
                # workaround for subsampled input
                if input_clip.format.subsampling_w > 0 or input_clip.format.subsampling_h > 0:
                    prefilter = input_clip.knlm.KNLMeansCL(
                        channels="y", **knlm_args
                    ).knlm.KNLMeansCL(channels="uv", **knlm_args)
                else:
                    prefilter = input_clip.knlm.KNLMeansCL(channels="yuv", **knlm_args)
            else:
                prefilter = input_clip.knlm.KNLMeansCL(**knlm_args, channels="y")
        else:
            prefilter = MinBlur(input_clip, r=prefilter.value, planes=planes)

        # Default Auto-Prefilter - Luma expansion TV->PC
        # (up to 16% more values for motion estimation)
        # had to add a workaround here. DitherLumaRebuild looks like it
        # *should* support float input, but also looks like value scaling
        # is incorrect. Just make sure to only pass it integer clips.
        rebuild_partial = partial(DitherLumaRebuild, s0=1.0, c=0.0625, chroma=chroma)
        prefilter = format.process_as(prefilter, rebuild_partial, "16")
    else:
        prefilter = input_clip

    super_args = dict(hpad=blksize, vpad=blksize, pel=pel)
    common_args = dict(search=4, chroma=chroma, truemotion=truemotion)
    recalculate_args = dict(thsad=thSAD, dct=0, **common_args)

    if subpel == SubPel.NNEDI3:
        import nnedi3_resample as nnrs

        cshift = 0.25 if pel == 2 else 0.375
        nnargs = dict(
            target_width=input_clip.width * pel,
            target_height=input_clip.height * pel,
            src_left=cshift,
            src_top=cshift,
            nns=4,
        )
        pclip = pclip2 = nnrs.nnedi3_resample(prefilter, **nnargs)
        super_search = mvmode.Super(
            prefilter, chroma=chroma, rfilter=4, pelclip=pclip, **super_args
        )
        if globalmode != GlobalMode.READ and resample_input:
            pclip2 = nnrs.nnedi3_resample(input_clip, **nnargs)
    else:
        super_search = mvmode.Super(
            prefilter, chroma=chroma, sharp=subpel.value, rfilter=4, **super_args
        )

    if globalmode != GlobalMode.READ:
        if subpel == 3:
            super_render = mvmode.Super(
                input_clip, levels=1, chroma=planes != [0], pelclip=pclip2, **super_args
            )
            if refine:
                refine_super = mvmode.Super(
                    prefilter, levels=1, chroma=chroma, pelclip=pclip, **super_args
                )
        else:
            super_render = mvmode.Super(
                input_clip,
                levels=1,
                chroma=planes != [0],
                sharp=subpel,
                **super_args,
            )
            if refine:
                refine_super = mvmode.Super(
                    prefilter, levels=1, chroma=chroma, sharp=subpel, **super_args
                )

        analyse_args = dict(
            super=super_search, blksize=blksize, overlap=blksize // 2, dct=0, **common_args
        )
        if mvmode != MVMode.FLOAT_NEW:
            for i in range(1, tr + 1):
                vectors[f"bv{i}"] = mvmode.Analyse(isb=True, delta=i, **analyse_args)
                vectors[f"fv{i}"] = mvmode.Analyse(isb=False, delta=i, **analyse_args)
                for j in range(refine):
                    recalculate_args.update(
                        blksize=blksize / 2 ** j, overlap=blksize / 2 ** (j + 1)
                    )
                    vectors[f"bv{i}"] = mvmode.Recalculate(
                        refine_super, vectors[f"bv{i}"], **recalculate_args
                    )
                    vectors[f"fv{i}"] = mvmode.Recalculate(
                        refine_super, vectors[f"fv{i}"], **recalculate_args
                    )
        else:
            vectors["mvmulti"] = mvmode.Analyse(radius=tr, **analyse_args)
            for i in range(refine):
                recalculate_args.update(blksize=blksize / 2 ** i, overlap=blksize / 2 ** (i + 1))
                vectors["mvmulti"] = mvmode.Recalculate(
                    refine_super, vectors["mvmulti"], **recalculate_args
                )
    else:
        super_render = super_search

    if globalmode == GlobalMode.WRITEONLY:
        return

    # Finally, MDegrain
    degrain_args = dict(
        plane=mvplane,
        thscd1=400,
        thscd2=130,
        thsad=[thSAD, thSADC, thSADC],
        limit=limit,
    )
    if mvmode == MVMode.INTEGER:
        degrain_args.update(thsad=thSAD, thsadc=thSADC)
    if mvmode != MVMode.FLOAT_NEW:
        degrain_vectors = []
        for i in range(1, tr + 1):
            degrain_vectors.append(vectors[f"bv{i}"])
            degrain_vectors.append(vectors[f"fv{i}"])
        output: vs.VideoNode = mvmode.Degrain(tr)(
            input_clip, super_render, *degrain_vectors, **degrain_args
        )
    else:
        output = mvmode.Degrain()(input_clip, super_render, vectors["mvmulti"], **degrain_args)

    output = format.make_similar(output, src)

    if freq_merge:
        if isinstance(freq_merge, vs.VideoNode):
            output = merge_frequency(freq_merge, output)
        else:
            output = merge_frequency(src, output)

    if contrasharp:
        if isinstance(contrasharp, vs.VideoNode):
            output = ContraSharpening(output, contrasharp, planes=planes)
        else:
            output = ContraSharpening(output, src, planes=planes)

    return output


def BM3D(
    src: vs.VideoNode,
    sigma: Union[float, List[float], List[List[float]]] = 3,
    radius: Union[Optional[int], List[Optional[int]]] = 1,
    profile: Union[Optional[str], List[Optional[str]]] = "lc",
    refine: int = 1,
    pre: Optional[vs.VideoNode] = None,
    ref: Optional[vs.VideoNode] = None,
    matrix: Optional[str] = None,
    full_range: Optional[bool] = None,
    skip_basic: bool = False,
    CUDA: Optional[Union[bool, List[bool]]] = None,
    chroma: bool = True,
    device_id: int = 0,
    fast: bool = True,
    **kwargs,
):
    from . import format
    from vsutil.clips import get_y, split, join
    import sys

    input_original = src

    def is_gray(clip: vs.VideoNode) -> bool:
        return clip.format.color_family == vs.GRAY

    if CUDA is None:
        CUDA = [False, False]
        if hasattr(core, "bm3dcuda"):
            CUDA = [True, True]
    elif isinstance(CUDA, bool):
        CUDA = [CUDA, CUDA]

    if not isinstance(sigma, list):
        sigma = [sigma]
    if not all(isinstance(elem, list) for elem in sigma):
        sigma = [sigma, sigma]
    sigma: List[List[float]] = [(s + [s[-1]] * 3)[:3] for s in sigma]
    for i in [0, 1]:
        # multiply luma sigmas by 0.8, and chroma by 0.95, if cuda
        # this seemed to give closer results to fully non cuda.
        if CUDA[i]:
            sigma[i] = [sigma[i][0] * 0.8] + [s * 0.95 for s in sigma[i][1:]]

    if not isinstance(profile, list):
        profile = [profile, profile]
    elif profile[1] is None:
        profile = [profile[0], profile[0]]

    if isinstance(radius, list) and radius[1] is None:
        radius = [radius[0], radius[0]]
    elif not isinstance(radius, list):
        radius = [radius, radius]
    for i, r in enumerate(radius):
        if r is None:
            radius[i] = {"fast": 1, None: 1, "lc": 2, "np": 3, "high": 4, "vn": 4}[profile[i]]

    if not (sigma[0][1] + sigma[0][2] + sigma[1][1] + sigma[1][2]):
        src = get_y(src)

    if src.format.color_family == vs.GRAY:
        chroma = False

    # guess the range based on either frame size (where unspec.) or from frame props
    if full_range is None:
        full_range = False if src.format.color_family in [vs.GRAY, vs.YUV] else True
        props = src.get_frame(0).props
        if "_ColorRange" in props:
            full_range = not props["_ColorRange"]

    if is_gray(src):
        matrix = None
    elif matrix is None:
        matrix = format.guess_matrix(src)

    def to_opp(v):
        v = format.set(
            v,
            formatstr="rgbs" if not is_gray(v) else "s",
            resizer=core.resize.Bicubic,
            filter_param_a=0,
            filter_param_b=0.5,
            matrix_in_s=matrix,
        )
        return core.bm3d.RGB2OPP(v, 1) if not is_gray(v) else v

    if CUDA[0] and pre is not None:
        warnings.warn(
            "BM3D: WARN --> BM3DCUDA doesn't accept a pre for the basic estimate, ignoring"
        )

    clips = {k: to_opp(v) for k, v in dict(src=src, pre=pre, ref=ref).items() if v is not None}

    if all(c not in clips.keys() for c in ["pre", "ref"]):
        clips["pre"] = clips["src"]

    # region profiles
    # fmt: off
    CUDA_BASIC_PROFILES: Dict[Union[str, None], Dict[str, int]] = {
        "fast": dict(block_step=8, bm_range=9,  ps_range=4), # noqa
        "lc":   dict(block_step=6, bm_range=9,  ps_range=4), # noqa
        "np":   dict(block_step=4, bm_range=16, ps_range=5), # noqa
        "high": dict(block_step=2, bm_range=16, ps_range=7), # noqa
        "vn":   dict(block_step=4, bm_range=16, ps_range=5), # noqa
        None: {},
    }

    CUDA_FINAL_PROFILES: Dict[Union[str, None], Dict[str, int]] = {
        "fast": dict(block_step=7, bm_range=9,  ps_range=5), # noqa
        "lc":   dict(block_step=5, bm_range=9,  ps_range=5), # noqa
        "np":   dict(block_step=3, bm_range=16, ps_range=6), # noqa
        "high": dict(block_step=2, bm_range=16, ps_range=8), # noqa
        # original vn profile for final uses a block size of 11, where cuda only supports 8. vn used
        # step 11, and 11-4 = an overlap of 7, meaning the closest I can really get is a step of 1.
        # still probably isn't ideal, a larger block size would be far better for noisy content.
        "vn":   dict(block_step=1, bm_range=16, ps_range=6), # noqa
        None: {},
    }
    # fmt: on
    # endregion profiles

    if any(CUDA):
        if CUDA[1] and profile[1] == "vn":
            print(
                "BM3D: WARN --> BM3DCUDA does not directly support the vn profile for final"
                " estimate, Emulating nearest parameters.",
                file=sys.stderr,
            )
        cudaargs = [CUDA_BASIC_PROFILES[profile[0]], CUDA_FINAL_PROFILES[profile[1]]]
        for i in [0, 1]:
            if radius[i]:
                if profile[i] == "fast":
                    cudaargs[i].update(bm_range=7)
                elif profile[i] in ["np", "vn"]:
                    cudaargs[i].update(bm_range=12)
        for args in cudaargs:
            args.update(chroma=chroma, device_id=device_id, fast=fast)

    if not (skip_basic or "ref" in clips.keys()):
        if not CUDA[0]:
            basicargs = dict(
                input=clips["src"], ref=clips["pre"], profile=profile[0], sigma=sigma[0], matrix=100
            )
            if not radius[0]:
                basic = core.bm3d.Basic(**basicargs)
            else:
                basic = core.bm3d.VBasic(radius=radius[0], **basicargs).bm3d.VAggregate(
                    radius[0], 1
                )
        else:
            basic = core.bm3dcuda.BM3D(
                clips["src"], sigma=sigma[0], radius=radius[0], **cudaargs[0]
            )
            basic = core.bm3d.VAggregate(basic, radius[0], 1) if radius[0] else basic
    else:
        basic = clips["ref"] or clips["src"]

    final = basic
    for _ in range(refine):
        if not CUDA[1]:
            finalargs = dict(
                input=clips["src"], ref=final, profile=profile[1], sigma=sigma[1], matrix=100
            )
            if not radius[1]:
                final = core.bm3d.Final(**finalargs)
            else:
                final = core.bm3d.VFinal(**finalargs, radius=radius[1]).bm3d.VAggregate(
                    radius[1], 1
                )
        else:
            final = core.bm3dcuda.BM3D(
                clips["src"], ref=final, sigma=sigma[1], radius=radius[1], **cudaargs[1]
            )

            final = core.bm3d.VAggregate(final, radius[1], 1) if radius[1] else final

    out = core.bm3d.OPP2RGB(final, 1) if not is_gray(final) else final
    out = format.make_similar(
        out,
        src,
        resizer=core.resize.Bicubic,
        filter_param_a=0,
        filter_param_b=0.5,
        matrix_s=matrix,
    )
    if input_original.format.num_planes == 3:
        # we didn't upload chroma
        if src.format.color_family == vs.GRAY:
            out = join([out] + split(input_original)[1:], input_original.format.color_family)
        else:
            # bm3dcuda seems to set the luma to garbage if it isnt processed
            for i in [0, 1]:
                if CUDA[i] and not sigma[i][0] and (sigma[i][1] + sigma[i][2]):
                    out = join(
                        [get_y(input_original)] + split(out)[1:], input_original.format.color_family
                    )

    return out


CMDE = CMDegrain


def prefilter3(clip):
    warnings.warn("prefilter3: deprecated, use prefilter_dfttest instead")
    return prefilter_dfttest(clip)
