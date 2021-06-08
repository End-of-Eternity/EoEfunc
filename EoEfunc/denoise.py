from typing import Any, Dict, List, Optional, Union
import vapoursynth as vs

core = vs.core


def prefilter3(clip: vs.VideoNode):
    from .util import peak
    from havsfunc import scale
    from vsutil import get_y

    peak = peak(clip)
    i = scale(16, peak)
    j = scale(75, peak)
    expr = f"x {i} < {peak} x {j} > 0 {peak} x {i} - {peak} {j} {i} - / * - ? ?"
    return core.std.MaskedMerge(
        clip.dfttest.DFTTest(tbsize=1, slocation=[0.0, 4.0, 0.2, 9.0, 1.0, 15.0]),
        clip,
        get_y(clip).std.Expr(expr=[expr]),
    )


def CMDegrain(
    # This needs to be cleaned up. Everything past `planes` (excluding prefilter) can easily
    # be moved into kwargs. the rest... idk
    src: vs.VideoNode,
    tr: int = 2,  # V W X Y Z
    thSAD: int = 300,
    thSADC: Optional[int] = None,
    RefineMotion: int = 3,
    highPrecision: bool = False,
    contrasharp: Optional[Union[int, bool]] = None,
    contraClip: Optional[vs.VideoNode] = None,
    freqMerge: Union[bool, vs.VideoNode] = True,
    slocation: Optional[List[int]] = None,
    interlaced: bool = False,
    tff: Optional[bool] = None,
    planes: Union[int, List[int]] = [0, 1, 2],
    Globals: int = 0,
    prefilter: Optional[Union[int, vs.VideoNode]] = None,
    pel: Optional[int] = None,
    subpixelInterpolation: int = 2,
    mfilter: Optional[vs.VideoNode] = None,
    blksize: Optional[int] = None,
    search: int = 4,
    truemotion: Optional[bool] = None,
    dct: int = 0,
    limit: int = 255,
    limitc: Optional[int] = None,
    thSCD1: int = 400,
    thSCD2: int = 130,
    chroma: bool = True,
    Str: float = 1.0,
    Amp: float = 0.0625,
):
    from .util import peak
    from .misc import ContraSharpening
    from . import format
    from .frequencies import _slocation, extract_frequency, merge_frequency
    from havsfunc import scale, MinBlur, DitherLumaRebuild, LSFmod

    inputClip = src

    if not isinstance(inputClip, vs.VideoNode):
        raise vs.Error("CMDegrain: This is not a clip")

    if inputClip.format.color_family == vs.GRAY:
        planes = [0]
        chroma = False

    if thSADC is None:
        thSADC = thSAD // 2

    if inputClip.format.sample_type == vs.FLOAT or tr > 3 or highPrecision:
        if not hasattr(core, "mvsf"):
            raise vs.Error(
                "CMDegrain: mvsf is required for float input, tr > 3, or highPrecision."
                " https://github.com/IFeelBloated/vapoursynth-mvtools-sf"
            )
        if not inputClip.format.sample_type == vs.FLOAT:
            inputClip = format.set_format(inputClip, "s")
        mv = core.mvsf
        mvint = False
    else:
        if not hasattr(core, "mv"):
            raise vs.Error(
                "CMDegrain: mvtools is required for integer input, and tr <= 3. "
                "https://github.com/dubhater/vapoursynth-mvtools"
            )
        mv = core.mv
        mvint = True

    planes = [planes] if isinstance(planes, int) else sorted(planes)
    if planes == [0, 1, 2]:
        mvplane = 4
    elif len(planes) == 1:
        mvplane = planes[0]
    elif planes == [1, 2]:
        mvplane = 3
    else:
        raise vs.Error(
            "CMDegrain: planes must be either all planes, "
            "an individual plane, or both chroma planes."
        )

    peak = peak(inputClip)

    # Globals 0 means just do everything
    readGlobals = Globals == 1
    outputGlobals = Globals == 3

    isHD = inputClip.width > 1024 or inputClip.height > 576

    if pel is None:
        pel = 1 if isHD else 2
    # subpixelInterpolation (originally just subpixel) is passed to mvtools as the "sharp" param
    # it has three modes, 0, 1 and 2. This version of SMDe adds a fourth mode, 3.
    # since it's calculated here, and not by mvtools, we need to make sure it doesnt execute
    # if pel is less than 2, since then theres no subpixel estimation anyway, and we'd just
    # be wasting cpu time
    if pel < 2:
        subpixelInterpolation = min(subpixelInterpolation, 2)

    if blksize is None:
        blksize = max(2 ** (RefineMotion + 2), 16 if isHD else 8)
    if truemotion is None:
        truemotion = not isHD

    limit = scale(limit, peak)
    if limitc is None:
        limitc = limit
    else:
        limitc = scale(limitc, peak)

    if interlaced:
        if not isinstance(tff, bool):
            raise vs.Error(
                "CMDegrain: when filtering interlaced inputs, tff must be specified as a boolean"
            )
        inputClip = core.std.SeparateFields(inputClip, tff)

        def check_seperated(clip, name):
            if isinstance(clip, vs.VideoNode) and clip.height == inputClip.height * 2:
                raise vs.Error(
                    f"CMDegrain: when filtering interlaced inputs, {name} fields must be seperated."
                )

        check_seperated(mfilter, "mfilter")
        check_seperated(prefilter, "prefilter")
        check_seperated(freqMerge, "freqMerge")

    def check_similar(clipa: Union[vs.VideoNode, Any], clipb: vs.VideoNode, name: str):
        if isinstance(clipa, vs.VideoNode) and (
            clipa.height != clipb.height
            or clipa.width != clipb.width
            or format.get_format(clipa, "8") != format.get_format(clipb, "8")
        ):
            raise vs.Error(
                f"CMDegrain: inputClip clip and {name} must have the same "
                "dimensions, color family and subsampling"
            )

    check_similar(mfilter, inputClip, "mfilter")
    check_similar(prefilter, inputClip, "prefilter")
    check_similar(contraClip, src, "contraClip")
    check_similar(freqMerge, src, "freqMerge")

    if contrasharp is None:
        contrasharp = contraClip is not None
    elif not (isinstance(contrasharp, bool) or isinstance(contrasharp, int)):
        raise vs.Error("CMDegrain: 'contrasharp' only accepts bool and integer inputs")
    if contrasharp:
        if src.format.sample_type == vs.FLOAT:
            raise vs.Error("CMDegrain: ContraSharpening is not supported for float inputClips")
        if contraClip is None:
            contraClip = src
        elif contraClip.format != src.format:
            contraClip = format.make_similar(contraClip, src)

    if not (isinstance(prefilter, vs.VideoNode) or isinstance(prefilter, int) or prefilter is None):
        raise vs.Error("CMDegrain: 'prefilter' only accepts integer and clip inputs")
    elif isinstance(prefilter, vs.VideoNode) and prefilter.format != inputClip.format:
        prefilter = format.make_similar(prefilter, inputClip)

    if not (isinstance(freqMerge, vs.VideoNode) or isinstance(freqMerge, bool)):
        raise vs.Error("CMDegrain: 'freqMerge' only accepts boolean and clip inputs")
    elif isinstance(freqMerge, vs.VideoNode) and freqMerge.format != src.format:
        freqMerge = format.make_similar(freqMerge, src)

    if mfilter is None:
        mfilter = inputClip
    elif mfilter.format != inputClip.format:
        mfilter = format.make_similar(mfilter, inputClip)

    if mvint and RefineMotion == 6:
        raise ValueError(
            "CMDegrain: Integer MVtools only supports up to RefineMotion=5. "
            "Use highPrecision=True for RefineMotion=6"
        )
    elif RefineMotion > 6:
        raise ValueError("CMDegrain: RefineMotion > 6 is not supported")

    if blksize & (blksize - 1) != 0:
        raise ValueError("CMDegrain: blksize must be a power of 2")
    if RefineMotion and blksize < 2 ** (RefineMotion + 2):
        raise ValueError(
            "CMDegrain: blksize needs to be at least 2^(RefineMotion + 2), "
            "when RefineMotion is enabled"
        )
    if mvint and blksize == 256:
        raise ValueError(
            "CMDegrain: Integer MVtools only supports up to blksize=128. "
            "Use highPrecision=True for blksize=256"
        )
    elif blksize > 256:
        raise ValueError("CMDegrain: blksize > 256 is not supported")

    if tr > 24:
        raise ValueError("CMDegrain: mvsf only supports up to tr=24. Any more and you'd be mad.")

    if not chroma and planes != [0]:
        raise ValueError(
            "CMDegrain: Denoising chroma with luma only vectors is bugged"
            " in mvtools and thus unsupported. Either set chroma=True, or planes=0"
        )

    if not readGlobals:
        if isinstance(prefilter, vs.VideoNode):
            pref = prefilter
        elif prefilter is None:
            pref = inputClip
        elif prefilter == 3:
            pref = prefilter3(inputClip)
        elif prefilter >= 4:
            knlm_args = dict(d=1, a=1, h=7)
            if chroma:
                # workaround for subsampled input
                if inputClip.format.subsampling_w > 0 or inputClip.format.subsampling_h > 0:
                    pref = inputClip.knlm.KNLMeansCL(channels="y", **knlm_args).knlm.KNLMeansCL(
                        channels="uv", **knlm_args
                    )
                else:
                    pref = inputClip.knlm.KNLMeansCL(channels="yuv", **knlm_args)
            else:
                pref = inputClip.knlm.KNLMeansCL(**knlm_args, channels="y")
        else:
            pref = MinBlur(inputClip, r=prefilter, planes=planes)

        prefilter = pref

        # Default Auto-Prefilter - Luma expansion TV->PC
        # (up to 16% more values for motion estimation)
        # had to add a workaround here. DitherLumaRebuild looks like it
        # *should* support float input, but also looks like value scaling
        # is incorrect. Just make sure to only pass it integer clips.
        pref = format.set_format(pref, "16")
        pref = DitherLumaRebuild(pref, s0=Str, c=Amp, chroma=chroma)
        pref = format.make_similar(pref, inputClip)
    else:
        pref = inputClip

    # subpixelInterpolation 3
    if subpixelInterpolation == 3:
        import nnedi3_resample as nnrs

        cshift = 0.25 if pel == 2 else 0.375
        nnargs = dict(
            inputClip.width * pel, inputClip.height * pel, src_left=cshift, src_top=cshift, nns=4
        )
        pclip = nnrs.nnedi3_resample(pref, **nnargs)
        if not readGlobals:
            # if prefilter <= -1, inputClip and pref are the same thing, and
            # therefore resampling the inputClip seperately is redundant
            pclip2 = pclip
            if prefilter >= 0:
                pclip2 = nnrs.nnedi3_resample(inputClip, **nnargs)

    # Motion vectors search
    # only use globals if user wants them - global variables are big bad
    # i now regret this
    if Globals > 0:
        vectorDictFunc = globals
    else:
        vectorsDict = {}
        vectorDictFunc = lambda: vectorsDict  # noqa
    super_args = dict(hpad=blksize, vpad=blksize, pel=pel)
    common = dict(search=search, chroma=chroma, truemotion=truemotion)
    recalculate_args = dict(thsad=thSAD, dct=dct, **common)

    if subpixelInterpolation == 3:
        super_search = mv.Super(pref, chroma=chroma, rfilter=4, pelclip=pclip, **super_args)
    else:
        super_search = mv.Super(
            pref, chroma=chroma, sharp=subpixelInterpolation, rfilter=4, **super_args
        )

    if not readGlobals:
        if subpixelInterpolation == 3:
            super_render = mv.Super(
                inputClip, levels=1, chroma=planes != [0], pelclip=pclip2, **super_args
            )
            if RefineMotion:
                Recalculate = mv.Super(pref, levels=1, chroma=chroma, pelclip=pclip, **super_args)
        else:
            super_render = mv.Super(
                inputClip, levels=1, chroma=planes != [0], sharp=subpixelInterpolation, **super_args
            )
            if RefineMotion:
                Recalculate = mv.Super(
                    pref, levels=1, chroma=chroma, sharp=subpixelInterpolation, **super_args
                )

        analyse_args = dict(
            super=super_search, blksize=blksize, overlap=blksize // 2, dct=dct, **common
        )
        for i in range(1, tr + 1):
            i = i * (interlaced + 1)
            vectorDictFunc()[f"bv{i}"] = mv.Analyse(isb=True, delta=i, **analyse_args)
            vectorDictFunc()[f"fv{i}"] = mv.Analyse(isb=False, delta=i, **analyse_args)
            for j in range(RefineMotion):
                recalculate_args.update(
                    blksize=blksize / 2 ** j,
                    blksizev=blksize / 2 ** (j + 1) if interlaced else None,
                    overlap=blksize / 2 ** (j + 1),
                    overlapv=blksize / 2 ** (j + 2) if interlaced else None,
                )
                vectorDictFunc()[f"bv{i}"] = mv.Recalculate(
                    Recalculate, vectorDictFunc()[f"bv{i}"], **recalculate_args
                )
                vectorDictFunc()[f"fv{i}"] = mv.Recalculate(
                    Recalculate, vectorDictFunc()[f"fv{i}"], **recalculate_args
                )
    else:
        super_render = super_search

    if outputGlobals:
        return

    # Finally, MDegrain
    degrain_args = dict(
        plane=mvplane,
        thscd1=thSCD1,
        thscd2=thSCD2,
        thsad=[thSAD, thSADC, thSADC],
        limit=[limit, limitc, limitc],
    )
    if mvint:
        degrain_args.update(thsad=thSAD, thsadc=thSADC, limit=limit, limitc=limitc)
    vectors = []
    for i in range(1, tr + 1):
        i = i * (interlaced + 1)
        vectors.append(vectorDictFunc()[f"bv{i}"])
        vectors.append(vectorDictFunc()[f"fv{i}"])
    output: vs.VideoNode = getattr(mv, f"Degrain{tr}")(
        mfilter, super_render, *vectors, **degrain_args
    )

    output = format.make_similar(output, src)

    if freqMerge and prefilter:
        if slocation is None:
            slocation = _slocation
        if isinstance(freqMerge, vs.VideoNode):
            output = merge_frequency(freqMerge, output, slocation=slocation)
        else:
            prefilter = format.make_similar(prefilter, output)
            output = merge_frequency(extract_frequency(prefilter), output, slocation=slocation)

    if interlaced:
        output = core.std.DoubleWeave(output, tff=tff)[::2]

    if contrasharp:
        if isinstance(contrasharp, bool):
            output = ContraSharpening(output, contraClip, planes=planes)
        else:
            output = LSFmod(
                output,
                strength=contrasharp,
                inputClip=contraClip,
                Lmode=0,
                soothe=False,
                defaults="slow",
            )

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
    from vsutil.clips import get_y
    import sys

    input_original = src

    def is_gray(clip):
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

    if radius is not None and "radius1" in kwargs:
        print("BM3D: WARN --> use radius=[radius1, radius2] instead!", file=sys.stderr)
        radius = kwargs["radius1"]

    if profile is not None and "profile1" in kwargs:
        print("BM3D: WARN --> use profile=[profile1, profile2] instead!", file=sys.stderr)
        radius = kwargs["profile1"]

    set_kwargs = set(kwargs.keys())
    if set_kwargs - set(["radius1", "profile1"]):
        print(
            "BM3D: WARN --> recieved extra args:"
            f" {', '.join(set_kwargs - set(['radius1', 'profile1']))}, ignoring."
        )

    if isinstance(radius, list) and radius[1] is None:
        radius = [radius[0], radius[0]]
    elif not isinstance(radius, list):
        radius = [radius, radius]
    for i, r in enumerate(radius):
        if r is None:
            radius[i] = {"fast": 1, None: 1, "lc": 2, "np": 3, "high": 4, "vn": 4}[profile[i]]

    if not (sigma[0][1] + sigma[0][1]):
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
        raise print("BM3D: WARN --> BM3DCUDA doesn't accept a pre for the basic estimate, ignoring")

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
            ).bm3d.VAggregate(radius[0], 1)
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
            ).bm3d.VAggregate(radius[1], 1)

    out = core.bm3d.OPP2RGB(final, 1) if not is_gray(final) else final
    out = format.make_similar(
        out,
        src,
        resizer=core.resize.Bicubic,
        filter_param_a=0,
        filter_param_b=0.5,
        matrix_s=matrix,
    )
    if not (sigma[0][1] + sigma[0][1]) and input_original.format.num_planes == 3:
        return core.std.ShufflePlanes(
            [out, input_original], [0, 1, 2], input_original.format.color_family
        )
    else:
        return out


def decheckerboard(
    src: vs.VideoNode, mask: Optional[vs.VideoNode], slocation=[0.0, 0, 0.85, 0, 0.9, 512, 1.0, 512]
) -> vs.VideoNode:
    from .frequencies import _dfttest_args
    import kagefunc as kgf
    from vsutil.func import iterate
    from vsutil.clips import get_y, split, join

    clip = src
    if src.format.color_family != vs.GRAY:
        clip = get_y(src)
    clip = core.dfttest.DFTTest(clip, sbsize=9, slocation=slocation, **_dfttest_args)
    if not mask:
        mask = kgf.retinex_edgemask(clip)
        mask = iterate(mask, core.std.Inflate, 2)
    knlm = core.knlm.KNLMeansCL(clip, d=0, a=5, s=1, h=0.45)
    clip = core.std.MaskedMerge(knlm, clip, mask)
    if src.format.color_family != vs.GRAY:
        clip = join([clip] + split(src)[1:], src.format.color_family)
    return clip


CMDE = CMDegrain
