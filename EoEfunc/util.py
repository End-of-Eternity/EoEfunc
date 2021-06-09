import vapoursynth as vs

core = vs.core


def chunked_cocktail_shaker(upper, lower=0, start=None, blocksize=100):
    from itertools import chain, zip_longest
    import math

    if start is None:
        start = (upper + lower) // 2 - blocksize // 2

    if lower >= upper:
        raise ValueError("lower must be smaller than upper")

    if blocksize <= 0:
        raise ValueError("blocksize must be more than 0")

    fwd = [
        range(
            max(start + blocksize * n, lower),
            min(start + blocksize * (n + 1), upper),
        )
        for n in range(math.ceil((upper - start) / blocksize))
    ]

    bwd = [None] + [
        range(
            max(lower + blocksize * n - blocksize // 2 + 1, lower),
            min(lower + blocksize * (n + 1) - blocksize // 2 + 1, start),
        )
        for n in range(math.ceil((start - lower) / blocksize) - 1, lower - 2, -1)
    ]

    return chain(*[x for x in chain.from_iterable(zip_longest(fwd, bwd)) if x is not None])


def range_wrap(start, max, min=0, step=1):
    from itertools import chain

    if min >= max or min > start > max:
        raise ValueError("Min must be smaller than max, and start must be between the two")
    return chain(range(start, max, step), range(min, start, step))


def peak(clip: vs.VideoNode):
    return (1 << clip.format.bits_per_sample) - 1 if clip.format.sample_type == vs.INTEGER else 1


def neutral(clip: vs.VideoNode):
    return peak(clip) // 2 if clip.format.sample_type == vs.INTEGER else 0.5
