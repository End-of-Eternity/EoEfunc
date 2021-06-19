from __future__ import annotations
from typing import List, Mapping, Sequence, Union
import vapoursynth as vs
import collections

core = vs.core

_dfttest_args = dict(smode=0, sosize=0, sbsize=9, tosize=0, tmode=0, tbsize=1)
_slocation = [0.0, 4, 0.35, 16, 0.4, 512, 1.0, 512]


class SLocation:
    def __init__(self, sigmaLocations: Union[Sequence[float], Mapping[float, float]]) -> None:
        if isinstance(sigmaLocations, collections.Mapping):
            self.frequencies = list(sigmaLocations.keys())
            self.sigmas = list(sigmaLocations.values())
        elif isinstance(sigmaLocations, collections.Sequence):
            if len(sigmaLocations) % 2:
                raise ValueError("sigmaLocations must have an even number of items")
            self.frequencies = list(sigmaLocations[0::2])
            self.sigmas = list(sigmaLocations[1::2])
        else:
            raise ValueError("sigmaLocations must be either a list, or a dict")

        self.boundsCheck()
        self.sort()

    def to_list(self) -> List[float]:
        return [v for pair in zip(self.frequencies, self.sigmas) for v in pair]

    @property
    def inverted(self) -> SLocation:
        return SLocation(
            dict(zip((1 - f for f in reversed(self.frequencies)), list(reversed(self.sigmas))))
        )

    def sorted(self):
        return (list(t) for t in zip(*sorted(zip(self.frequencies, self.sigmas))))

    def sort(self):
        self.frequencies, self.sigmas = self.sorted()

    # doesn't check if the bounds make sense - should be fine because its internal
    # that is, until i forget and screw up
    def _boundsCheck(values, bounds, correct):
        for i in range(len(values)):
            if bounds[0] is not None and values[i] < bounds[0]:
                if correct:
                    values[i] = bounds[0]
                else:
                    raise ValueError("Invalid frequency (0-1) or sigma (0-inf)")
            if bounds[1] is not None and values[i] > bounds[1]:
                if correct:
                    values[i] = bounds[1]
                else:
                    raise ValueError("Invalid frequency (0-1) or sigma (0-inf)")
        return values

    def boundsCheck(self, correct: bool = False) -> None:
        self.frequencies = SLocation._boundsCheck(self.frequencies, [0, 1], correct)
        self.sigmas = SLocation._boundsCheck(self.sigmas, [0, None], correct)

    def interpolated(self, method: str = "linear", resolution: int = 20) -> SLocation:
        try:
            import numpy as np
            from scipy.interpolate import interp1d
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "numpy and scipy are both required for "
                "slocation interpolation. Use "
                "`pip install numpy scipy`"
            )
        interp = interp1d(
            self.frequencies, self.sigmas, method, fill_value="extrapolate", assume_sorted=True
        )
        frequencies = np.linspace(0, 1, resolution, endpoint=True)
        sigmas: np.ndarray = interp(frequencies)
        # overwrite any calculated values with originals
        frequencies, sigmas = zip(
            *dict(list(zip(frequencies, sigmas)) + list(zip(self.frequencies, self.sigmas))).items()
        )
        # force sorted, since the above step may have caused issues
        frequencies, sigmas = (list(t) for t in zip(*sorted(zip(frequencies, sigmas))))
        frequencies = SLocation._boundsCheck(frequencies, [0, 1], True)
        sigmas = SLocation._boundsCheck(sigmas, [0, None], True)

        return SLocation(dict(zip(frequencies, sigmas)))

    def interpolated_ends(self, method: str = "linear") -> SLocation:
        return self.interpolated(method, 2, set)

    def graph(self, width: int = 1920, height: int = 1080) -> vs.VideoNode:
        import matplotlib.pyplot as plt
        import numpy as np
        import io

        def _ndarrayToClip(array: np.ndarray) -> vs.VideoNode:
            height = array.shape[-3]
            width = array.shape[-2]
            planes = array.shape[-1]
            clip = core.std.BlankClip(None, width, height, vs.RGB24 if planes == 3 else vs.GRAY8, 1)

            def convert_func(n, f: vs.VideoFrame):
                fout = f.copy()
                for plane in range(planes):
                    writearray = np.array(fout.get_write_array(plane), copy=False)
                    np.copyto(writearray, array[:, :, plane])
                    del writearray
                return fout

            return core.std.ModifyFrame(clip, clip, convert_func)

        # matplotlib works in inches, so we set width and height
        # to the param/dpi, and the dpi to some arbitrary number
        DPI = 240
        fig = plt.figure(figsize=(width / DPI, height / DPI), dpi=DPI)
        ax = fig.add_subplot(111)
        ax.plot(self.frequencies, self.sigmas, "-")
        ax.set_xlabel("frequency")
        ax.set_ylabel("sigma")

        buffer = io.BytesIO()
        # write our figure to a tempory buffer
        fig.savefig(buffer, format="raw", transparent=False)
        plt.close(fig)
        buffer.seek(0)
        clip = _ndarrayToClip(
            np.reshape(
                np.frombuffer(buffer.getvalue(), dtype=np.uint8), newshape=(height, width, -1)
            )[:, :, :3]
        )
        buffer.close()
        return clip

    intrp = interpolated
    intrp_ends = interpolated_ends


def __dfttest(clip, **dfttest_args):
    keys = dfttest_args.keys()
    if not any(param in keys for param in ["slocation", "ssx", "ssy", "sst"]):
        dfttest_args.update(slocation=_slocation)
    if all(param in keys for param in ["slocation", "sigma"]):
        raise ValueError("sigma cannot be specified with slocation")
    if "sst" in keys and ("tbsize" not in keys or keys["tbsize"] < 2):
        raise ValueError("tbsize should be > 1 when using sst")
    kwargs = {**_dfttest_args, **dfttest_args}
    kwargs = {k: v.to_list() if isinstance(v, SLocation) else v for k, v in kwargs.items()}
    return core.dfttest.DFTTest(clip, **kwargs)


def extract_frequency(clip: vs.VideoNode, **dfttest_args):
    return __dfttest(clip, **dfttest_args)


def insert_frequency(lowf, hi, **dfttest_args):
    hif = core.std.MakeDiff(hi, __dfttest(hi, **dfttest_args))
    clip = core.std.MergeDiff(lowf, hif)
    return clip


def merge_frequency(low, hi, **dfttest_args):
    hif = core.std.MakeDiff(hi, __dfttest(hi, **dfttest_args))
    clip = core.std.MergeDiff(__dfttest(low, **dfttest_args), hif)
    return clip


SLoc = SLocation
extract = extract_frequency
insert = insert_frequency
merge = merge_frequency
