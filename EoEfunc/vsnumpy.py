from typing import Any, Callable, Mapping, Optional, Protocol, Sequence, TypeVar, Union, overload
import vapoursynth as vs
import numpy as np

vsapi_below_4 = vs.__api_version__.api_major < 4

core = vs.core

# region: types

T = TypeVar("T")
SingleAndSequence = Union[T, Sequence[T]]
if vsapi_below_4:
    FramePropsValue = Union[
        SingleAndSequence[int],
        SingleAndSequence[float],
        SingleAndSequence[str],
        SingleAndSequence[vs.VideoNode],
        SingleAndSequence[vs.VideoFrame],
        SingleAndSequence[Callable[..., Any]],
    ]
else:
    FramePropsValue = Union[
        SingleAndSequence[int],
        SingleAndSequence[float],
        SingleAndSequence[str],
        SingleAndSequence[vs.VideoNode],
        SingleAndSequence[vs.VideoFrame],
        SingleAndSequence[vs.AudioNode],
        SingleAndSequence[vs.AudioFrame],
        SingleAndSequence[Callable[..., Any]],
    ]

# endregion: types


class VideoFrameArray:
    def __init__(self, frame: vs.VideoFrame) -> None:
        self.planes: list[np.ndarray] = VideoFrameArray._construct_arrays(frame)
        self.format: vs.VideoFormat = frame.format
        self.props: Mapping[str, FramePropsValue] = dict(frame.props)

    @property
    def width(self):
        return self.planes[0].shape[1]

    @property
    def height(self):
        return self.planes[0].shape[0]

    # TODO: Allow direct (writable) access to frame data
    # @overload
    # def __getitem__(self, key: int) -> np.ndarray:
    #     ...

    # @overload
    # def __getitem__(self, key: slice) -> Union[np.ndarray, list[np.ndarray]]:
    #     ...

    # def __getitem__(self, key: Union[int, slice]) -> Union[np.ndarray, list[np.ndarray]]:
    #     if isinstance(key, (int, slice)):
    #         return self.planes[key]
    #     else:
    #         raise IndexError

    @staticmethod
    def _construct_arrays(frame: vs.VideoFrame) -> list[np.ndarray]:
        if vsapi_below_4:
            arrays = []
            for plane in range(frame.format.num_planes):
                plane_array = np.array(frame.get_read_array(plane), copy=False)
                arrays.append(plane_array.reshape(list(plane_array.shape) + [1]))
            return arrays
        else:
            return [np.array(plane) for plane in frame]

    def write_to(self, src_frame: vs.VideoFrame, copy_src_props: bool = False) -> vs.VideoFrame:
        dst_frame = src_frame.copy()
        if dst_frame.width != self.width or dst_frame.height != self.height:
            raise vs.Error(
                f"vsnumpy: Attempted to write frame with dimensions {self.width}x{self.height} to"
                f" frame of size {dst_frame.width}x{dst_frame.height}."
            )
        if (df := dst_frame.format.name) != (sf := self.format.name):
            raise vs.Error(f"vsnumpy: Attempted to write {sf} data to a {df} frame.")
        if (dp := dst_frame.format.num_planes) != (sp := len(self.planes)):
            raise vs.Error(f"vsnumpy: Attempted to write {sp} planes to a frame with {dp} planes.")
        for plane_idx in range(dst_frame.format.num_planes):
            if vsapi_below_4:
                plane_array = np.asarray(dst_frame.get_write_array(plane_idx))
            else:
                plane_array = np.asarray(dst_frame[plane_idx])
            np.copyto(plane_array, self.planes[plane_idx])
            del plane_array
        if not copy_src_props:
            for key in dst_frame.props.keys():
                del dst_frame.props[key]
        for key, value in self.props.items():
            dst_frame.props[key] = value
        return dst_frame


class ArrayEvalFunction(Protocol):
    def __call__(
        self, n: int, fa: Union[VideoFrameArray, list[VideoFrameArray]]
    ) -> VideoFrameArray:
        ...


class SingleArrayEvalFunction(ArrayEvalFunction, Protocol):
    def __call__(self, n: int, fa: VideoFrameArray) -> VideoFrameArray:
        ...


class MultiArrayEvalFunction(ArrayEvalFunction, Protocol):
    def __call__(self, n: int, fa: list[VideoFrameArray]) -> VideoFrameArray:
        ...


@overload
def array_eval(
    clips: vs.VideoNode, func: SingleArrayEvalFunction, output: Optional[vs.VideoNode]
) -> vs.VideoNode:
    ...


@overload
def array_eval(
    clips: list[vs.VideoNode], func: MultiArrayEvalFunction, output: Optional[vs.VideoNode]
) -> vs.VideoNode:
    ...


def array_eval(
    clips: Union[vs.VideoNode, list[vs.VideoNode]],
    func: ArrayEvalFunction,
    output: Optional[vs.VideoNode] = None,
) -> vs.VideoNode:
    def process(n: int, f: list[vs.VideoFrame]):
        try:
            fa_in = [VideoFrameArray(frame) for frame in f[:-1]]
            if len(fa_in) == 1:
                fa_in = fa_in[0]
            fa_out: VideoFrameArray = func(n=n, fa=fa_in)
            f_out = fa_out.write_to(f[-1].copy())
        except Exception as e:
            # TODO: make this actually not awful (thanks python)
            raise type(e)(f"Error occured during processing of frame {n}: {e.args[0]}") from e
        return f_out

    if not isinstance(clips, list):
        clips = [clips]
    output = output if output is not None else clips[0]
    return core.std.ModifyFrame(output, [*clips, output], process)
