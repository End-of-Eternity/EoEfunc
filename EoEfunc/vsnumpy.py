from typing import Optional, Protocol
import vapoursynth as vs
import numpy as np

core = vs.core


def frame_to_array(frame: vs.VideoFrame) -> np.ndarray:
    frame_array = []
    for plane in range(frame.format.num_planes):
        plane_array = np.array(frame.get_read_array(plane), copy=False)
        frame_array.append(plane_array.reshape(list(plane_array.shape) + [1]))
    return np.concatenate(frame_array, axis=2)


class ArrayEvalFunction(Protocol):
    def __call__(self, n: int, array: np.ndarray) -> np.ndarray:
        ...


def array_eval(
    clip: vs.VideoNode, func: ArrayEvalFunction, output: Optional[vs.VideoNode] = None
) -> vs.VideoNode:
    if clip.format.subsampling_h != 0 or clip.format.subsampling_w != 0:
        raise ValueError("Subsampling is not supported")

    def process(n: int, f: list[vs.VideoFrame]):
        f_in = f[0]
        array_in = frame_to_array(f_in)
        array_out = func(n=n, array=array_in)
        f_out = f[1].copy()
        for plane_idx in range(f_out.format.num_planes):
            plane_array = np.asarray(f_out.get_write_array(plane_idx))
            np.copyto(plane_array, array_out[:, :, plane_idx])
            del plane_array
        for key in f_out.props.keys():
            del f_out.props[key]
        for key, value in f[0].props.items():
            f_out.props[key] = value
        return f_out

    output = output if output is not None else clip
    return core.std.ModifyFrame(output, [clip, output], process)
