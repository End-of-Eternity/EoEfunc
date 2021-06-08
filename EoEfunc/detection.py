from typing import Iterator, List, Tuple, Union, Optional, overload
from enum import Enum
import vapoursynth as vs

core = vs.core


class SyncMode(Enum):
    NONE = 0
    COPY_SRC = 1


def sync_clips(
    src: vs.VideoNode,
    sync: vs.VideoNode,
    approx_offset: Optional[int] = None,
    thr: Optional[float] = 0.004,
    checks: int = 5,
    allow_fail: int = 1,
    sync_mode: SyncMode = SyncMode.COPY_SRC,
    silent: bool = True,
):
    from vsutil.clips import get_y
    from .format import make_similar
    from .util import chunked_cocktail_shaker

    sync_y = get_y(make_similar(sync, src))
    src_y = get_y(src)

    if thr is None or thr == 0:
        thr = 0

    start_frame = src_y.num_frames // 2
    approx_offset = approx_offset or 0

    offset = _search_nearest(
        core.std.PlaneStats(src_y, sync_y[start_frame] * src_y.num_frames),
        chunked_cocktail_shaker(src_y.num_frames, start_frame + approx_offset),
        thr,
        start_frame,
        False,
        silent,
    )

    failed = 0
    start = max(0, -offset)
    end = min(src_y.num_frames, src_y.num_frames + offset)
    step = (end - start) / (checks - 1)
    for i in (int(start + step * i) for i in range(checks)):
        stats = core.std.PlaneStats(src_y[i], sync_y[i + offset])
        if stats.get_frame(0).props["PlaneStatsDiff"] > thr:
            failed += 1
            if failed > allow_fail:
                raise LookupError(f"Offset of {offset} failed after {failed} checks")

    if sync_mode is SyncMode.NONE:
        return offset, sync
    elif sync_mode is SyncMode.COPY_SRC:
        return offset, (src[:start] + sync + src[end:])[: src.num_frames]


def find_scenes(
    src: vs.VideoNode,
    scenes: Union[vs.VideoNode, List[vs.VideoNode]],
    ordered: bool = False,
    thr: Optional[Union[float, List[float]]] = 0.004,
    strict: Optional[bool] = None,
    index_file: Optional[str] = None,
    overwrite: bool = False,
    silent: bool = True,
):
    from pathlib import Path
    from vsutil.clips import get_y
    from .format import make_similar

    if not isinstance(scenes, list):
        scenes = [scenes]

    if index_file is not None:
        index_file: Path = Path(f"{index_file}.scenes")
        if index_file.is_file() and not overwrite:
            indexes = []
            with index_file.open("r", encoding="utf-8") as f:
                for line in f:
                    indexes.append(int(line))
            return indexes

    if not isinstance(thr, list):
        thr = [thr] * len(scenes)
    elif len(thr) != len(scenes):
        raise ValueError("thr length must be equal to the number of scenes if a list")

    for i in len(thr):
        if thr[i] is None or thr[i] == 0:
            if strict:
                raise ValueError("strict cannot be used with thr = None or 0")
            elif strict is None:
                strict = False
            thr[i] = 0
    if strict is None:
        strict = True

    for scene in scenes:
        if scene.width != src.width or scene.height != src.height:
            raise ValueError("scene dimensions must be the same as the source")

    scenes = make_similar(scenes, src)
    scenes = [get_y(scene) * src.num_frames for scene in scenes]
    src = get_y(src)
    stats_clips = [core.std.PlaneStats(src, scene) for scene in scenes]

    def _save_indexes(indexes: List[int]):
        if index_file is not None:
            with index_file.open("w", encoding="utf-8") as f:
                f.write("\n".join(indexes))

    if ordered:
        start_frame = 0
        indexes: List[int] = []
        for i in range(scenes):
            stats_clip = stats_clips[i][start_frame:]
            if not silent:
                print(f"Searching for scene {i}/{len(scenes)}")
            indexes.append(_search_nearest(stats_clip, range(len(stats_clips)), thr[i], 0, strict, silent))
            start_frame = indexes[i] + 1
    else:
        indexes = _search_nearest(scenes, range(len(stats_clips)), thr, 0, strict, silent)

    _save_indexes(indexes)
    return indexes


# fmt: off
@overload
def _search_nearest(stats_clips: vs.VideoNode, *args, **kwargs) -> int: ... # noqa

@overload # noqa
def _search_nearest(stats_clips: List[vs.VideoNode], *args, **kwargs) -> List[int]: ... # noqa
# fmt: on


def _search_nearest(
    stats_clips: Union[vs.VideoNode, List[vs.VideoNode]],
    gen: Iterator[int],
    thr: Union[int, List[int]] = 0.02,
    offset: int = 0,
    strict: bool = True,
    silent: bool = True,
) -> Union[int, List[int]]:
    from timeit import default_timer as timer

    def _print(*args, **kwargs):
        if not silent:
            print(*args, **kwargs)

    notlist = False
    if not isinstance(stats_clips, list):
        notlist = True
        stats_clips = [stats_clips]

    if not isinstance(thr, list):
        thr = [thr]

    if len(stats_clips) != len(thr):
        raise ValueError("stats_clip and thr must be the same length!")

    stats_clips: List[Tuple[int, vs.VideoNode]] = list(enumerate(stats_clips))
    guesses: List[List[int, float]] = [[0, 1] for _ in range(len(stats_clips))]
    count = 0
    start = timer()
    for count, frame_number in enumerate(gen):
        for k, (i, frame) in enumerate(list((i, c.get_frame(frame_number)) for i, c in stats_clips)):
            diff = frame.props["PlaneStatsDiff"]
            if not (count % 100) or diff < guesses[i][1]:
                if diff < guesses[i][1]:
                    guesses[i] = [frame_number - offset, diff]
                if diff < thr[i]:
                    _print(f"\n\nFound frame: {guesses[i][0]}, with low diff of {guesses[i][1]}")
                    stats_clips.pop(k)
        if not (count % 100):
            end = timer()
            fps = 100 / (end - start)
            _print(
                f"Attempt: {count}  {fps:.2f} fps, eta {(stats_clips[0][1].num_frames - count) / fps:.2f}s      \r",
                end="",
            )
            start = timer()
        if not stats_clips:
            break
    else:
        _print("\n\n", end="")
        missed = list(i for i, _ in stats_clips)
        if strict:
            raise LookupError(
                f"Couldn't find a matching frame for scenes {', '.join(missed)}.\nClosest matches"
                f" were {', '.join(f'{guesses[i][0]} (diff: {guesses[i][1]})' for i in missed)}"
            )
        else:
            _print(
                f"Couldn't find a matching frame for scenes {', '.join(missed)}.\nUsing best"
                f" guesses {', '.join(f'{guesses[i][0]} (diff: {guesses[i][1]})' for i in missed)}",
            )
    indexes = [guess[0] for guess in guesses]
    return indexes[0] if notlist else indexes
