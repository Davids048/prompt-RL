#!/usr/bin/env python3
from __future__ import annotations

from typing import Any, Iterable

import numpy as np


_BRIDGE = "native"


class bridge:
    @staticmethod
    def set_bridge(name: str) -> None:
        global _BRIDGE
        _BRIDGE = name


def cpu(*args: Any, **kwargs: Any) -> None:
    del args, kwargs
    return None


class VideoReader:
    def __init__(
        self,
        uri: str,
        width: int | None = None,
        height: int | None = None,
        num_threads: int | None = None,
        **kwargs: Any,
    ) -> None:
        del num_threads, kwargs
        self.uri = str(uri)
        self.width = width
        self.height = height
        self._frames, self._fps = read_video(self.uri, width=width, height=height)

    def __len__(self) -> int:
        return len(self._frames)

    def get_avg_fps(self) -> float:
        return self._fps

    def get_batch(self, indices: Iterable[int]) -> Any:
        index_list = [int(index) for index in indices]
        batch = np.stack([self._frames[index] for index in index_list], axis=0)
        if _BRIDGE == "torch":
            import torch

            return torch.from_numpy(batch)
        return Batch(batch)


class Batch:
    def __init__(self, array: np.ndarray) -> None:
        self.array = array

    def asnumpy(self) -> np.ndarray:
        return self.array


def read_video(
    uri: str,
    width: int | None = None,
    height: int | None = None,
) -> tuple[list[np.ndarray], float]:
    import cv2

    cap = cv2.VideoCapture(uri)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {uri}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 1.0
        frames: list[np.ndarray] = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if width is not None and height is not None:
                frame = cv2.resize(frame, (int(width), int(height)), interpolation=cv2.INTER_AREA)
            frames.append(frame)
        if not frames:
            raise RuntimeError(f"Video has no readable frames: {uri}")
        return frames, fps
    finally:
        cap.release()
