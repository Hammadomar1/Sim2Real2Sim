"""Frames from the Logitech C920 through OpenCV: 1920x1080 MJPG at 30 fps with focus, exposure and white
balance fixed, and a thread that keeps only the newest frame (so a slow reader never sees an old one).

    cam = C920()                               # settings from calibration/c920_settings.json, or the defaults
    frame, stamp, n = cam.wait(after=0)        # BGR frame, time.monotonic() when it arrived, frame counter
    cam.close()

``ReplaySource`` serves stored frames (a list of images or a video file) through the same ``wait`` interface,
for tests and for re-running the estimator on a recording.
"""
from __future__ import annotations

import json
import math
import sys
import threading
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import cv2
import numpy as np

from .vision import CALIBRATION

SETTINGS = CALIBRATION / "c920_settings.json"
BACKENDS = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "v4l2": cv2.CAP_V4L2, "any": cv2.CAP_ANY}


@dataclass
class CameraSettings:
    """What the capture asks the driver for. ``camera_calibrate settings`` writes the locked values.

    Units are the driver's: with DirectShow (Windows) exposure is usually log2 of seconds (-6 = 1/64 s), with
    V4L2 (Linux) 100 us steps. The C920's focus runs 0-250 in steps of 5; ``camera_calibrate settings
    --sweep-focus`` picks the sharpest for the mat. ``None`` leaves a control as the driver has it. Drivers
    differ in what they honour: ``camera_calibrate settings`` prints what the driver reports back.
    """
    index: int = 0
    backend: str = "dshow" if sys.platform == "win32" else "v4l2"
    width: int = 1920
    height: int = 1080
    fps: int = 30
    fourcc: str = "MJPG"
    autofocus: bool = False
    focus: float | None = 30
    auto_exposure: bool = False
    exposure: float | None = -6
    gain: float | None = 0
    auto_white_balance: bool = False
    white_balance: float | None = 4500
    brightness: float | None = None
    contrast: float | None = None
    saturation: float | None = None

    def save(self, path=SETTINGS):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=1))

    @classmethod
    def load(cls, path=SETTINGS) -> "CameraSettings":
        """The saved settings, or the defaults when there are none yet."""
        if not Path(path).exists():
            return cls()
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in json.loads(Path(path).read_text()).items() if k in known})


def open_capture(s: CameraSettings) -> cv2.VideoCapture:
    """Open the camera and apply ``s``. Order matters on Windows: the MJPG FOURCC before the size."""
    cap = cv2.VideoCapture(s.index, BACKENDS[s.backend])
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera {s.index} with backend {s.backend}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*s.fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, s.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, s.height)
    cap.set(cv2.CAP_PROP_FPS, s.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    v4l2 = s.backend == "v4l2"
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if s.autofocus else 0)
    if not s.autofocus and s.focus is not None:
        cap.set(cv2.CAP_PROP_FOCUS, s.focus)
    # Auto exposure: V4L2 uses 1 = manual, 3 = aperture priority; other backends the legacy 0.25 / 0.75.
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, (3 if s.auto_exposure else 1) if v4l2 else (0.75 if s.auto_exposure else 0.25))
    if not s.auto_exposure and s.exposure is not None:
        cap.set(cv2.CAP_PROP_EXPOSURE, s.exposure)
    cap.set(cv2.CAP_PROP_AUTO_WB, 1 if s.auto_white_balance else 0)
    if not s.auto_white_balance and s.white_balance is not None:
        cap.set(cv2.CAP_PROP_WB_TEMPERATURE, s.white_balance)
    for prop, value in ((cv2.CAP_PROP_GAIN, s.gain), (cv2.CAP_PROP_BRIGHTNESS, s.brightness),
                        (cv2.CAP_PROP_CONTRAST, s.contrast), (cv2.CAP_PROP_SATURATION, s.saturation)):
        if value is not None:
            cap.set(prop, value)
    return cap


def read_back(cap: cv2.VideoCapture) -> dict:
    """What the driver reports for each control (-1 or 0 often means "not supported by this backend")."""
    four = int(cap.get(cv2.CAP_PROP_FOURCC))
    props = dict(width=cv2.CAP_PROP_FRAME_WIDTH, height=cv2.CAP_PROP_FRAME_HEIGHT, fps=cv2.CAP_PROP_FPS,
                 autofocus=cv2.CAP_PROP_AUTOFOCUS, focus=cv2.CAP_PROP_FOCUS, auto_exposure=cv2.CAP_PROP_AUTO_EXPOSURE,
                 exposure=cv2.CAP_PROP_EXPOSURE, gain=cv2.CAP_PROP_GAIN, auto_white_balance=cv2.CAP_PROP_AUTO_WB,
                 white_balance=cv2.CAP_PROP_WB_TEMPERATURE, brightness=cv2.CAP_PROP_BRIGHTNESS,
                 contrast=cv2.CAP_PROP_CONTRAST, saturation=cv2.CAP_PROP_SATURATION, buffer=cv2.CAP_PROP_BUFFERSIZE)
    out = {k: cap.get(p) for k, p in props.items()}
    out["fourcc"] = "".join(chr((four >> 8 * i) & 0xFF) for i in range(4))
    return out


class C920:
    """The camera with a grabbing thread: ``wait`` returns the newest frame, never a queued old one."""

    def __init__(self, settings: CameraSettings | None = None, first_frame_timeout: float = 5.0):
        self.settings = settings or CameraSettings.load()
        self.cap = open_capture(self.settings)
        self.driver = read_back(self.cap)
        self._cond = threading.Condition()
        self._frame, self._stamp, self._count, self.errors = None, 0.0, 0, 0
        self._stop = False
        self._thread = threading.Thread(target=self._grab, name="c920-grab", daemon=True)
        self._thread.start()
        if self.wait(0, first_frame_timeout) is None:
            self.close()
            raise RuntimeError("no frame from the camera")
        frame = self._frame
        if frame.shape[1] != self.settings.width or frame.shape[0] != self.settings.height:
            got = f"{frame.shape[1]}x{frame.shape[0]}"
            self.close()
            raise RuntimeError(f"camera delivers {got}, not {self.settings.width}x{self.settings.height} "
                               f"(another backend? --backend msmf / dshow)")

    def _grab(self):
        while not self._stop:
            if not self.cap.grab():                 # blocks until the next frame arrives
                self.errors += 1
                time.sleep(0.005)
                continue
            stamp = time.monotonic()
            ok, frame = self.cap.retrieve()         # MJPG decode
            if not ok:
                self.errors += 1
                continue
            with self._cond:
                self._frame, self._stamp, self._count = frame, stamp, self._count + 1
                self._cond.notify_all()

    def wait(self, after: int = 0, timeout: float = 1.0):
        """The newest frame once it is newer than frame number ``after``: (frame, stamp, number), or None."""
        with self._cond:
            if not self._cond.wait_for(lambda: self._count > after or self._stop, timeout) or self._stop:
                return None
            return self._frame, self._stamp, self._count

    def close(self):
        self._stop = True
        with self._cond:
            self._cond.notify_all()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=1.0)
        self.cap.release()


class ReplaySource:
    """Stored frames through the camera's ``wait`` interface: a list of BGR images or a video file, served at
    ``fps`` (0: as fast as they are asked for). Each frame is stamped when it is served."""

    def __init__(self, frames, fps: float = 0.0, loop: bool = False):
        self.frames, self.fps, self.loop = frames, fps, loop
        self._video = cv2.VideoCapture(str(frames)) if isinstance(frames, (str, Path)) else None
        self._count, self._t0 = 0, time.monotonic()

    def _next(self):
        if self._video is not None:
            ok, frame = self._video.read()
            if not ok and self.loop:
                self._video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._video.read()
            return frame if ok else None
        if self._count >= len(self.frames) and not self.loop:
            return None
        return self.frames[self._count % len(self.frames)]

    def wait(self, after: int = 0, timeout: float = 1.0):
        if self.fps > 0:                            # real time: the frame due now
            due = self._t0 + self._count / self.fps
            time.sleep(max(0.0, due - time.monotonic()))
        frame = self._next()
        if frame is None:
            time.sleep(min(timeout, 0.05))
            return None
        self._count += 1
        return frame, time.monotonic(), self._count

    def close(self):
        if self._video is not None:
            self._video.release()


def sharpness(image: np.ndarray) -> float:
    """Variance of the Laplacian over the image centre: higher is sharper (for choosing the focus)."""
    h, w = image.shape[:2]
    centre = cv2.cvtColor(image[h // 4:3 * h // 4, w // 4:3 * w // 4], cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(centre, cv2.CV_64F).var())


def describe(driver: dict) -> str:
    return ", ".join(f"{k} {v:g}" if isinstance(v, float) and math.isfinite(v) else f"{k} {v}" for k, v in driver.items())
