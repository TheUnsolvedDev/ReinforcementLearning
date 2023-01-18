import numpy as np
import cv2

from params import *


def preprocess_frame(screen, exclude=(30, -4, -12, 4), output=84):
    screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
    screen = screen[exclude[0]:exclude[2], exclude[3]:exclude[1]]
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = cv2.resize(screen, (output, output), interpolation=cv2.INTER_AREA)
    return screen


def stack_frames(frames, frame, is_new):
    if is_new:
        frames = np.stack(arrays=[frame, frame, frame, frame])
    else:
        frames = frames.reshape(
            (INPUT_SHAPE[-1], INPUT_SHAPE[0], INPUT_SHAPE[1]))
        frames[0] = frames[1]
        frames[1] = frames[2]
        frames[2] = frames[3]
        frames[3] = frame

    return frames.reshape(INPUT_SHAPE)
