# -*- coding: utf-8 -*-
"""Pre-compute sprite_meta.json for all states.
Scans alpha channel of each frame to extract content bbox and body_bottom.
Output: assets/sprite_meta.json (~186KB for 20 states × 121 frames)

Used by _build_state streaming load to avoid loading all frames just to scan bbox.
Reduces _build_state peak from 477MB (all raw frames) to ~3.6MB (1 frame at a time).

Usage: env -u PYTHONPATH -u PYTHONHOME C:/Users/humac/anaconda3/python.exe precompute_sprite_meta.py
"""
import os, json
import numpy as np
from PIL import Image

ASSETS_DIR = 'assets'
STATES = [
    'idle', 'walk', 'run', 'eat', 'bark', 'sleep', 'sit', 'lick',
    'happy', 'roll', 'dance', 'stretch', 'beg', 'bath', 'surprised',
    'play_dead', 'pet', 'kiss', 'wave', 'type'
]

def scan_frame(alpha):
    mask = alpha > 128
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    body_bottom = alpha.shape[0] - 1
    for row in range(alpha.shape[0] - 1, -1, -1):
        if np.any(alpha[row, ::8] > 16):
            body_bottom = row
            break
    return {"top": int(ys.min()), "bottom": int(ys.max()),
            "left": int(xs.min()), "right": int(xs.max()),
            "body_bottom": body_bottom}

def scan_state(state):
    frames_data = []
    max_content_h = max_content_w = 0
    frame_w = frame_h = 0
    for i in range(121):
        fn = os.path.join(ASSETS_DIR, f'{state}_{i:03d}.webp')
        if not os.path.exists(fn):
            break
        arr = np.array(Image.open(fn))
        if frame_w == 0:
            frame_h, frame_w = arr.shape[:2]
        info = scan_frame(arr[:, :, 3])
        if info is None:
            info = {"top": 0, "bottom": 0, "left": 0, "right": 0, "body_bottom": 0}
        ch = info["bottom"] - info["top"] + 1
        cw = info["right"] - info["left"] + 1
        max_content_h = max(max_content_h, ch)
        max_content_w = max(max_content_w, cw)
        frames_data.append(info)
    return {"frame_w": frame_w, "frame_h": frame_h, "num_frames": len(frames_data),
            "max_content_h": max_content_h, "max_content_w": max_content_w,
            "max_dim": max(max_content_h, max_content_w), "frames": frames_data}

if __name__ == '__main__':
    meta = {}
    for state in STATES:
        print(f'Scanning {state}...', flush=True)
        meta[state] = scan_state(state)
        d = meta[state]
        print(f'  {d["num_frames"]} frames, {d["frame_w"]}x{d["frame_h"]}, max_dim={d["max_dim"]}')
    out_path = os.path.join(ASSETS_DIR, 'sprite_meta.json')
    with open(out_path, 'w') as f:
        json.dump(meta, f)
    print(f'Saved: {out_path} ({os.path.getsize(out_path)} bytes)')
