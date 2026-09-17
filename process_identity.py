# -*- coding: utf-8 -*-
"""Phase 1a-post: BiRefNet matting + COM centering + dark-blue canvas for selected identity candidate.

Usage: env -u PYTHONPATH -u PYTHONHOME C:/Users/humac/anaconda3/python.exe process_identity.py
"""
import os, sys, numpy as np
from PIL import Image
import torch
from transformers import AutoModelForImageSegmentation
from torchvision import transforms

# ── CONFIG ──
CANVAS_W = 1088
CANVAS_H = 832
BG_OUT = (18, 37, 69)         # darkblue background RGB
BG = np.array([18, 37, 69], dtype=float)
PET_H_PCT = 54                # target subject height as % of canvas

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, 'identity_candidates', 'cand_2.png')  # user selected
DST = os.path.join(ROOT, 'darkblue_refs_new', 'ref_front.png')
BACKUP = os.path.join(ROOT, 'darkblue_refs_new', 'backup', 'ref_front.png')
# ── CONFIG END ──

def main():
    print(f'Loading BiRefNet...', flush=True)
    local_path = os.path.expanduser(
        '~/.cache/huggingface/hub/models--zhengpeng7--BiRefNet/snapshots/'
        'e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4'
    )
    model = AutoModelForImageSegmentation.from_pretrained(
        local_path, trust_remote_code=True, local_files_only=True
    ).cuda().eval()
    tf = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    print(f'Processing {SRC}...', flush=True)
    img = Image.open(SRC).convert('RGB')
    img_np = np.array(img)
    h, w = img_np.shape[:2]
    print(f'  Source: {w}x{h}', flush=True)

    # BiRefNet inference
    inp = tf(img).unsqueeze(0).cuda()
    with torch.no_grad():
        sig = model(inp)[-1].sigmoid().squeeze().cpu().numpy()
    sig_pil = Image.fromarray((sig * 65535).astype(np.uint16))
    sig_pil = sig_pil.resize((w, h), Image.BILINEAR)
    sig = np.array(sig_pil).astype(np.float64) / 65535.0

    # Three-zone alpha
    alpha = np.zeros((h, w), dtype=np.uint8)
    alpha[sig > 0.7] = 255
    soft = (sig > 0.05) & (sig <= 0.7)
    alpha[soft] = (sig[soft] * 255).astype(np.uint8)

    rgba = np.dstack([img_np, alpha])

    # Crop to bounding box of foreground
    fg = alpha > 10
    ys, xs = np.where(fg)
    margin = 5
    y0, y1 = max(0, ys.min() - margin), min(h, ys.max() + margin)
    x0, x1 = max(0, xs.min() - margin), min(w, xs.max() + margin)
    cat_crop = rgba[y0:y1, x0:x1]

    crop_h = ys.max() - ys.min() + 1
    target_h = int(CANVAS_H * PET_H_PCT / 100)
    scale = target_h / crop_h
    new_w = int(cat_crop.shape[1] * scale)
    new_h = int(cat_crop.shape[0] * scale)

    # If width exceeds 80%W, scale by width instead
    if new_w > CANVAS_W * 0.80:
        scale = (CANVAS_W * 0.80) / cat_crop.shape[1]
        new_w = int(cat_crop.shape[1] * scale)
        new_h = int(cat_crop.shape[0] * scale)

    cat_img = Image.fromarray(cat_crop)
    cat_resized = cat_img.resize((new_w, new_h), Image.LANCZOS)

    # COM centering (use solid alpha center of mass)
    crop_alpha = cat_crop[:, :, 3].astype(float)
    solid_alpha = crop_alpha.copy()
    solid_alpha[solid_alpha <= 128] = 0
    total = solid_alpha.sum()
    if total > 0:
        com_x_in_crop = (solid_alpha * np.arange(cat_crop.shape[1])).sum() / total
    else:
        com_x_in_crop = cat_crop.shape[1] / 2
    com_x_resized = com_x_in_crop * scale

    # Paste: bottom-anchored with 8% margin from bottom
    paste_y = CANVAS_H - new_h - int(CANVAS_H * 0.08)
    paste_x = int(CANVAS_W / 2 - com_x_resized)
    paste_x = max(10, min(paste_x, CANVAS_W - new_w - 10))

    canvas = Image.new('RGBA', (CANVAS_W, CANVAS_H), BG_OUT + (255,))
    canvas.paste(cat_resized, (paste_x, paste_y), cat_resized)
    canvas.save(DST)
    print(f'  Saved: {DST}', flush=True)

    # Backup (read-only after creation)
    os.makedirs(os.path.dirname(BACKUP), exist_ok=True)
    canvas.save(BACKUP)
    print(f'  Backup: {BACKUP}', flush=True)

    # Composition audit (RGB color distance, not alpha)
    arr2 = np.array(canvas.convert('RGB'), dtype=float)
    dist2 = np.sqrt(((arr2 - BG) ** 2).sum(axis=2))
    fg2 = dist2 > 25
    ys2, xs2 = np.where(fg2)
    if len(ys2) > 0:
        dw = xs2.max() - xs2.min() + 1
        dh = ys2.max() - ys2.min() + 1
        pct_w = dw / CANVAS_W * 100
        pct_h = dh / CANVAS_H * 100
        print(f'  Composition: {pct_w:.1f}%W x {pct_h:.1f}%H', flush=True)
        if pct_w > 85 or pct_h > 80:
            print(f'  ⚠️ Subject too large! Adjust PET_H_PCT', flush=True)
        else:
            print(f'  ✅ Composition OK', flush=True)
    else:
        print(f'  ⚠️ No foreground detected!', flush=True)

    sz = os.path.getsize(DST) // 1024
    print(f'  File size: {sz}KB', flush=True)

    # Cleanup
    del model
    torch.cuda.empty_cache()
    print('\nDONE', flush=True)

if __name__ == '__main__':
    main()
