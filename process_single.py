# -*- coding: utf-8 -*-
"""Process a single selected candidate through BiRefNet → COM centering → dark blue canvas.

Usage: env -u PYTHONPATH -u PYTHONHOME C:/Users/humac/anaconda3/python.exe process_single.py <input.png> <output_name>
Example: python process_single.py darkblue_refs_new/ref_side_c1.png side
"""
import os, sys, numpy as np
from PIL import Image
import torch
from transformers import AutoModelForImageSegmentation
from torchvision import transforms

CANVAS_W = 1088
CANVAS_H = 832
BG_OUT = (18, 37, 69)
BG = np.array([18, 37, 69], dtype=float)
PET_H_PCT = 54

def main():
    if len(sys.argv) < 3:
        print('Usage: process_single.py <input.png> <name>')
        sys.exit(1)
    src = sys.argv[1]
    name = sys.argv[2]
    root = os.path.dirname(os.path.abspath(__file__))
    dst = os.path.join(root, 'darkblue_refs_new', f'ref_{name}.png')
    backup = os.path.join(root, 'darkblue_refs_new', 'backup', f'ref_{name}.png')

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

    img = Image.open(src).convert('RGB')
    img_np = np.array(img)
    h, w = img_np.shape[:2]
    print(f'Source: {w}x{h}', flush=True)

    inp = tf(img).unsqueeze(0).cuda()
    with torch.no_grad():
        sig = model(inp)[-1].sigmoid().squeeze().cpu().numpy()
    sig_pil = Image.fromarray((sig * 65535).astype(np.uint16))
    sig_pil = sig_pil.resize((w, h), Image.BILINEAR)
    sig = np.array(sig_pil).astype(np.float64) / 65535.0

    alpha = np.zeros((h, w), dtype=np.uint8)
    alpha[sig > 0.7] = 255
    soft = (sig > 0.05) & (sig <= 0.7)
    alpha[soft] = (sig[soft] * 255).astype(np.uint8)
    rgba = np.dstack([img_np, alpha])
    fg = alpha > 10
    ys, xs = np.where(fg)
    margin = 5
    y0, y1 = max(0, ys.min()-margin), min(h, ys.max()+margin)
    x0, x1 = max(0, xs.min()-margin), min(w, xs.max()+margin)
    cat_crop = rgba[y0:y1, x0:x1]
    crop_h = ys.max() - ys.min() + 1
    target_h = int(CANVAS_H * PET_H_PCT / 100)
    scale = target_h / crop_h
    new_w = int(cat_crop.shape[1] * scale)
    new_h = int(cat_crop.shape[0] * scale)
    if new_w > CANVAS_W * 0.80:
        scale = (CANVAS_W * 0.80) / cat_crop.shape[1]
        new_w = int(cat_crop.shape[1] * scale)
        new_h = int(cat_crop.shape[0] * scale)
    cat_img = Image.fromarray(cat_crop)
    cat_resized = cat_img.resize((new_w, new_h), Image.LANCZOS)
    crop_alpha = cat_crop[:,:,3].astype(float)
    solid_alpha = crop_alpha.copy()
    solid_alpha[solid_alpha <= 128] = 0
    total = solid_alpha.sum()
    com_x = (solid_alpha * np.arange(cat_crop.shape[1])).sum() / total if total > 0 else cat_crop.shape[1] / 2
    com_x_r = com_x * scale
    paste_y = CANVAS_H - new_h - int(CANVAS_H * 0.08)
    paste_x = int(CANVAS_W / 2 - com_x_r)
    paste_x = max(10, min(paste_x, CANVAS_W - new_w - 10))
    canvas = Image.new('RGBA', (CANVAS_W, CANVAS_H), BG_OUT + (255,))
    canvas.paste(cat_resized, (paste_x, paste_y), cat_resized)
    canvas.save(dst)
    os.makedirs(os.path.dirname(backup), exist_ok=True)
    canvas.save(backup)

    arr2 = np.array(canvas.convert('RGB'), dtype=float)
    dist2 = np.sqrt(((arr2 - BG)**2).sum(axis=2))
    fg2 = dist2 > 25
    ys2, xs2 = np.where(fg2)
    if len(ys2) > 0:
        pct_w = (xs2.max()-xs2.min()+1) / CANVAS_W * 100
        pct_h = (ys2.max()-ys2.min()+1) / CANVAS_H * 100
        print(f'Composition: {pct_w:.1f}%W x {pct_h:.1f}%H', flush=True)
    print(f'Saved: {dst} ({os.path.getsize(dst)//1024}KB)', flush=True)
    print(f'Backup: {backup}', flush=True)

    del model
    torch.cuda.empty_cache()
    print('DONE', flush=True)

if __name__ == '__main__':
    main()
