# -*- coding: utf-8 -*-
"""Phase 1b: Side-view identity reference image for Maine Coon Domo-kun cat.
Generates 3 side-view candidates for user selection, then BiRefNet processing.

Usage: env -u PYTHONPATH -u PYTHONHOME C:/Users/humac/anaconda3/python.exe gen_side_ref.py
"""
import base64, json, os, io, time, sys
import requests, numpy as np
from PIL import Image
import torch
from transformers import AutoModelForImageSegmentation
from torchvision import transforms

BASE = 'https://api.agnes-ai.cn'
ROOT = os.path.dirname(os.path.abspath(__file__))
CANDIDATES = 3

# ── CONFIG ──
CANVAS_W = 1088
CANVAS_H = 832
BG_OUT = (18, 37, 69)
BG = np.array([18, 37, 69], dtype=float)
PET_H_PCT = 54

KEYHEX = os.path.join(ROOT, 'keyhex.txt')

SUBJ_SIDE = (
    'A 1-year-old Maine Coon cat, silver tabby fur with silvery-white undercoat '
    'and soft grey stripes, pale green eyes, large pointed ears with dark lynx-tip tufts, '
    'pink nose, neat white ruff, well-groomed sleek coat with clean neat edges and smooth '
    'outline, long plumed tail held neatly. '
    'Wearing a white Domo-kun space-suit pet vest with cyan stand-up collar and center zipper, '
    'orange chest strap with two silver D-rings, orange-piped front panels each with a small '
    'cyan zipper, a round black-and-white D-badge on the left panel, '
    'a brown Domo-kun character patch on the right panel.'
)

AVOID = (
    'extra legs, extra paws, five legs, two tails, extra tail, second tail, '
    'deformed, mutated, watermark, text, blurry, close up, zoomed in, '
    'cartoon, short-haired, fluffy, shaggy, messy fur, flyaway hairs, '
    'Siamese, Persian, British Shorthair, Sphynx, Bengal, '
    'hood, sleeves, leg sleeves, collar only, red bandana'
)

# Side-view poses — ear type = pointed with lynx tufts
SIDE_POSES = [
    ' Standing in an exact side profile view facing right on all four paws, '
    'alert and confident, exactly four legs clearly visible, one single long tail, '
    'two large pointed ears with dark lynx tufts. Calm attentive expression.',

    ' Standing in an exact side profile view facing right, mid-stride with one front paw '
    'slightly forward, exactly four legs, one single tail, two pointed ears with lynx tips. '
    'Curious friendly expression, ears perked forward.',

    ' Standing in an exact side profile view facing right, all four paws on the ground, '
    'head held high proudly, exactly four legs, one single long tail, two large pointed '
    'ears with dark tufts. Regal poised expression, neat ruff prominent.',
]

BG_TEXT = ' Dark navy blue background #000D43. Photorealistic ultra-sharp fur soft even lighting.'
COMMON = ' Centered. The subject takes up about 60 percent of frame height with some dark blue space. Static camera.'
# ── CONFIG END ──

tok = bytes.fromhex(open(KEYHEX).read().strip()).decode()
HDR = {'Authorization': tok, 'Content-Type': 'application/json'}

OUT_DIR = os.path.join(ROOT, 'darkblue_refs_new')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Anatomy audit ──
def audit_anatomy(img):
    arr = np.array(img.convert('RGB'))
    h, w = arr.shape[:2]
    bg_ref = np.array([0, 13, 67], dtype=float)
    dist = np.sqrt(((arr.astype(float) - bg_ref)**2).sum(axis=2))
    fg_mask = dist > 35
    ys, xs = np.where(fg_mask)
    if len(ys) == 0:
        return False, "no fg"
    very_bot = int(ys.max() * 0.90)
    if very_bot >= h:
        very_bot = h - 1
    bot_slice = fg_mask[very_bot:, :]
    if bot_slice.shape[0] == 0:
        return True, "ok (bottom edge empty)"
    col_mean = bot_slice.mean(axis=0)
    paw_cols = col_mean > 0.15
    transitions = np.diff(paw_cols.astype(int))
    paw_count = (transitions == 1).sum()
    if paw_count > 5:
        return False, f"too many paw clusters: {paw_count}"
    return True, f"paws={paw_count}"

# ── BiRefNet processing ──
def process_ref(model, tf, src_path, dst_path):
    img = Image.open(src_path).convert('RGB')
    img_np = np.array(img)
    h, w = img_np.shape[:2]
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
    com_x_in_crop = (solid_alpha * np.arange(cat_crop.shape[1])).sum() / total if total > 0 else cat_crop.shape[1] / 2
    com_x_resized = com_x_in_crop * scale
    paste_y = CANVAS_H - new_h - int(CANVAS_H * 0.08)
    paste_x = int(CANVAS_W / 2 - com_x_resized)
    paste_x = max(10, min(paste_x, CANVAS_W - new_w - 10))
    canvas = Image.new('RGBA', (CANVAS_W, CANVAS_H), BG_OUT + (255,))
    canvas.paste(cat_resized, (paste_x, paste_y), cat_resized)
    canvas.save(dst_path)
    # Audit
    arr2 = np.array(canvas.convert('RGB'), dtype=float)
    dist2 = np.sqrt(((arr2 - BG)**2).sum(axis=2))
    fg2 = dist2 > 25
    ys2, xs2 = np.where(fg2)
    if len(ys2) > 0:
        dw = xs2.max()-xs2.min()+1
        dh = ys2.max()-ys2.min()+1
        pct_w = dw / CANVAS_W * 100
        pct_h = dh / CANVAS_H * 100
        return f'{os.path.getsize(dst_path)//1024}KB cat={pct_w:.1f}%W x {pct_h:.1f}%H'
    return f'{os.path.getsize(dst_path)//1024}KB (no fg audit)'

def main():
    action = sys.argv[1] if len(sys.argv) > 1 else 'side'
    
    print(f'Generating {CANDIDATES} candidates for ref_{action}...', flush=True)
    results = []
    for i, pose in enumerate(SIDE_POSES):
        prompt = SUBJ_SIDE + pose + BG_TEXT + COMMON + f' Avoid: {AVOID}'
        print(f'\nCandidate {i}: prompt={len(prompt)}ch', flush=True)
        payload = {'model': 'agnes-image-2.1-flash', 'prompt': prompt,
                   'size': '1024x1024', 'response_format': 'b64_json'}
        r = None
        for attempt in range(3):
            try:
                r = requests.post(f'{BASE}/v1/images/generations', headers=HDR,
                                  json=payload, timeout=300)
                if r.status_code == 200:
                    break
                print(f'  HTTP {r.status_code}: {r.text[:120]}', flush=True)
                time.sleep(5)
            except Exception as e:
                print(f'  retry {attempt+1}: {e}', flush=True)
                time.sleep(5)
        if r is None or r.status_code != 200:
            print(f'  FAILED', flush=True)
            continue
        d = r.json()
        b64 = d.get('data',[{}])[0].get('b64_json')
        url = d.get('data',[{}])[0].get('url','')
        if b64:
            img = Image.open(io.BytesIO(base64.b64decode(b64))).convert('RGBA')
        elif url:
            r2 = requests.get(url, timeout=60)
            img = Image.open(io.BytesIO(r2.content)).convert('RGBA')
        else:
            continue
        cand_path = os.path.join(OUT_DIR, f'ref_{action}_c{i}.png')
        img.save(cand_path)
        ok, msg = audit_anatomy(img)
        print(f'  saved {cand_path} {img.size} audit={ok} ({msg})', flush=True)
        results.append({'i': i, 'path': cand_path, 'ok': ok})
        time.sleep(3)
    
    print(f'\nGenerated {len(results)} candidates. User selection needed.', flush=True)
    
    # Write results for user reference
    with open(os.path.join(OUT_DIR, f'ref_{action}_candidates.json'), 'w') as f:
        json.dump(results, f, indent=1)

if __name__ == '__main__':
    main()
