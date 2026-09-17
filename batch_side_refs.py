#!/usr/bin/env python3
"""Batch generate and process side-view reference images for Maine Coon cat."""
import requests, json, io, os, time, base64, sys
import torch, numpy as np
from PIL import Image
from transformers import AutoModelForImageSegmentation
from torchvision import transforms

# ── Config ──
CANVAS_W = 1088
CANVAS_H = 832
BG_OUT = (18, 37, 69)
BG = np.array([18, 37, 69], dtype=float)
OUT_DIR = 'darkblue_refs_new'
CANDIDATES = 3  # 3候选→用户手选
MAX_RETRIES = 3

tok = bytes.fromhex(open('keyhex.txt').read().strip()).decode()
HDR = {'Authorization': tok, 'Content-Type': 'application/json'}

# SUBJ: 必须与gen_identity.py一字不差
SUBJ = (
    'A 1-year-old Maine Coon cat, silver tabby fur with silvery-white undercoat '
    'and soft grey stripes, pale green eyes, large pointed ears with dark lynx-tip tufts, '
    'pink nose, fluffy white ruff, bushy plumed smoky-grey tail. '
    'Wearing a white Domo-kun space-suit pet vest with cyan stand-up collar and center zipper, '
    'orange chest strap with two silver D-rings, orange-piped front panels each with a small '
    'cyan zipper, a round black-and-white D-badge on the left panel, '
    'a brown Domo-kun character patch on the right panel.'
)

NEG = (
    ', extra legs, extra paws, five legs, two tails, extra tail, second tail, '
    'double tail, four ears, extra ears, extra limb, deformed, mutated, duplicate, '
    'short-haired, Siamese, Persian, British Shorthair, Sphynx, Bengal, '
    'blank face, stiff expression, dead eyes, emotionless, dull, vacant'
)

BG_TEXT = ' Dark navy blue studio background (color #000D43). Photorealistic, ultra-sharp fur.'

# 侧面动作参考图——关键约束前置（词序铁律）
POSES = {
    'walk': (
        'ONE single tail only. PERFECT 90-DEGREE SIDE PROFILE view facing right. '
        'Only ONE eye visible — the near-side eye. The far-side eye is COMPLETELY HIDDEN. '
        'Muzzle points to the right. Body silhouette is WIDE and ELONGATED horizontally, '
        'clearly wider than tall, body perfectly horizontal and level. '
        'Head and body point in EXACTLY the same direction, ZERO head turn. '
        'Walking mid-stride with one front paw forward, '
        'exactly four legs, two pointed ears with lynx tips. '
        'Calm confident alert expression: eyes bright and focused looking ahead, '
        'ears slightly forward, tail gently swaying, carefree and content.'
    ),
    'run': (
        'ONE single tail only. PERFECT 90-DEGREE SIDE PROFILE view facing right. '
        'Only ONE eye visible — the near-side eye. The far-side eye is COMPLETELY HIDDEN. '
        'Muzzle points to the right. Body silhouette is WIDE and ELONGATED horizontally, '
        'clearly wider than tall, body perfectly horizontal and level. '
        'Head and body point in EXACTLY the same direction, ZERO head turn. '
        'Running at full speed, all four legs extended in a full gallop, body stretched forward, '
        'exactly four legs, one single tail streaming behind, two pointed ears with lynx tips. '
        'Wild exhilarated expression: ears flying back with wind, '
        'mouth wide open with tongue flapping, eyes squinting with pure thrill.'
    ),
    'eat': (
        'ONE single tail only. PERFECT 90-DEGREE SIDE PROFILE view facing right. '
        'Only ONE eye visible — the near-side eye. The far-side eye is COMPLETELY HIDDEN. '
        'Muzzle points to the right. Body silhouette is WIDE and ELONGATED horizontally, '
        'clearly wider than tall, body perfectly horizontal and level. '
        'Head and body point in EXACTLY the same direction, ZERO head turn. '
        'Standing in side profile facing right, a small RED food bowl full of brown kibble '
        'on the ground in front. Head held high, flat front lighting, no facial shadows. '
        'Exactly four legs, two pointed ears with lynx tips.'
    ),
    'bark': (
        'ONE single tail only. PERFECT 90-DEGREE SIDE PROFILE view facing right. '
        'Only ONE eye visible — the near-side eye. The far-side eye is COMPLETELY HIDDEN. '
        'Muzzle points to the right. Body silhouette is WIDE and ELONGATED horizontally, '
        'clearly wider than tall, body perfectly horizontal and level. '
        'Head and body point in EXACTLY the same direction, ZERO head turn. '
        'Standing upright on all four legs, mouth open barking, '
        'exactly four legs, one single tail, two pointed ears with lynx tips.'
    ),
    'happy': (
        'ONE single tail only. PERFECT 90-DEGREE SIDE PROFILE view facing right. '
        'Only ONE eye visible — the near-side eye. The far-side eye is COMPLETELY HIDDEN. '
        'Muzzle points to the right. Body silhouette is WIDE and ELONGATED horizontally, '
        'clearly wider than tall, body perfectly horizontal and level. '
        'Head and body point in EXACTLY the same direction, ZERO head turn. '
        'Standing in side view, body bouncing up and down joyfully, '
        'exactly four legs, one single tail wagging, two pointed ears with lynx tips. '
        'Overjoyed ecstatic expression: mouth wide open, eyes sparkling.'
    ),
    'lick': (
        'ONE single tail only. PERFECT 90-DEGREE SIDE PROFILE view facing right. '
        'Only ONE eye visible — the near-side eye. The far-side eye is COMPLETELY HIDDEN. '
        'Muzzle points to the right. Body silhouette is WIDE and ELONGATED horizontally, '
        'clearly wider than tall, body perfectly horizontal and level. '
        'Head and body point in EXACTLY the same direction, ZERO head turn. '
        'Sitting in side profile, one front paw raised to its mouth licking it, '
        'exactly four legs, one single tail, two pointed ears with lynx tips.'
    ),
    'sleep': (
        'ONE single tail only. PERFECT 90-DEGREE SIDE PROFILE view. '
        'Only ONE eye visible — the near-side eye. The far-side eye is COMPLETELY HIDDEN. '
        'Lying on its side curled up peacefully sleeping, '
        'exactly four legs tucked in, one single tail wrapped around body, two pointed ears with lynx tips. '
        'Serene blissful sleeping expression: eyes gently closed, '
        'mouth softly relaxed, ears completely limp and relaxed.'
    ),
    'roll': (
        'ONE single tail only. PERFECT 90-DEGREE SIDE PROFILE view. '
        'Only ONE eye visible — the near-side eye. The far-side eye is COMPLETELY HIDDEN. '
        'Lying on its back in side view, belly exposed, paws in the air, '
        'rolling playfully, exactly four legs, one single tail, two pointed ears with lynx tips.'
    ),
    'stretch': (
        'ONE single tail only. PERFECT 90-DEGREE SIDE PROFILE view facing right. '
        'Only ONE eye visible — the near-side eye. The far-side eye is COMPLETELY HIDDEN. '
        'Muzzle points to the right. Body silhouette is WIDE and ELONGATED horizontally, '
        'clearly wider than tall, body perfectly horizontal and level. '
        'Head and body point in EXACTLY the same direction, ZERO head turn. '
        'Doing a full body downward stretch, front legs stretched far forward, hindquarters raised high, '
        'exactly four legs, one single tail, two pointed ears with lynx tips.'
    ),
    'play_dead': (
        'ONE single tail only. PERFECT 90-DEGREE SIDE PROFILE view. '
        'Only ONE eye visible — the near-side eye. The far-side eye is COMPLETELY HIDDEN. '
        'Lying completely still on its side, all four legs stiff and extended, playing dead, '
        'exactly four legs, one single tail, two pointed ears with lynx tips. '
        'Dramatic playing-dead expression: eyes squeezed shut, tongue hanging out.'
    ),
}

def gen_image(prompt):
    for retry in range(MAX_RETRIES):
        try:
            r = requests.post('https://api.agnes-ai.cn/v1/images/generations', headers=HDR, json={
                'model': 'agnes-image-2.1-flash',
                'prompt': prompt,
                'size': '1024x1024',
            }, timeout=300)
            if r.status_code == 200:
                d = r.json()
                data = d.get('data', [{}])[0]
                url = data.get('url', '')
                if url:
                    r2 = requests.get(url, timeout=60)
                    return Image.open(io.BytesIO(r2.content))
                b64j = data.get('b64_json', '')
                if b64j:
                    return Image.open(io.BytesIO(base64.b64decode(b64j)))
            print(f'    retry {retry+1}: HTTP {r.status_code}', flush=True)
            time.sleep(3)
        except Exception as e:
            print(f'    retry {retry+1}: {e}', flush=True)
            time.sleep(3)
    return None

def audit_anatomy(img):
    """Check for obvious deformities."""
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
    if paw_count >= 2:
        return True, f"paws={paw_count}"
    fg_width = xs.max() - xs.min()
    if fg_width > w * 0.5:
        return True, "wide fg (likely lying)"
    return True, f"narrow paws={paw_count} (accept)"

def process_image(model, tf, src_path, dst_path, dog_h_pct=54):
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
    dog_crop = rgba[y0:y1, x0:x1]
    dog_img = Image.fromarray(dog_crop)
    crop_h = ys.max() - ys.min() + 1
    target_h = int(CANVAS_H * dog_h_pct / 100)
    scale = target_h / crop_h
    new_w = int(dog_img.size[0] * scale)
    new_h = int(dog_img.size[1] * scale)
    if new_w > CANVAS_W * 0.80:
        scale = (CANVAS_W * 0.80) / dog_img.size[0]
        new_w = int(dog_img.size[0] * scale)
        new_h = int(dog_img.size[1] * scale)
    dog_resized = dog_img.resize((new_w, new_h), Image.LANCZOS)
    crop_alpha = dog_crop[:,:,3].astype(float)
    solid_alpha = crop_alpha.copy()
    solid_alpha[solid_alpha <= 128] = 0
    total = solid_alpha.sum()
    com_x_in_crop = (solid_alpha * np.arange(dog_crop.shape[1])).sum() / total if total > 0 else dog_crop.shape[1] / 2
    com_x_resized = com_x_in_crop * scale
    paste_y = CANVAS_H - new_h - int(CANVAS_H * 0.08)
    paste_x = int(CANVAS_W / 2 - com_x_resized)
    paste_x = max(10, min(paste_x, CANVAS_W - new_w - 10))
    canvas = Image.new('RGBA', (CANVAS_W, CANVAS_H), BG_OUT + (255,))
    canvas.paste(dog_resized, (paste_x, paste_y), dog_resized)
    canvas.save(dst_path)
    arr2 = np.array(canvas.convert('RGB'), dtype=float)
    dist2 = np.sqrt(((arr2 - BG)**2).sum(axis=2))
    fg2 = dist2 > 25
    ys2, xs2 = np.where(fg2)
    dw = xs2.max()-xs2.min()+1
    dh = ys2.max()-ys2.min()+1
    pct_w = dw / CANVAS_W * 100
    pct_h = dh / CANVAS_H * 100
    sz = os.path.getsize(dst_path) // 1024
    return f'{sz}KB cat={pct_w:.1f}%W x {pct_h:.1f}%H'

def main():
    actions = sys.argv[1:] if len(sys.argv) > 1 else list(POSES.keys())
    print(f'Actions: {actions}', flush=True)
    
    # Phase 1: Generate all candidates
    selected = {}
    for name in actions:
        pose_desc = POSES[name]
        prompt = SUBJ + pose_desc + BG_TEXT + NEG
        print(f'\n[{name}] prompt={len(prompt)}ch', flush=True)
        
        best_idx = -1
        for i in range(CANDIDATES):
            img = gen_image(prompt)
            if img is None:
                print(f'  c{i}: FAILED', flush=True)
                continue
            cand_path = f'{OUT_DIR}/ref_{name}_c{i}.png'
            img.save(cand_path)
            sz = os.path.getsize(cand_path) // 1024
            ok, msg = audit_anatomy(img)
            print(f'  c{i}: {sz}KB audit={ok} ({msg})', flush=True)
            if ok and best_idx < 0:
                best_idx = i
            time.sleep(2)
        
        if best_idx < 0:
            best_idx = 0
        selected[name] = best_idx
        print(f'  [{name}] selected c{best_idx}', flush=True)
    
    # Phase 2: Clean up non-selected candidates, process with BiRefNet
    print('\n--- Processing with BiRefNet ---', flush=True)
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
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    
    for name in actions:
        idx = selected[name]
        cand = f'{OUT_DIR}/ref_{name}_c{idx}.png'
        dst = f'{OUT_DIR}/ref_{name}.png'
        if not os.path.exists(cand):
            print(f'  [{name}] no candidate, skipping', flush=True)
            continue
        # Backup existing ref
        if os.path.exists(dst):
            bk = f'{OUT_DIR}/backup/ref_{name}.png'
            if not os.path.exists(bk):
                os.rename(dst, bk)
                print(f'  [{name}] backed up old ref', flush=True)
            else:
                os.remove(dst)
        result = process_image(model, tf, cand, dst)
        print(f'  [{name}] {result}', flush=True)
        # Clean candidates
        for i in range(CANDIDATES):
            p = f'{OUT_DIR}/ref_{name}_c{i}.png'
            if os.path.exists(p):
                os.remove(p)
    
    del model
    torch.cuda.empty_cache()
    print('\nALL DONE', flush=True)

if __name__ == '__main__':
    main()
