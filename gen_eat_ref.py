# -*- coding: utf-8 -*-
"""Generate eat reference image: side profile + red food bowl."""
import requests, json, io, os, time, base64
from PIL import Image
import torch, numpy as np
from transformers import AutoModelForImageSegmentation
from torchvision import transforms

ROOT = os.path.dirname(os.path.abspath(__file__))
KEYHEX = os.path.join(ROOT, 'keyhex.txt')
OUT_DIR = os.path.join(ROOT, 'darkblue_refs_new')
CANDIDATES = 3

CANVAS_W, CANVAS_H = 1088, 832
BG_OUT = (18, 37, 69)
BG = np.array(BG_OUT, dtype=float)

tok = bytes.fromhex(open(KEYHEX).read().strip()).decode()
HDR = {'Authorization': tok, 'Content-Type': 'application/json'}

SUBJ = (
    'A 1-year-old Maine Coon cat, silver tabby fur with silvery-white undercoat '
    'and soft grey stripes, pale green eyes, large pointed ears with dark lynx-tip tufts, '
    'pink nose, neat white ruff, well-groomed sleek coat with clean neat edges and smooth '
    'outline, long plumed tail held neatly. '
    'Wearing a white Domo-kun space-suit pet vest with cyan stand-up collar and center zipper, '
    'orange chest strap with two silver D-rings, orange-piped front panels each with a small '
    'cyan zipper, a round black-and-white D-badge on the left panel, '
    'a brown Domo-kun character patch on the right panel. '
)

POSES = {
    'eat': (
        ' EXACT SIDE PROFILE VIEW facing right: only ONE eye visible, '
        'tail extends to the left. Standing on all four legs with head LOWED DOWN '
        'INTO a small RED plastic food bowl filled with brown kibble — the muzzle '
        'is INSIDE the bowl rim, nose touching the kibble, clearly eating from the bowl. '
        'ONE single tail only, exactly four legs. '
        'Dark navy blue studio background (color #000D43). '
        'The subject takes up about 45 percent of frame height.',
        ', fluffy, shaggy, messy fur, flyaway hairs, '
        'front view, facing camera, both eyes visible, '
        'extra legs, extra tail, five legs, two tails, second tail, double tail, forked tail, '
        'split tail, duplicated tail, '
        'without vest, vest removed, '
        'white bowl, blue bowl, empty bowl, '
        'head up, head raised, looking up, standing alert, '
        'food scattered on floor, kibble on ground, '
        'obese, bloated'
    ),
}

# BiRefNet processing
def process_candidate(img_pil):
    """BiRefNet matting + COM centering + dark blue canvas."""
    local_path = os.path.expanduser(
        '~/.cache/huggingface/hub/models--zhengpeng7--BiRefNet/snapshots/'
        'e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4'
    )
    if os.path.isdir(local_path):
        model = AutoModelForImageSegmentation.from_pretrained(
            local_path, trust_remote_code=True, local_files_only=True
        ).cuda().eval()
    else:
        model = AutoModelForImageSegmentation.from_pretrained(
            'ZhengPeng7/BiRefNet-matting', trust_remote_code=True
        ).cuda().eval()

    tf = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    img_rgb = img_pil.convert('RGB')
    inp = tf(img_rgb).unsqueeze(0).cuda()
    with torch.no_grad():
        sig = model(inp)[-1].sigmoid().squeeze().cpu().numpy()

    # Soft alpha three-zone
    h, w = img_rgb.size[1], img_rgb.size[0]
    alpha = np.zeros((h, w), dtype=np.uint8)
    sig_resized = np.array(Image.fromarray((sig * 65535).astype(np.uint16)).resize((w, h), Image.BILINEAR)).astype(np.float64) / 65535.0
    alpha[sig_resized > 0.7] = 255
    soft = (sig_resized > 0.01) & (sig_resized <= 0.7)
    alpha[soft] = np.clip(sig_resized[soft] * 255, 1, 254).astype(np.uint8)

    # Color-distance supplement for red bowl
    img_np = np.array(img_rgb)
    color_dist = np.sqrt(((img_np.astype(np.float64) - BG) ** 2).sum(axis=2))
    chroma_mask = color_dist > 20
    biref_zero = alpha == 0
    alpha[biref_zero & chroma_mask] = 255

    # ── Darkblue border removal (from golden_vest_pet process_darkblue.py) ──
    # Prong 1: Color-based alpha zeroing — remove darkblue pixels BiRefNet misclassified
    fg_mask = alpha > 0
    if fg_mask.any():
        fg_rgb = img_np[fg_mask].astype(np.float64)
        dist_db = np.sqrt(((fg_rgb - BG) ** 2).sum(axis=1))
        b_minus_g = fg_rgb[:, 2] - fg_rgb[:, 1]
        is_darkblue = (dist_db < 60) & (b_minus_g > 3)
        # Exclude BiRefNet core foreground (sigmoid > 0.5)
        fg_sigmoid = sig_resized[fg_mask]
        is_darkblue = is_darkblue & (fg_sigmoid <= 0.5)
        if is_darkblue.any():
            dark_full = np.zeros(alpha.shape, dtype=bool)
            dark_full[fg_mask] = is_darkblue
            alpha[dark_full] = 0

    # Prong 2: Protected erode + blur for thin darkblue border
    orig_fg_mask = alpha > 128
    from scipy import ndimage
    def erode_alpha(a, radius, protect_mask=None):
        struct = ndimage.generate_binary_structure(2, 1)
        eroded = ndimage.binary_erosion(a > 0, structure=struct, iterations=radius)
        result = a.copy()
        result[~eroded] = 0
        if protect_mask is not None:
            # Don't erode interior holes created by darkblue removal
            result[protect_mask & (a > 0) & ~eroded] = a[protect_mask & (a > 0) & ~eroded]
        return result
    def blur_alpha(a, sigma):
        blurred = ndimage.gaussian_filter(a.astype(np.float64), sigma=sigma)
        return np.clip(blurred, 0, 255).astype(np.uint8)

    ERODE_RADIUS = 4
    EDGE_BLUR_SIGMA = 1.8
    alpha = erode_alpha(alpha, ERODE_RADIUS, protect_mask=orig_fg_mask)
    alpha = blur_alpha(alpha, EDGE_BLUR_SIGMA)
    alpha[alpha > 250] = 255
    alpha[alpha < 10] = 0

    # COM centering + scale to 55-65% H
    solid = alpha > 128
    if not solid.any():
        return None
    ys, xs = np.where(solid)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    crop = img_rgb.crop((x0, y0, x1 + 1, y1 + 1))
    crop_a = Image.fromarray(alpha[y0:y1+1, x0:x1+1])

    crop_h, crop_w = crop.size[1], crop.size[0]
    target_h = int(CANVAS_H * 0.60)
    target_w = int(CANVAS_W * 0.85)
    scale = min(target_h / crop_h, target_w / crop_w)
    new_w = max(1, int(crop_w * scale))
    new_h = max(1, int(crop_h * scale))
    crop = crop.resize((new_w, new_h), Image.LANCZOS)
    crop_a = crop_a.resize((new_w, new_h), Image.LANCZOS)

    # COM x centering
    a_np = np.array(crop_a)
    weights = a_np.astype(np.float64)
    weights[weights < 128] = 0
    w_sum = weights.sum()
    com_x = (weights * np.arange(new_w)).sum() / w_sum if w_sum > 0 else new_w / 2

    canvas = Image.new('RGBA', (CANVAS_W, CANVAS_H), (*BG_OUT, 255))
    rgba = crop.convert('RGBA')
    rgba.putalpha(crop_a)

    paste_x = int(CANVAS_W / 2 - com_x)
    # Bottom anchor
    paste_y = CANVAS_H - new_h - 14
    if paste_y < 0:
        paste_y = 0
    # Adjust for COM x
    paste_x = max(0, min(CANVAS_W - new_w, paste_x))

    canvas.paste(rgba, (paste_x, paste_y), rgba)
    return canvas.convert('RGB')


def generate():
    os.makedirs(OUT_DIR, exist_ok=True)
    for pose_name, (desc, neg) in POSES.items():
        prompt = SUBJ + desc
        for i in range(CANDIDATES):
            print(f'  Generating {pose_name} candidate {i}...', flush=True)
            r = requests.post('https://api.agnes-ai.cn/v1/images/generations',
                              headers=HDR, json={
                'model': 'agnes-image-2.1-flash',
                'prompt': prompt,
                'size': '1024x1024',
            }, timeout=120)
            if r.status_code != 200:
                print(f'    FAIL: {r.status_code} {r.text[:200]}')
                continue
            data = r.json()
            url = data.get('url') or data.get('data', [{}])[0].get('url', '')
            if not url:
                print(f'    FAIL: no URL in response')
                continue
            img_resp = requests.get(url, timeout=60)
            img = Image.open(io.BytesIO(img_resp.content)).convert('RGB')
            cand_path = os.path.join(OUT_DIR, f'ref_{pose_name}_c{i}.png')
            img.save(cand_path)
            print(f'    SAVED {cand_path}')
            if i < CANDIDATES - 1:
                time.sleep(5)
        print(f'  Done generating {pose_name} candidates. Please select one for BiRefNet processing.')


if __name__ == '__main__':
    import sys
    if '--process' in sys.argv:
        # Process a specific candidate: python gen_eat_ref.py --process ref_eat_c1
        idx = sys.argv.index('--process')
        src = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None
        if not src:
            print('Usage: --process <candidate_filename>')
            sys.exit(1)
        src_path = os.path.join(OUT_DIR, src)
        if not os.path.exists(src_path):
            print(f'Not found: {src_path}')
            sys.exit(1)
        img = Image.open(src_path)
        result = process_candidate(img)
        if result:
            out_path = os.path.join(OUT_DIR, 'ref_eat.png')
            result.save(out_path)
            print(f'SAVED {out_path}')
        else:
            print('BiRefNet failed: no foreground detected')
    else:
        generate()
