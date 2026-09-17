# -*- coding: utf-8 -*-
"""Generate down/lying pose reference candidates for Maine Coon Domo-kun cat.

Usage: env -u PYTHONPATH -u PYTHONHOME C:/Users/humac/anaconda3/python.exe gen_down_ref.py
"""
import base64, json, os, io, time, sys
import requests, numpy as np
from PIL import Image

BASE = 'https://api.agnes-ai.cn'
ROOT = os.path.dirname(os.path.abspath(__file__))

KEYHEX = os.path.join(ROOT, 'keyhex.txt')

SUBJ_DOWN = (
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
    'hood, sleeves, leg sleeves, collar only, red bandana, obese, bloated'
)

DOWN_POSES = [
    ' Lying on its side in an exact side profile view facing right, curled up peacefully, '
    'exactly four legs tucked in, one single tail wrapped around body, two pointed ears. '
    'Serene sleeping expression: eyes gently closed, mouth softly relaxed, '
    'ears completely limp and relaxed.',

    ' Lying on its side in an exact side profile view facing right, all four legs relaxed '
    'and extended, one single tail, two pointed ears with lynx tips. '
    'Peaceful resting expression: eyes half-open drowsy, calm and content.',

    ' Lying on its side in an exact side profile view, curled into a neat compact ball, '
    'exactly four legs tucked under body, one single tail curled around, two pointed ears. '
    'Calm relaxed expression, neatly groomed coat visible.',
]

BG_TEXT = ' Dark navy blue background #000D43. Photorealistic ultra-sharp fur soft even lighting.'
COMMON = ' Centered. The subject takes up about 60 percent of frame height with some dark blue space. Static camera.'

tok = bytes.fromhex(open(KEYHEX).read().strip()).decode()
HDR = {'Authorization': tok, 'Content-Type': 'application/json'}

OUT_DIR = os.path.join(ROOT, 'darkblue_refs_new')
os.makedirs(OUT_DIR, exist_ok=True)

def audit_anatomy(img):
    arr = np.array(img.convert('RGB'))
    bg_ref = np.array([0, 13, 67], dtype=float)
    dist = np.sqrt(((arr.astype(float) - bg_ref)**2).sum(axis=2))
    fg_mask = dist > 35
    ys, xs = np.where(fg_mask)
    if len(ys) == 0:
        return False, "no fg"
    return True, f"fg_h={ys.max()-ys.min()+1}"

def main():
    action = 'down'
    print(f'Generating 3 candidates for ref_{action}...', flush=True)
    results = []
    for i, pose in enumerate(DOWN_POSES):
        prompt = SUBJ_DOWN + pose + BG_TEXT + COMMON + f' Avoid: {AVOID}'
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

    with open(os.path.join(OUT_DIR, f'ref_{action}_candidates.json'), 'w') as f:
        json.dump(results, f, indent=1)
    print(f'\nGenerated {len(results)} candidates. User selection needed.', flush=True)

if __name__ == '__main__':
    main()
