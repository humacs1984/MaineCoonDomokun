# -*- coding: utf-8 -*-
"""Phase 1a: Identity candidate image generation for Domo-kun Maine Coon cat.
Generates 3 front-view candidates for user selection.

Usage: env -u PYTHONPATH -u PYTHONHOME C:/Users/humac/anaconda3/python.exe gen_identity.py
"""
import base64, json, os, io, time
from PIL import Image
import requests

BASE = 'https://api.agnes-ai.cn'
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, 'identity_candidates')
os.makedirs(OUT, exist_ok=True)

# ══════════ CONFIG ══════════
KEYHEX = os.path.join(ROOT, 'keyhex.txt')

# SUBJ: Global subject description — reuse VERBATIM in all video generation prompts.
# ⚠️ Keep concise (≤800ch). Over-description confuses model and may trigger content filter.
SUBJ = (
    'A 1-year-old Maine Coon cat, silver tabby fur with silvery-white undercoat '
    'and soft grey stripes, pale green eyes, large pointed ears with dark lynx-tip tufts, '
    'pink nose, fluffy white ruff, bushy plumed smoky-grey tail. '
    'Wearing a white Domo-kun space-suit pet vest with cyan stand-up collar and center zipper, '
    'orange chest strap with two silver D-rings, orange-piped front panels each with a small '
    'cyan zipper, a round black-and-white D-badge on the left panel, '
    'a brown Domo-kun character patch on the right panel.'
)

# Avoid words: naked, groin, anatomical terms — trigger content filter
AVOID = (
    'extra legs, extra paws, five legs, two tails, extra tail, second tail, '
    'deformed, mutated, watermark, text, blurry, close up, zoomed in, '
    'cartoon, short-haired, '
    'Siamese, Persian, British Shorthair, Sphynx, Bengal, '
    'hood, sleeves, leg sleeves, collar only, red bandana'
)

VARIANTS = [
    ' Sitting facing the camera in a front view, front paws on the ground, '
    'hind legs tucked under body out of sight, calm confident expression. '
    'Four legs total, one tail curving to side, two large pointed ears with lynx tips.',

    ' Sitting in a slight three-quarter front view, head turned toward camera, '
    'curious expression with wide-open eyes, front paws neatly side by side, '
    'hind legs tucked under body out of sight. Four legs, one tail, two pointed ears.',

    ' Sitting facing the camera, adorable face with big round green eyes, '
    'fluffy white ruff framing the face, front paws on the ground, '
    'hind legs tucked under body out of sight. Four legs, one tail, two ears with lynx tips.',
]

PICK_CRITERIA = (
    '1) Maine Coon kitten (6 months): large pointed ears with lynx tufts, square muzzle, '
    'fluffy ruff, bushy tail, silver tabby coat. 2) Kitten not adult. '
    '3) Pale greenish-gold eyes. 4) Wearing Domo-kun vest: white body, cyan collar+zipper, '
    'orange strap with D-rings, orange-piped panels with D-badge and Domo-kun patch. '
    '5) Sitting facing camera, front paws visible, hind legs tucked. '
    '6) Full body visible. 7) Exactly 4 legs, 1 tail, 2 ears.'
)
# ══════════ CONFIG END ══════════

BG_TEXT = ' Dark navy blue background #000D43. Photorealistic ultra-sharp fur soft even lighting.'
COMMON = ' Centered. The subject takes up about 60 percent of frame height with some dark blue space around. Static camera.'

tok = bytes.fromhex(open(KEYHEX).read().strip()).decode()
HDR = {'Authorization': tok, 'Content-Type': 'application/json'}

print('Verifying API auth...', flush=True)
probe = requests.get(f'{BASE}/v1/models', headers=HDR, timeout=10)
print(f'  Auth: HTTP {probe.status_code}', flush=True)
if probe.status_code != 200:
    raise SystemExit(f'Auth failed: {probe.text[:200]}')

results = []
for i, desc in enumerate(VARIANTS):
    prompt = SUBJ + desc + BG_TEXT + COMMON + f' Avoid: {AVOID}'
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
    b64 = d.get('data', [{}])[0].get('b64_json')
    url = d.get('data', [{}])[0].get('url', '')
    if b64:
        img = Image.open(io.BytesIO(base64.b64decode(b64))).convert('RGBA')
    elif url:
        r2 = requests.get(url, timeout=60)
        img = Image.open(io.BytesIO(r2.content)).convert('RGBA')
    else:
        print(f'  no image data', flush=True)
        continue
    path = os.path.join(OUT, f'cand_{i}.png')
    img.save(path)
    print(f'  saved {path} {img.size}', flush=True)
    results.append({'i': i, 'path': path})
    time.sleep(3)

with open(os.path.join(OUT, 'results.json'), 'w') as f:
    json.dump(results, f, indent=1)
print(f'\nDONE: {len(results)} candidates in {OUT}', flush=True)
