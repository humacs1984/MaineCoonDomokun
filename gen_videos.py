# -*- coding: utf-8 -*-
"""Phase 2: Video generation for Maine Coon Domo-kun desktop pet.
12 basic actions using 3 reference images by pose group.

Usage: env -u PYTHONPATH -u PYTHONHOME C:/Users/humac/anaconda3/python.exe gen_videos.py [action ...]
"""
import sys, os, time, json, base64
import requests

ROOT = os.path.dirname(os.path.abspath(__file__))

# ══════════ CONFIG ══════════
KEYHEX = os.path.join(ROOT, 'keyhex.txt')
REFS_DIR = os.path.join(ROOT, 'darkblue_refs_new')
VIDEOS_DIR = os.path.join(ROOT, 'videos')

# Reference image mapping by pose group + per-action special refs
FRONT_VIEW = {'idle', 'sit', 'surprised', 'wave', 'beg', 'dance', 'pet'}
SIDE_VIEW = {'walk', 'run', 'eat', 'bark', 'happy', 'lick', 'stretch'}
DOWN_VIEW = {'sleep', 'roll', 'play_dead'}
# Per-action special ref images (t2i generated with prop in frame)
SPECIAL_REFS = {'eat': 'ref_eat.png', 'type': 'ref_type.png', 'bath': 'ref_bath.png', 'walk': 'ref_walk.png', 'pet': 'ref_pet.png'}

def get_ref(name):
    # Special ref (with prop) takes priority
    if name in SPECIAL_REFS:
        p = os.path.join(REFS_DIR, SPECIAL_REFS[name])
        if os.path.exists(p):
            return p
    if name in FRONT_VIEW:
        p = os.path.join(REFS_DIR, 'ref_front.png')
    elif name in SIDE_VIEW:
        p = os.path.join(REFS_DIR, 'ref_side.png')
    elif name in DOWN_VIEW:
        p = os.path.join(REFS_DIR, 'ref_down.png')
    else:
        p = os.path.join(REFS_DIR, 'ref_side.png')
    if not os.path.exists(p):
        raise FileNotFoundError(f'Ref not found: {p}')
    return p

# SUBJ: reference image encodes appearance — prompt anchors breed + clothing only
SUBJ = (
    'A 1-year-old Maine Coon cat wearing a white Domo-kun space-suit pet vest '
    '(cyan collar and zipper, orange chest strap with D-rings, orange-piped panels, '
    'D-badge and Domo-kun patch) at ALL times. '
    'The appearance must match the reference image exactly. '
)

NEG_BREED = (
    'different cat, wrong breed, '
    'stretched tall, thin body, elongated body, long thin legs, '
    'collar, leash, naked cat, cat without vest, vest removed, '
    'vest disappearing, vest changing color, vest changing shape'
)

COMMON = (' The subject stays perfectly centered in the same spot the whole time, not moving '
          'forward at all. The ENTIRE body from ear tips to bottom of all four paws is fully '
          'visible inside the frame at ALL times with generous empty margin. '
          'Extreme wide shot, subject takes up less than 40 percent of frame height. '
          'Static locked camera, solid dark navy blue background (color #000D43), '
          'soft even lighting, photorealistic, sharp crisp fur detail.')

COMMON_WALK = (' The ENTIRE body from ear tips to bottom of all four paws is fully '
               'visible inside the frame at ALL times with generous empty margin. '
               'Extreme wide shot, subject takes up less than 40 percent of frame height. '
               'The camera pans at the same speed as the cat, keeping it centered. '
               'Static locked camera height, solid dark navy blue background (color #000D43), '
               'soft even lighting, photorealistic, sharp crisp fur detail.')

# kiss专用COMMON（需要走近+特写，与通用COMMON冲突）
COMMON_KISS = (' The ENTIRE background and floor is one UNIFORM SOLID dark navy blue '
               'studio backdrop (color #000D43), with soft even lighting, photorealistic, '
               'sharp crisp fur detail. Static locked camera at the cat\'s eye level: '
               'the camera never moves or zooms; any change in subject size comes ONLY '
               'from the cat walking closer to or farther from the camera.')

NEG_KISS = ('cartoon, ugly, deformed, mutated, subtitles, watermark, text, logo, blurry, '
            'jittery, distorted, inconsistent appearance, '
            'extra legs, extra tail, '
            'other animals, '
            'different cat, wrong breed, '
            'illustration, 3d render, cgi, animation style, anime, low quality, '
            + NEG_BREED)

NEG_BASE = ('cartoon, ugly, deformed, mutated, subtitles, watermark, text, logo, blurry, '
            'jittery, distorted, inconsistent appearance, '
            'extra legs, extra tail, '
            'other animals, human, person, '
            'close up, zoomed in, filling frame, large subject, '
            'cropped, cut off, '
            'moving forward, walking forward, changing position, '
            'zooming in, growing bigger, camera moving, '
            + NEG_BREED)

COAT_CLAUSE = ' Fur and vest colors remain perfectly consistent in every frame.'

# LEGS_POS: unified leg constraint appended to all states with standing segments
LEGS_POS = (' Whenever the cat stands or rises on its legs, all four legs are straight down '
            'and clearly separated, the two hind legs close together and parallel, never '
            'splayed wide or twisted.')
LEGS_NEG = (', splayed legs, wide stance, legs apart, five legs, extra legs, '
            'splayed hind legs, twisted legs, unnatural legs')

# ── Actions (cat-specific wording) ──
ACTIONS = {
    # --- FRONT VIEW ---
    'idle': (
        ' Stands UPRIGHT on all four legs facing the camera the whole time, '
        'gently swaying with occasional ear twitches and slow tail swish. '
        'Four legs straight down, belly well above the floor. '
        'Calm confident alert expression: eyes bright and attentive looking straight ahead, '
        'ears slightly forward with interest, mouth gently closed with a subtle content look, '
        'radiating quiet self-assured poise.',
        ', side profile, lying down, sitting, '
        'five legs, extra legs, paws out of frame, cropped'
    ),
    'sit': (
        ' Calmly sits down facing the camera within the first second and REMAINS '
        'SITTING: hind legs folded under body out of sight, front legs straight down, '
        'tail curled around paws. '
        'While sitting, it shifts weight slightly from paw to paw, tilts head '
        'curiously side to side, blinks slowly, one ear flicks then the other, '
        'tail tip gives a lazy flick. '
        'Expression shifts naturally: mostly warm contented — eyes half-closed in '
        'relaxed bliss with a tiny gentle smile, ears relaxed — but occasionally '
        'perks up with bright attentive eyes as if noticing something interesting, '
        'then settles back into contentment. '
        'Never fully static or frozen, always subtly alive.',
        ', lying down, standing up, walking, '
        'paws out of frame, cropped, large subject, filling frame'
    ),
    'surprised': (
        ' TINY CAT SEEN FROM VERY FAR AWAY, extreme wide shot. It sits calmly facing the camera, then INSTANTLY gets startled in a FAST '
        'EXPLOSIVE reaction: ears shoot straight up in a split second, eyes go '
        'impossibly wide open, mouth drops open in shock, body recoils with a '
        'quick startled hop backward, tail puffs up and sticks straight out — '
        'all happening FAST and sharp within 1-2 seconds. '
        'Then stays frozen in wide-eyed stunned disbelief for the rest, '
        'occasionally blinking in confusion. '
        'The startle reaction must be QUICK and SUDDEN, NOT slow or gradual. '
        'CRITICAL SIZE CONSTRAINT: the entire cat is VERY SMALL in the frame, taking up '
        'less than 35 percent of the frame height, with LOTS of empty space above '
        'the head and below the paws. '
        'NO human, NO hands, the cat is completely alone.',
        ', hand, hands, fingers, arm, person, '
        'slim, lean, stretched tall, thin body, elongated body, long thin legs, '
        'five legs, extra legs, lying down, slow reaction, slow motion'
    ),
    'wave': (
        ' Sits facing the camera: hind legs tucked under, chest up. Raises ONE front paw '
        'beside its head and waves it side to side in a friendly greeting. '
        'The raised paw moves in SLOW SMOOTH clean arcs with NO trailing ghost '
        'or afterimage, the paw is always a single clean paw shape with crisp edges. '
        'Only the wrist and paw move, the arm stays at the same height. '
        'The paw always faces sideways toward the viewer with paw pad visible. '
        'The other three legs stay on the floor. '
        'Friendly eager expression: eyes sparkling with warmth, '
        'mouth open in a happy little grin, ears perked forward enthusiastically, '
        'face beaming with cheerful welcome. '
        'FULL BODY VISIBLE AT ALL TIMES. '
        'CONSISTENT STABLE FRAMING, camera stays FIXED.',
        ', standing up, walking, both paws raised, '
        'five legs, extra legs, '
        'zooming, camera pan, paws out of frame, '
        'motion blur on paw, trailing paw, ghost paw, afterimage, '
        'blurred paw, finger pointing up'
    ),

    # --- SIDE VIEW ---
    'walk': (
        ' ALREADY in a full EXACT SIDE PROFILE VIEW from the VERY FIRST FRAME (no '
        'front-facing approach, no turning toward the camera at any moment), the muzzle points to '
        'the right, only ONE eye is visible, the tail extends to the left. The camera '
        'sees the cat in a PERFECT 90-DEGREE SIDE PROFILE the ENTIRE video: the full length '
        'of the body from chest to tail is visible, the body silhouette looks WIDE and '
        'ELONGATED horizontally (clearly wider than tall), absolutely NO three-quarter '
        'view, NO oblique angle, NO chest or face turning toward the viewer at any '
        'moment. The cat '
        'performs a FULL natural walk gait with clear strides: '
        'diagonal legs alternating — when the left front and right hind lift together, '
        'the right front and left hind push off the ground, creating a clear rhythmic '
        'walk pattern. Each paw lifts well off the ground with visible leg extension '
        'every step, hind legs pushing off firmly behind the hip. '
        'SLOW CALM WALK, NOT running, NOT galloping — a leisurely relaxed pace. The camera pans at the '
        'same speed as the cat, keeping it centered in the frame the entire time. '
        'Tail gently swaying. The head and back stay level and horizontal the entire '
        'time; the cat stays UPRIGHT on all four straight legs, chest high, belly well '
        'above the floor. The cat keeps WALKING CONTINUOUSLY without stopping: no '
        'pausing, no standing still, no sitting, no hopping, every single frame of the '
        'video shows active stride motion with legs swinging, from the first frame to '
        'the last frame, an unbroken rhythmic walk. '
        'The fur color and vest color remain perfectly consistent in every single frame '
        'with zero flickering or color shift. The coat is clean uniform silver-gray tabby on '
        'every part of the body throughout the entire video. Smooth steady motion with '
        'no jerky or sudden changes between frames. Exactly ONE tail, never two tails.',
        ', front view, facing camera, standing still, static legs, stiff legs, locked '
        'legs, tiny steps, shuffling, legs together, splayed hind legs, twisted legs, '
        'unnatural legs, moving across the frame, drifting right, drifting left, '
        'moving left, moving right, turning back, turning around, changing direction, '
        'walking backward, horizontal drift, position shifting, traveling, '
        'head turning, head rotating, looking at camera, both eyes visible, '
        'three-quarter view, oblique angle, chest facing viewer, face toward camera, '
        'jerky motion, sudden brightness change, frame flicker, color pop, '
        'two tails, extra tail, duplicate tail, forked tail, '
        'running, galloping, trotting fast, sprinting, '
        'five legs, extra legs, two tails, treadmill'
    ),
    'run': (
        ' It runs at FULL SPEED in an EXACT SIDE PROFILE VIEW facing right: the muzzle '
        'points to the right, only ONE eye is visible, the tail extends to the left. On '
        'the floor it performs CONTINUOUS large-amplitude gallop '
        'strides without pause: every single stride the legs stretch far forward and '
        'far backward, a clear suspension moment with all four paws off the ground, '
        'then legs tucking under the body, stride after stride at the same big '
        'amplitude, powerful energetic sprint, ears and fur flowing, body holding the '
        'same screen position the whole time. The head and back stay level and '
        'horizontal the entire time. '
        'Wild exhilarated expression: mouth wide open with tongue flapping out the side, '
        'eyes squinting with pure thrill and joy.'
        ' Fur and vest colors remain perfectly consistent in every frame.',
        ', front view, facing camera, standing still, standing, static legs, stiff '
        'legs, locked legs, tiny steps, shuffling, legs together, hopping, hop, '
        'bouncing in place, treadmill, belt, black strip, dark strip, platform, '
        'machine, prop, object on floor, head '
        'down, head lowered to floor, butt up, rear raised, play bow, bowing, '
        'stretching down, sniffing floor, sniffing ground, nose to floor, lying down, '
        'lying, prone, sitting down, head turned to camera, looking at viewer, both '
        'eyes visible, five legs, extra legs, two tails'
    ),
    'eat': (
        ' Stands in an EXACT SIDE PROFILE VIEW facing right: only ONE eye is '
        'visible, the tail extends to the left, the whole body seen strictly from the '
        'side at ALL times, front legs vertical and parallel, hind legs close together. '
        'A small red plastic food bowl filled with brown kibble sits on the floor right '
        'in front of its chest, resting firmly on the floor at floor level. It lowers its '
        'head IN PROFILE and eats FROM the bowl: the muzzle goes down INTO the red bowl, '
        'the nose disappearing inside the bowl rim while chewing, head bobbing at bowl '
        'level 80 percent of the time, lifting the head only briefly with a piece of '
        'kibble in the mouth then returning into the bowl, tail wagging gently, the body '
        'and the bowl staying in the exact same spot the whole time. '
        'Satisfied contented expression while eating: eyes half-closed in gustatory bliss, '
        'mouth busy chewing with visible enjoyment, ears relaxed and droopy, '
        'face showing pure food happiness.',
        ', food scattered on floor, kibble on ground, eating from floor, licking '
        'floor, crumbs on floor, floating bowl, bowl in air, moving bowl, multiple '
        'bowls, giant bowl, empty floor, white bowl, '
        'front view, facing camera, both eyes visible, '
        'three quarter view, chest toward camera, front legs spread, splayed '
        'front legs, splayed legs, legs apart, wide stance, head up, looking around, '
        'standing alert, sniffing air, head turned to camera, looking at viewer, '
        'white food, milk, white liquid in bowl, shrinking bowl, changing bowl size, '
        'bowl morphing, five legs, extra legs'
    ),
    'type': (
        ' Sits behind a MINI RETRO TYPEWRITER-STYLE MECHANICAL KEYBOARD with ROUND '
        'PASTEL MACARON-COLORED KEYCAPS: each keycap is a small PERFECT CIRCLE like a '
        'candy, arranged in neat even rows. CRITICAL: the entire keyboard base, shell, '
        'housing and body is BRIGHT CREAM WHITE — clearly light-colored and bright, '
        'NEVER dark, NEVER black, NEVER gray, NEVER navy. The keycaps are in soft pastel '
        'colors: pale pink, butter yellow, mint green, baby blue, and cream white. '
        'The spacebar is a longer OBLONG PILL-SHAPE in pale pink. '
        'The keyboard is COMPACT and sits flat on the floor in front of the cat, '
        'SMALLER than the cat body. Both front paws rest on the nearest keycap rows '
        'and ALTERNATE PRESSING THE KEYS in a clear rhythmic left-right-left-right '
        'tapping motion. The head is slightly lowered watching the keys. '
        'Concentrated focused expression: eyes narrowed with laser focus on the keys, '
        'mouth gently closed with quiet determination, ears slightly forward, '
        'face showing adorable serious dedication to the important task of typing. '
        'The FACE AND EYES stay PERFECTLY STABLE the whole time, no eye flicker. '
        'The keyboard and EVERY keycap stay PERFECTLY STATIC AND IDENTICAL in every '
        'single frame: same size, same position, same colors, same shape, never moving, '
        'never growing or shrinking, never morphing or changing.',
        ', keyboard reversed, keyboard facing camera, keyboard moving, keyboard growing, '
        'keyboard shrinking, morphing keyboard, changing keyboard, floating keyboard, '
        'multiple keyboards, giant keyboard, paws off keyboard, transparent paws, '
        'ghost paws, frozen paws, paws not moving, flickering eyes, shifting catchlights, '
        'glowing eyes, bright eyes, '
        'standing up, walking, lying down, '
        'square keycaps, rectangular keycaps, black keyboard, dark keyboard, '
        'dark keyboard base, black keyboard base, gray keyboard, navy keyboard, '
        'standard keyboard, realistic keyboard, beads, pearl beads, '
        'floor, ground, surface, discs, circles, orbs, '
        'five legs, extra legs'
    ),
    'bath': (
        ' Sits DEEP INSIDE a small RED plastic bathtub placed on the floor, '
        'the tub resting firmly on the floor. The tub is filled to the rim with '
        'thick soft white soap foam; the lower body and legs are fully submerged under '
        'the foam so only the chest, front legs and head rise above the foam line, the '
        'foam wrapping snugly around the chest like a soft white blanket. The fur is '
        'VISIBLY WET everywhere: damp clumped strands sticking to the body, darker '
        'soaked silver-grey coat on the head, ears and back, clearly wet and sleek, never '
        'dry or fluffy. The foam clings to the wet fur and melts softly into it: a '
        'soft foam mound on top of the head blends into the damp hair with soft blurry '
        'edges, and a few small foam patches on the back and chest fade gradually into '
        'the wet coat, no hard outline anywhere. It wiggles gently in the foam, the '
        'foam surface rippling softly, then shakes its head once sending tiny water '
        'droplets flying, the red tub and the foam staying in the exact same spot the '
        'whole time. The cat NEVER stands up, NEVER steps out of the tub, NEVER '
        'stretches its body tall — it stays compact and seated inside the tub the entire video. '
        'Relaxed contented expression: eyes half-closed in comfortable bliss, '
        'mouth gently closed with a subtle happy smile, ears slightly droopy and wet, '
        'face showing warm satisfied enjoyment of the bath.',
        ', dry fur, fluffy dry coat, standing dry fur, foam hat, white cap on '
        'head, foam sticker, hard-edged foam patch, detached foam, floating foam '
        'clumps, paws on tub rim, front paws over rim, leaning on rim, lying on rim, '
        'empty tub, no foam, no bubbles, dry body, '
        'blue foam, blue-gray foam, gray foam, colored foam, '
        'foam same color as fur, '
        'hand, hands, fingers, person, '
        'human, floating tub, moving tub, multiple tubs, giant tub, '
        'white tub, blue tub, '
        'standing up, standing, walking, stepping out, getting out, '
        'stretching up, tall pose, elongated body, '
        'lying down, five legs, extra legs'
    ),
    # --- INTERACTIVE: bark (side view, no prop) ---
    'bark': (
        ' Stands UPRIGHT on all four straight legs in an EXACT SIDE PROFILE VIEW '
        'facing right: muzzle points right, only ONE eye visible, tail on the left, '
        'four legs straight down and clearly separated, the two hind legs close '
        'together and parallel. It barks loudly several times while staying upright in '
        'profile: mouth wide open and closing rhythmically with the muzzle still '
        'pointing right, head lifting slightly with each bark, body energetic, chest '
        'high, belly well above the floor. The head NEVER turns toward the camera. '
        'Fierce dramatic barking expression: mouth stretched WIDE open showing teeth, '
        'eyes intense and blazing with territorial alertness, ears pinned back sharply, '
        'face radiating bold confident warning and authority.',
        ', head turned to camera, looking at viewer, both eyes visible, facing camera, '
        'front view, three quarter view, splayed legs, wide stance, five legs, '
        'extra legs, lying down, sitting, closed mouth, mouth shut'
    ),
    # --- INTERACTIVE: beg (front view, upright on hind legs) ---
    'beg': (
        ' ALREADY standing up on its hind legs holding a begging pose, '
        'both front paws raised and staying STILL in that position, '
        'looking up expectantly. It HOLDS this begging pose in place without moving. '
        'The body ALWAYS has EXACTLY FOUR LEGS: two hind legs on the ground, two front paws raised. '
        'FULL BODY VISIBLE AT ALL TIMES. '
        'CONSISTENT STABLE FRAMING, camera stays FIXED. '
        'Hopeful eager begging expression: eyes HUGE and round with desperate longing, '
        'mouth slightly open in an adorable pleading whimper, ears pushed forward eagerly, '
        'face radiating irresistible "please please please" cuteness.',
        ', feet cut off, paws out of frame, cropped, '
        'walking toward camera, approaching camera, getting closer, zooming in, '
        'lying down, sitting, '
        'five legs, extra legs'
    ),
    # --- INTERACTIVE: dance (front view, upright on hind legs) ---
    'dance': (
        ' TINY CAT SEEN FROM VERY FAR AWAY, extreme wide shot. It dances playfully BALANCED UPRIGHT on its hind legs the whole time, body '
        'vertical with chest high and both front paws raised waving in the air, '
        'stepping and bouncing rhythmically in place. '
        'The body ALWAYS has EXACTLY FOUR LEGS: two hind legs on the ground, two front paws raised. '
        'CRITICAL SIZE CONSTRAINT: the entire cat is VERY SMALL in the frame, taking up '
        'less than 35 percent of the frame height, with LOTS of empty space above '
        'the head and below the paws. It never lies down, its belly '
        'never touches the floor. '
        'FULL BODY VISIBLE AT ALL TIMES. '
        'CONSISTENT STABLE FRAMING, camera stays FIXED. '
        'Ecstatic dancing expression: mouth wide open in the biggest grin, '
        'tongue hanging out joyfully, eyes sparkling with pure party excitement, '
        'ears bouncing with the rhythm, face radiating total euphoria and groove.',
        ', lying down, lying, prone, belly on floor, chest on floor, crouching, '
        'sitting down, sitting, '
        'slim, lean, stretched tall, thin body, elongated body, long thin legs, '
        'skinny cat, tall cat, narrow chest, narrow body, '
        'five legs, extra legs, zooming, camera pan, paws out of frame'
    ),
    # --- INTERACTIVE: pet (front three-quarter, human hand) ---
    'pet': (
        ' ONE single tail only. TINY CAT SEEN FROM VERY FAR AWAY, extreme wide shot. '
        'It sits calmly in a gentle three-quarter view, eyes closed with a content '
        'relaxed happy expression. The body is compact and well-proportioned. '
        'A single human hand is ALREADY RESTING on top of its head from the very first '
        'frame — palm down on the crown, fingers gently curved between the ears — and '
        'slowly strokes in repeated gentle petting motions; the cat leans its head '
        'into the palm, ears relaxing back, enjoying the petting, body staying compact '
        'in the same spot the whole time. The hand NEVER leaves the head. '
        'The subject takes up less than 35 percent of the frame height.',
        ', biting hand, licking hand, mouth open, teeth, fearful, scared, cowering, '
        'aggressive, growling, two hands, multiple hands, hand under chin, '
        'hand entering from left, hand from below, hand disappearing, '
        'two tails, extra tail, second tail, double tail, forked tail, split tail, '
        'slim, lean, stretched tall, thin body, elongated body, long thin legs, '
        'skinny cat, tall cat, narrow chest, narrow body, '
        'five legs, extra legs'
    ),
    # --- INTERACTIVE: kiss (walks toward camera, licks lens) ---
    'kiss': (
        ' The video starts with the cat FAR from the camera: a small full-body '
        'cat taking up about 25 percent of the frame height, standing in a '
        'gentle three-quarter view on a solid dark navy blue background. It then '
        'WALKS DIRECTLY TOWARD THE CAMERA with a happy bouncy walk, growing MUCH BIGGER '
        'frame after frame, until its muzzle and face become a LARGE CLOSE-UP taking '
        'about 70 percent of the frame height. It looks directly at the camera with '
        'warm pale green eyes and gives several affectionate licks toward the camera lens: '
        'the pink tongue VISIBLY EXTENDS from the mouth and touches toward the lens '
        'several times. After licking, it steps back to medium distance and '
        'wags its tail happily.'
        ' Fur and vest colors remain perfectly consistent in every frame.',
        ', sitting the whole time, never approaching, staying far away, same size '
        'forever, turning away, back to camera, lying down, prone, closed mouth '
        'forever, tongue hidden, head down, sniffing floor, side view, profile view'
    ),
    'happy': (
        ' Hops up and down joyfully in place in an EXACT SIDE PROFILE VIEW facing right, '
        'tail swishing rapidly. '
        'Overjoyed ecstatic expression: mouth wide open in the biggest grin, '
        'tongue hanging out, eyes sparkling with pure happiness, '
        'ears perked up, radiating unstoppable excitement and delight. '
        'EXACTLY FOUR LEGS at all times. '
        'CONSISTENT STABLE FRAMING, camera stays FIXED.'
        ' Fur and vest colors remain perfectly consistent in every frame.',
        ', motion blur, blurry, head turned to camera, facing camera, '
        'five legs, extra legs, zooming, camera pan, paws out of frame'
    ),
    'lick': (
        ' Sits in an EXACT SIDE PROFILE VIEW, raises ONE front paw close to chest, '
        'bends head DOWN so the open mouth is pressed right against the raised paw, '
        'and the pink tongue VISIBLY TOUCHES and strokes the paw fur, licking the paw '
        'repeatedly. Then it turns its head sideways and buries the muzzle into the '
        'shoulder and side fur, tongue extended and clearly in contact with the coat, '
        'licking the shoulder fur with rhythmic tongue movements. '
        'The mouth and tongue always stay in physical contact with the body or paw '
        'while licking, no gap between tongue and fur. '
        'Focused grooming expression: eyes narrowed in deep concentration, '
        'ears angled back slightly, face showing adorable determined focus '
        'on getting perfectly clean.',
        ', licking the air, tongue not touching body, mouth away from body, '
        'gap between mouth and fur, paw raised away from mouth, waving paw, '
        'tongue hanging out, front view, panting with head up'
    ),
    'stretch': (
        ' In an EXACT SIDE PROFILE VIEW the whole time, performs a full body '
        'stretch: lowering front chest to the ground with front paws extended '
        'forward, rear end raised high, holding the stretch briefly, then slowly '
        'standing back up. '
        'Satisfied blissful expression: mouth slightly open in a content sigh, '
        'eyes half-closed in pleasure, ears droopy and relaxed, '
        'face showing the pure bliss of a perfect morning stretch. '
        'Never turns toward camera.',
        ', front view, facing camera, looking at camera, turning toward camera'
    ),

    # --- DOWN VIEW ---
    'sleep': (
        ' Seen in an EXACT SIDE PROFILE VIEW, it sits down compactly with hind legs '
        'tucked under, yawns sleepily, then slowly curls down into a compact relaxed ball. '
        'Serene blissful sleeping expression: eyes gently closed, '
        'mouth softly relaxed with a tiny content smile, ears completely limp and relaxed, '
        'face radiating deep peaceful comfort and safety. '
        'Remains curled until the end.',
        ', splayed legs, legs apart, wide stance, '
        'front view, both eyes visible, standing up, '
        'mouth open, tongue out, panting, obese, bloated'
    ),
    'roll': (
        ' Seen from the SIDE the whole time, it gently lies down on its side and '
        'rolls slowly onto its back for a relaxed happy wiggle with legs loosely in '
        'the air, then rolls back to its side and stands up again. The roll stays a '
        'gentle partial roll in the side plane, the cat never turns its back to the '
        'camera and is never seen from behind. '
        'The fur color and vest color remain perfectly consistent in every single frame '
        'with zero flickering or color shift. Smooth steady motion with '
        'no jerky or sudden changes between frames. '
        'Playful blissful expression while rolling: mouth open in a silly happy grin, '
        'tongue lolling out, eyes bright and mischievous, '
        'ears flopped back, radiating carefree fun and silliness.',
        ', back facing camera, seen from behind, butt facing camera, extreme twist, '
        'contorted body, unnatural twisted pose, spine twisted, '
        'two tails, extra tail, duplicate tail, forked tail, '
        'five legs, six legs, extra legs, more than four legs'
    ),
    'play_dead': (
        ' Seen from the SIDE, QUICKLY flops down onto its SIDE in one fast '
        'dramatic drop — NOT a slow gentle descent but an instant comedic collapse — '
        'and lies completely still: all four legs go limp and floppy, head drops flat, '
        'eyes squeezed shut, tongue hanging out the side dramatically. '
        'Dramatic over-the-top playing-dead expression: hilariously theatrical and adorable. '
        'The flop-down must happen FAST within the first 1-2 seconds, then stays '
        'completely still in this sideways pose until the end.',
        ', standing up, sitting, head up, eyes open, walking, moving, slow fall, slow collapse'
    ),
}
# Append LEGS_POS to states with standing segments (same pattern as golden_vest_pet)
for _k in ('sleep', 'roll', 'stretch', 'happy', 'play_dead', 'eat', 'lick'):
    _d, _n = ACTIONS[_k]
    ACTIONS[_k] = (_d + LEGS_POS, _n + LEGS_NEG)

# ══════════ CONFIG END ══════════

BASE = 'https://api.agnes-ai.cn/v1'
tok = bytes.fromhex(open(KEYHEX).read().strip()).decode()
HDR = {'Authorization': tok, 'Content-Type': 'application/json'}

def submit(prompt, neg, ref_image, width=1440, height=1080):
    img = base64.b64encode(open(ref_image, 'rb').read()).decode()
    r = requests.post(f'{BASE}/video/generations', headers=HDR, json={
        'model': 'agnes-video-v2.0', 'prompt': prompt, 'negative_prompt': neg,
        'image': f'data:image/png;base64,{img}',
        'num_frames': 121, 'frame_rate': 24,
        'width': width, 'height': height,
    }, timeout=180)
    if r.status_code != 200:
        return None, f'HTTP {r.status_code}: {r.text[:200]}'
    d = r.json()
    return d.get('video_id') or d.get('task_id'), None

def poll(video_id, timeout_s=1800):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            d = requests.get('https://api.agnes-ai.cn/agnesapi',
                             params={'video_id': video_id},
                             headers={'Authorization': tok}, timeout=60).json()
        except Exception:
            time.sleep(15); continue
        st = (d.get('status') or '').lower()
        if st == 'completed':
            return d.get('url'), None
        if st == 'failed':
            return None, json.dumps(d.get('error'), ensure_ascii=False)[:300]
        print(f'    {st} {d.get("progress", "")}%', flush=True)
        time.sleep(20)
    return None, 'poll timeout'

def download_video(url, name):
    r = requests.get(url, timeout=300)
    out = os.path.join(VIDEOS_DIR, f'{name}_darkblue.mp4')
    open(out, 'wb').write(r.content)
    return out, len(r.content)

if __name__ == '__main__':
    os.makedirs(VIDEOS_DIR, exist_ok=True)
    done = {f.replace('_darkblue.mp4', '') for f in os.listdir(VIDEOS_DIR) if f.endswith('_darkblue.mp4')}
    only = sys.argv[1:] or [k for k in ACTIONS]
    todo = [x for x in only if x not in done]
    print(f'[start] {len(todo)} videos to generate', flush=True)

    k = 0
    for name in only:
        if name in done:
            print(f'[skip] {name} exists', flush=True); continue
        k += 1
        if k > 1:
            print('  ... waiting 70s rate limit ...', flush=True)
            time.sleep(70)

        desc, extra_neg = ACTIONS[name]
        ref_image = get_ref(name)

        if name == 'kiss':
            prompt = SUBJ + desc + COMMON_KISS
            neg = NEG_KISS + extra_neg
        elif name == 'walk':
            prompt = SUBJ + desc + COMMON_WALK
            neg = (NEG_BASE + extra_neg).replace(
                'moving forward, walking forward, changing position, ', '')
        elif name == 'pet':
            # 摸摸头互动必须有人手出现——豁免 NEG_BASE 的 human/person 排除词
            prompt = SUBJ + desc + COMMON
            neg = (NEG_BASE + extra_neg).replace('human, person, ', '').replace(', human, person', '')
        else:
            prompt = SUBJ + desc + COMMON
            neg = NEG_BASE + extra_neg

        vid, err = submit(prompt, neg, ref_image=ref_image)

        for attempt in range(6):
            if vid:
                break
            wait = 75 * (2 ** attempt)
            print(f'  [retry {attempt+1}/6] {err} — wait {wait}s', flush=True)
            time.sleep(wait)
            # kiss重试用COMMON_KISS（v130d教训：用通用COMMON导致非走近视频）
            if name == 'kiss':
                vid, err = submit(SUBJ + desc + COMMON_KISS, neg, ref_image=ref_image)
            else:
                vid, err = submit(prompt, neg, ref_image=ref_image)

        if not vid:
            print(f'[FAIL submit] {name}: {err}', flush=True); continue

        print(f'[submit] {name} video_id={vid}', flush=True)
        url, err = poll(vid)
        if not url:
            print(f'[FAIL poll] {name}: {err}', flush=True); continue

        out_path, size = download_video(url, name)
        print(f'  SAVED {name}.mp4 ({size} bytes)', flush=True)

    print('VIDEOS_DONE', flush=True)
