"""
Indian license-plate generator (HSRP Edition)
- Adds IND Blue Strip, Hologram, and Bolts
- Adds Directional Shadows and Glare
- Outputs images to data/images/*.jpg
"""

import os
import random
import string
import csv
from typing import Tuple, List
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance, ImageOps

# ----------------- CONFIG -----------------
OUT_DIR = Path("data")
IMG_DIR = OUT_DIR / "images"
LABELS_CSV = OUT_DIR / "labels.csv"
LABELS_TXT = OUT_DIR / "labels.txt"

N_SAMPLES = 1000
IMG_W, IMG_H = 320, 100 
JPG_QUALITY = 90

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Includes all Indian states and union territories commonly used on vehicle plates.
STATE_CODES = [
    # States
    "AP","AR","AS","BR","CG","GA","GJ","HR","HP","JH","KA","KL",
    "MH","MP","ML","MN","MZ","NL","OD","PB","RJ","SK","TN","TS",
    "TR","UP","UK","WB",

    # Union Territories & special codes
    "AN",  # Andaman & Nicobar
    "CH",  # Chandigarh
    "DL",  # Delhi
    "DN",  # Dadra & Nagar Haveli and Daman & Diu
    "JK",  # Jammu & Kashmir (admin code on plates)
    "LA",  # Ladakh
    "LD",  # Lakshadweep
    "PY"   # Puducherry
]
# --------------------------------------------------------------------

UPPER = string.ascii_uppercase
DIGITS = string.digits

STANDARD_RATIO = 0.65
VIP_PROB = 0.18

BG_REALISTIC = 0.60
BG_FADED = 0.25
BG_RANDOM = 0.15

WHITE_PLATE = (245, 245, 245)
YELLOW_PLATE = (255, 220, 30)
FADED1 = (235, 235, 210)
FADED2 = (230, 230, 230)
GRAY = (200, 200, 200)
COMMON_REALISTIC = [WHITE_PLATE, YELLOW_PLATE]
FADED_SET = [FADED1, FADED2, GRAY]

SEPARATORS = [' ', '-', '.']

# ----------------- FONT CONFIG (KAGGLE SYSTEM FONTS) -----------------
POSSIBLE_FONTS = [
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf"
]
AVAILABLE_FONTS = [f for f in POSSIBLE_FONTS if os.path.exists(f)]

# ----------------- UTIL -----------------
def rand_letters(n: int) -> str:
    return ''.join(random.choices(UPPER, k=n))

def maybe_sep() -> str:
    return random.choice(SEPARATORS)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def choose_background() -> Tuple[tuple, str]:
    r = random.random()
    if r < BG_REALISTIC:
        return random.choice(COMMON_REALISTIC), "realistic"
    elif r < BG_REALISTIC + BG_FADED:
        return random.choice(FADED_SET), "faded"
    else:
        return random_contrast_color((0,0,0)), "random"

def luminance(rgb):
    r, g, b = rgb
    return 0.299*r + 0.587*g + 0.114*b

def contrast_ok(bg: tuple, fg: tuple, threshold: float = 80.0) -> bool:
    return abs(luminance(bg) - luminance(fg)) >= threshold

def random_contrast_color(fg=(0,0,0), tries=50):
    for _ in range(tries):
        c = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        if contrast_ok(c, fg): return c
    return (180, 180, 180)

def choose_text_color(bg):
    dark = (0,0,0)
    light = (255,255,255)
    return dark if contrast_ok(bg, dark) else light

def join_dirty(parts: List[str]) -> str:
    out = []
    for i, p in enumerate(parts):
        out.append(p)
        if i < len(parts) - 1:
            if random.random() < 0.78:
                out.append(maybe_sep())
    s = ''.join(out)
    res = []
    prev = ''
    for ch in s:
        if ch == prev and ch in SEPARATORS: continue
        res.append(ch)
        prev = ch
    return ''.join(res)

# ----------------- PLATE GENERATION -----------------
def generate_plate_text() -> Tuple[str, str]:
    is_standard = random.random() < STANDARD_RATIO
    is_vip = random.random() < VIP_PROB
    state = random.choice(STATE_CODES)
    district = random.randint(1,99)

    if is_standard:
        series = rand_letters(2)
        number = random.randint(1, 9999)
        district_label = str(district) if is_vip else f"{district:02d}"
        number_label = str(number) if is_vip else f"{number:04d}"
        clean_label = state + district_label + series + number_label
        
        vis_district = str(district) if (is_vip and random.random() < 0.9) else f"{district:02d}"
        vis_number = str(number) if (is_vip and random.random() < 0.9) else f"{number:04d}"
        visual = join_dirty([state, vis_district, series, vis_number])
    else:
        # Simplified challenging branch logic
        if random.random() < 0.5:
            clean_label = state + str(district)
            visual = join_dirty([state, str(district)])
        else:
            series = rand_letters(2)
            number = random.randint(1, 99999)
            clean_label = state + str(district) + series + str(number)
            visual = join_dirty([state, str(district), series, str(number)])

    clean_label = ''.join(ch for ch in clean_label.upper() if ch in (UPPER + DIGITS))
    if len(clean_label) < 3: clean_label += random.choice(UPPER)
    if len(clean_label) > 12: clean_label = clean_label[:12]
    return visual, clean_label

def load_font(size=40):
    """
    Loads a font from the pre-installed system fonts on Kaggle.
    (This utility is needed by render_plate_image)
    """
    if not AVAILABLE_FONTS: return ImageFont.load_default()
    try:
        return ImageFont.truetype(random.choice(AVAILABLE_FONTS), size=size)
    except:
        return ImageFont.load_default()


def render_plate_image(visual_text: str, bg_color: tuple) -> Image.Image:
    text_color = choose_text_color(bg_color)
    img = Image.new("RGB", (IMG_W, IMG_H), bg_color)
    draw = ImageDraw.Draw(img)

    # 1. DRAW IND STRIP (The Blue Bar on the Left)
    has_ind_strip = random.random() < 0.85 # 85% chance of HSRP style
    strip_w = 0
    
    if has_ind_strip:
        strip_w = int(IMG_W * 0.12) # ~40 pixels
        draw.rectangle([0, 0, strip_w, IMG_H], fill=(0, 50, 150)) # Deep Blue
        
        # Add "IND" text small (if font loads)
        try:
            ind_font = load_font(size=int(IMG_H * 0.15))
            draw.text((5, int(IMG_H*0.75)), "IND", fill=(255,255,255), font=ind_font)
            
            # Add Fake Chakra/Hologram (Yellow-ish oval)
            draw.ellipse([5, 15, strip_w-5, 45], outline=(200, 200, 50), width=1)
        except: pass

    # 2. DRAW BOLTS/SCREWS (Black dots at corners)
    # Positions are relative to the plate, shifted past the IND strip
    bolt_color = (50, 50, 50)
    r = 3 # radius
    bolt_positions = [
        (strip_w + 15, 15), (IMG_W - 15, 15),
        (strip_w + 15, IMG_H - 15), (IMG_W - 15, IMG_H - 15)
    ]
    for bx, by in bolt_positions:
        if random.random() < 0.9: # 90% chance a bolt is visible
            draw.ellipse([bx-r, by-r, bx+r, by+r], fill=bolt_color)

    # 3. SETUP TEXT SPACE
    avail_w = IMG_W - strip_w - 10
    start_x = strip_w + 5
    
    font_size = int(IMG_H * 0.55)
    font = load_font(size=font_size)
    
    # Measure total text width to check for scaling
    bbox = draw.textbbox((0, 0), visual_text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Scale down if text is too wide for the AVAILABLE area
    if text_w > avail_w * 0.95:
        scale = (avail_w * 0.95) / text_w
        font_size = max(10, int(font_size * scale))
        font = load_font(size=font_size)
        bbox = draw.textbbox((0, 0), visual_text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

    # Calculate starting point X and Y (centered)
    x = start_x + (avail_w - text_w) // 2
    y = (IMG_H - text_h) // 2 - 5 
    
    # 4. DRAW TEXT ON RGBA CANVAS FOR TRANSPARENCY
    # Initialize the temporary canvas as RGBA (transparent background)
    temp_img = Image.new("RGBA", img.size, (0, 0, 0, 0))
    temp_draw = ImageDraw.Draw(temp_img)
    text_color_alpha = text_color + (255,) # Text is fully opaque
    
    # 5. DRAW TEXT CHARACTER-BY-CHARACTER (Jitter, Emboss, Smudge)
    current_x = x
    
    # Apply Emboss/Outline first (drawn slightly offset beneath the main text)
    if random.random() < 0.35: 
        emboss_color = (255, 255, 255) if luminance(bg_color) < 128 else (0, 0, 0)
        emboss_color_alpha = emboss_color + (255,)
        # Use the whole text measure for the consistent shadow placement
        temp_draw.text((x - 1, y - 1), visual_text, fill=emboss_color_alpha, font=font)

    # Draw the main text loop
    for char in visual_text:
        char_bbox = temp_draw.textbbox((0, 0), char, font=font)
        char_width = char_bbox[2] - char_bbox[0]
        
        # Character Shift/Crookedness (Y-axis jitter)
        char_shift_y = random.randint(-2, 2)
        
        # Draw the main character layer
        temp_draw.text((current_x, y + char_shift_y), char, fill=text_color_alpha, font=font)
        
        # Advance X position, adding a small random kerning jitter
        kerning_jitter = random.randint(-1, 1)
        current_x += char_width + kerning_jitter
    
    # 6. SIMULATE STROKE THICKNESS SMUDGE (Drawn offset over the main text)
    if random.random() < 0.6: 
        offset_x = random.randint(-1, 1)
        offset_y = random.randint(-1, 1)
        if offset_x != 0 or offset_y != 0:
            smudge_color = (max(0, text_color[0] - 50), max(0, text_color[1] - 50), max(0, text_color[2] - 50))
            smudge_color_alpha = smudge_color + (255,)
            temp_draw.text((x + offset_x, y + offset_y), visual_text, fill=smudge_color_alpha, font=font)
            
    
    # 7. COMPOSITE IMAGE
    # Paste the RGBA text layer onto the RGB plate image using its alpha channel as the mask.
    img.paste(temp_img, (0, 0), temp_img) 
    
    # 8. Border
    # Need to re-initialize draw handle after pasting onto RGB image
    draw = ImageDraw.Draw(img) 
    draw.rectangle([0, 0, IMG_W-1, IMG_H-1], outline=(100, 100, 100), width=1)
    return img
    
# ----------------- AUGMENTATIONS -----------------
def find_perspective_coeffs(src, dst):
    matrix = []
    for p1, p2 in zip(dst, src):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])
    A = np.array(matrix, dtype=np.float64)
    B = np.array(src).reshape(8)
    res = np.linalg.lstsq(A, B, rcond=None)[0]
    return tuple(res.tolist())

def apply_shadow(img: Image.Image):
    """Adds a random dark shadow polygon to simulate trees/poles/buildings."""
    if random.random() < 0.4: # 40% chance
        overlay = Image.new('RGBA', img.size, (0,0,0,0))
        draw = ImageDraw.Draw(overlay)
        
        # Random polygon points
        w, h = img.size
        points = [
            (random.randint(0, w), 0),
            (random.randint(0, w), h),
            (random.randint(0, w), h),
            (random.randint(0, w), 0)
        ]
        
        # Alpha is density of shadow (50-150 out of 255)
        alpha = random.randint(50, 120)
        draw.polygon(points, fill=(0, 0, 0, alpha))
        
        # Composite
        img = img.convert('RGBA')
        img = Image.alpha_composite(img, overlay)
        img = img.convert('RGB')
    return img

def apply_perspective(img: Image.Image):
    w, h = img.size
    max_dx = w * random.uniform(0.01, 0.05)
    max_dy = h * random.uniform(0.01, 0.08)
    src = [(0,0),(w,0),(w,h),(0,h)]
    dst = [
        (random.uniform(0, max_dx), random.uniform(0, max_dy)),
        (w - random.uniform(0, max_dx), random.uniform(0, max_dy)),
        (w - random.uniform(0, max_dx), h - random.uniform(0, max_dy)),
        (random.uniform(0, max_dx), h - random.uniform(0, max_dy)),
    ]
    coeffs = find_perspective_coeffs(src, dst)
    return img.transform(img.size, Image.PERSPECTIVE, coeffs, resample=Image.BICUBIC)

def augment_image(img: Image.Image):
    # Shadow -> Perspective -> Rotate -> Blur -> Noise
    img = apply_shadow(img) # New Shadow
    
    if random.random() < 0.9:
        img = apply_perspective(img)
        
    angle = random.uniform(-5, 5)
    img = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=choose_background()[0])
    
    if random.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 1.2)))
        
    if random.random() < 0.5:
        arr = np.array(img).astype(np.int16)
        noise = np.random.normal(0, random.uniform(5, 15), arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        
    if random.random() < 0.7:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.2))
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.3))
        
    return img

# ----------------- MAIN -----------------
def save_dataset(n_samples: int):
    ensure_dir(OUT_DIR)
    ensure_dir(IMG_DIR)
    
    csv_rows = []
    txt_lines = []
    
    print(f"Generators started... {n_samples} samples.")

    for i in range(n_samples):
        visual, label = generate_plate_text()
        bg_color, _ = choose_background()
        
        # Render with New IND logic
        img = render_plate_image(visual, bg_color)
        
        # Augment with Shadows
        img = augment_image(img)
        
        fname = f"{label}_{i}.jpg"
        img.save(IMG_DIR / fname, quality=JPG_QUALITY)
        
        csv_rows.append((str(Path("images") / fname), label))
        txt_lines.append(f"{str(Path('images') / fname)} {label}")
        
        if (i+1) % 10000 == 0: print(f"  -> Generated {i+1}...")

    with open(LABELS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        writer.writerows(csv_rows)
        
    with open(LABELS_TXT, "w") as f:
        f.write("\n".join(txt_lines))
        
    print(f"Dataset Ready at {OUT_DIR}")

if __name__ == "__main__":
    save_dataset(100)
