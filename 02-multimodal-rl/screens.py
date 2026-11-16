"""Render the agent's observations: small grayscale 'screenshots' of a wallet balance or a limit panel."""
import random, numpy as np, torch
from PIL import Image, ImageDraw, ImageFont, ImageFilter

H, W = 32, 64
FONTS = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"]

def render(value, kind, rng, noise=0.08, blur=0.0):
    """kind 'BAL' or 'LIM'. Random font, size, offset, contrast, sensor noise, optional blur."""
    bg = rng.randint(200, 255); fg = rng.randint(0, 70)
    img = Image.new("L", (W, H), bg); d = ImageDraw.Draw(img)
    small = ImageFont.truetype(rng.choice(FONTS), 9)
    d.text((2 + rng.randint(0, 3), 1), kind, fill=fg + 60, font=small)
    big = ImageFont.truetype(rng.choice(FONTS), rng.randint(15, 19))
    d.text((14 + rng.randint(0, 14), 10 + rng.randint(0, 3)), str(value), fill=fg, font=big)
    if blur: img = img.filter(ImageFilter.GaussianBlur(blur))
    a = np.asarray(img, dtype=np.float32) / 255.0
    a = a + np.random.default_rng(rng.randint(0, 2**31)).normal(0, noise, a.shape)
    return np.clip(a, 0, 1).astype(np.float32)

def bank(n_per_value, seed, noise=0.08, blur=0.0, hi=20):
    """Pre-rendered images: bank[kind][value] -> tensor [n, 1, H, W]."""
    rng = random.Random(seed); out = {}
    for kind in ("BAL", "LIM"):
        out[kind] = {v: torch.tensor(np.stack([render(v, kind, rng, noise, blur) for _ in range(n_per_value)]))[:, None]
                     for v in range(1, hi + 1)}
    return out

def save_grid(path, seed=7, noise=0.08, blur=0.0):
    rng = random.Random(seed); tiles = []
    for kind, vals in (("BAL", [3, 9, 14, 20]), ("LIM", [5, 11, 17, 8])):
        tiles.append(np.concatenate([np.pad(render(v, kind, rng, noise, blur), 2, constant_values=1) for v in vals], 1))
    g = np.concatenate(tiles, 0)
    Image.fromarray((g * 255).astype(np.uint8)).resize((g.shape[1] * 5, g.shape[0] * 5), Image.NEAREST).save(path)

if __name__ == "__main__":
    save_grid("obs_clean.png"); save_grid("obs_noisy.png", noise=0.22, blur=0.9)
