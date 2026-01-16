"""Shop2Post App
=================

This script reads an e‑commerce product page or a local image and produces ad‑ready
images featuring the product. The images are resized to fill social media
aspect ratios while preserving the original product photograph. No text
overlays are added – captions can be added separately on the social platforms.

Usage
-----
The script can be executed from the command line with either a product URL or
a local image path:

```
python shop2post_app.py --url "https://example.com/product/page" --output out_dir --full-frame

python shop2post_app.py --image-path /path/to/image.jpg --title "Product Name" --output out_dir --full-frame
```

Arguments
~~~~~~~~~
* `--url`: URL of the product page to scrape. If provided, the script will
  attempt to fetch the HTML, parse images, and pick the largest photo as the
  hero image. Requires internet access on the host running the script.
* `--image-path`: Path to a local image file. Use this when network
  restrictions prevent scraping or when you already have the product image.
* `--title`: Optional title for the product (used only when scraping is
  disabled); included in logs but not on the images. Captions can be added
  manually on social platforms.
* `--output`: Output directory to save the generated images. Defaults to
  `out_no_text`.
* `--full-frame`: When set, images are sized to fill the target aspect ratios
  without borders. By default, images are letterboxed to fit.

The script outputs four PNG files named:
```
ad_instagram_portrait.png (1080×1350)
ad_instagram_story.png   (1080×1920)
ad_instagram_square.png  (1080×1080)
ad_facebook_landscape.png (1200×628)
```

Note: This script uses only local Python libraries (Pillow for image
processing and requests/BeautifulSoup for scraping). It does not rely on any
external AI services.
"""

import argparse
import os
import re
import io
from typing import List, Optional

import requests
from bs4 import BeautifulSoup
from PIL import Image


def fetch_page(url: str) -> str:
    """Fetches the HTML content of the URL. Returns empty string on error."""
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code == 200:
            return resp.text
    except Exception:
        pass
    return ""


def parse_images(html: str, base_url: str) -> List[str]:
    """Parses image URLs from the HTML. Returns a list of full image URLs."""
    soup = BeautifulSoup(html, "html.parser")
    images = []
    # Include og:image if present
    og = soup.find("meta", property="og:image")
    if og and og.get("content"):
        images.append(requests.compat.urljoin(base_url, og["content"]))
    # Find all <img> tags with src or data-src attributes
    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src") or img.get("data-lazy")
        if not src:
            continue
        src = requests.compat.urljoin(base_url, src)
        # Filter out obvious icons or sprites
        if any(part in src.lower() for part in ["sprite", "icon", "logo"]):
            continue
        images.append(src)
    # Deduplicate while preserving order
    seen = set()
    unique_images = []
    for img_url in images:
        if img_url not in seen:
            seen.add(img_url)
            unique_images.append(img_url)
    return unique_images


def download_image(url: str) -> Optional[Image.Image]:
    """Downloads an image and returns a PIL Image or None on error."""
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        image = Image.open(io.BytesIO(resp.content)).convert("RGBA")
        return image
    except Exception:
        return None


def score_image(img: Image.Image) -> float:
    """Scores an image based on area and aspect ratio; prefers large, not overly wide or tall images."""
    w, h = img.size
    area = w * h
    aspect = w / h
    aspect_penalty = abs(1.0 - min(aspect, 1 / aspect))  # 0 when square
    return area * (1.0 - 0.35 * aspect_penalty)


def choose_best_image(image_urls: List[str]) -> Optional[Image.Image]:
    """Downloads and returns the best image among the provided URLs based on scoring."""
    best_img = None
    best_score = 0.0
    for url in image_urls:
        img = download_image(url)
        if img is None:
            continue
        s = score_image(img)
        if s > best_score:
            best_score = s
            best_img = img
    return best_img


def resize_fill(image: Image.Image, target_size: tuple) -> Image.Image:
    """Resizes and crops the image to fill the target size without borders."""
    src = image.convert("RGBA")
    sw, sh = src.size
    tw, th = target_size
    scale = max(tw / sw, th / sh)
    nw, nh = int(sw * scale), int(sh * scale)
    resized = src.resize((nw, nh), Image.LANCZOS)
    left = (nw - tw) // 2
    top = (nh - th) // 2
    return resized.crop((left, top, left + tw, top + th))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Generate ad images from a product page or local image.")
    parser.add_argument("--url", help="Product page URL to scrape", default=None)
    parser.add_argument("--image-path", help="Local image path", default=None)
    parser.add_argument("--title", help="Optional product title for logging", default="Product")
    parser.add_argument("--output", help="Output directory", default="out_full_frame")
    parser.add_argument("--full-frame", action="store_true", help="Fill the entire frame without borders")
    args = parser.parse_args()

    if not args.url and not args.image_path:
        print("Error: either --url or --image-path must be provided.")
        return

    best_img: Optional[Image.Image] = None
    if args.url:
        html = fetch_page(args.url)
        if not html:
            print("Failed to fetch or parse the page; falling back to local image if provided.")
        else:
            imgs = parse_images(html, args.url)
            if imgs:
                best_img = choose_best_image(imgs)
            if not best_img:
                print("Could not download any usable image from the page; falling back to local image if provided.")
    if not best_img and args.image_path:
        try:
            best_img = Image.open(args.image_path).convert("RGBA")
        except Exception as e:
            print(f"Failed to open local image: {e}")

    if not best_img:
        print("No image could be processed. Aborting.")
        return

    # Output sizes and file names
    sizes = {
        "ad_instagram_portrait.png": (1080, 1350),
        "ad_instagram_story.png": (1080, 1920),
        "ad_instagram_square.png": (1080, 1080),
        "ad_facebook_landscape.png": (1200, 628),
    }
    ensure_dir(args.output)
    for name, size in sizes.items():
        full_frame_img = resize_fill(best_img, size)
        out_path = os.path.join(args.output, name)
        full_frame_img.save(out_path)
        print(f"Generated: {out_path} ({size[0]}x{size[1]})")


if __name__ == "__main__":
    main()