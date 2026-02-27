"""Shop2Post App
=================

Generate ad-ready social media images from a product page or local image.
The tool preserves the product appearance, replaces or extends the background,
and exports platform-ready dimensions without text overlays.
"""

import argparse
import io
import json
import os
import re
import uuid
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageFilter

try:
    from flask import Flask, flash, redirect, render_template, request, send_from_directory, url_for
    from werkzeug.utils import secure_filename
except ImportError:  # Flask is optional for CLI-only usage
    Flask = None
    flash = None
    redirect = None
    render_template = None
    request = None
    send_from_directory = None
    url_for = None
    secure_filename = None


USER_AGENT = "Mozilla/5.0 (Shop2Post)"
TIMEOUT = 15
MIN_IMAGE_SIDE = 300
OUTPUT_SIZES = {
    "ad_instagram_portrait.png": (1080, 1350),
    "ad_instagram_story.png": (1080, 1920),
    "ad_instagram_square.png": (1080, 1080),
    "ad_facebook_landscape.png": (1200, 628),
}
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}


@dataclass
class ProductData:
    title: Optional[str]
    price: Optional[str]
    image_urls: List[str]


@dataclass
class GenerationResult:
    output_dir: str
    metadata_path: str
    metadata: Dict[str, object]


def fetch_page(url: str) -> str:
    """Fetches the HTML content of the URL. Returns empty string on error."""
    try:
        resp = requests.get(
            url,
            timeout=TIMEOUT,
            headers={"User-Agent": USER_AGENT},
            allow_redirects=True,
        )
        resp.raise_for_status()
        return resp.text
    except requests.RequestException:
        return ""


def _safe_json_loads(payload: str) -> Iterable[dict]:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return []
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def parse_product_data(html: str, base_url: str) -> ProductData:
    soup = BeautifulSoup(html, "html.parser")
    title = None
    price = None
    image_urls: List[str] = []

    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        title = og_title["content"].strip()

    if not title:
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)

    meta_price = soup.find("meta", property="product:price:amount")
    if meta_price and meta_price.get("content"):
        price = meta_price["content"].strip()

    if not price:
        price_meta = soup.find("meta", attrs={"name": "price"})
        if price_meta and price_meta.get("content"):
            price = price_meta["content"].strip()

    for tag in soup.find_all("meta", property=re.compile(r"^og:image")):
        if tag.get("content"):
            image_urls.append(requests.compat.urljoin(base_url, tag["content"]))

    twitter_image = soup.find("meta", property="twitter:image")
    if twitter_image and twitter_image.get("content"):
        image_urls.append(requests.compat.urljoin(base_url, twitter_image["content"]))

    image_src = soup.find("link", rel="image_src")
    if image_src and image_src.get("href"):
        image_urls.append(requests.compat.urljoin(base_url, image_src["href"]))

    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src") or img.get("data-lazy")
        if not src:
            continue
        src = requests.compat.urljoin(base_url, src)
        if any(part in src.lower() for part in ["sprite", "icon", "logo", "thumbnail"]):
            continue
        image_urls.append(src)

    for script in soup.find_all("script", type="application/ld+json"):
        for data in _safe_json_loads(script.string or ""):
            if data.get("@type") == "Product":
                title = title or data.get("name")
                image_value = data.get("image") or data.get("imageUrl")
                if isinstance(image_value, list):
                    image_urls.extend(image_value)
                elif isinstance(image_value, str):
                    image_urls.append(image_value)
                offers = data.get("offers")
                if isinstance(offers, dict):
                    price = price or offers.get("price")
            if "@graph" in data:
                for item in data.get("@graph", []):
                    if isinstance(item, dict) and item.get("@type") == "Product":
                        title = title or item.get("name")
                        offers = item.get("offers")
                        if isinstance(offers, dict):
                            price = price or offers.get("price")

    image_urls = [requests.compat.urljoin(base_url, url) for url in image_urls]

    seen = set()
    deduped = []
    for url in image_urls:
        if url not in seen:
            seen.add(url)
            deduped.append(url)

    return ProductData(title=title, price=price, image_urls=deduped)


def download_image(url: str) -> Optional[Image.Image]:
    """Downloads an image and returns a PIL Image or None on error."""
    try:
        resp = requests.get(url, timeout=TIMEOUT, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        image = Image.open(io.BytesIO(resp.content)).convert("RGBA")
        return image
    except (requests.RequestException, OSError):
        return None


def score_image(img: Image.Image) -> float:
    """Scores an image based on area and aspect ratio; prefers large, balanced images."""
    w, h = img.size
    if min(w, h) < MIN_IMAGE_SIDE:
        return 0.0
    area = w * h
    aspect = w / h
    aspect_penalty = abs(1.0 - min(aspect, 1 / aspect))
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


def _corner_samples(image: Image.Image, pct: float = 0.06) -> List[Tuple[int, int, int]]:
    w, h = image.size
    dx = max(1, int(w * pct))
    dy = max(1, int(h * pct))
    rgb = image.convert("RGB")
    samples = []
    for x0 in [0, w - dx]:
        for y0 in [0, h - dy]:
            crop = rgb.crop((x0, y0, x0 + dx, y0 + dy))
            samples.extend(list(crop.getdata()))
    return samples


def _median_color(samples: List[Tuple[int, int, int]]) -> Tuple[int, int, int]:
    if not samples:
        return (255, 255, 255)
    channels = list(zip(*samples))
    medians = []
    for channel in channels:
        sorted_vals = sorted(channel)
        mid = len(sorted_vals) // 2
        medians.append(sorted_vals[mid])
    return tuple(medians)


def create_product_mask(image: Image.Image) -> Optional[Image.Image]:
    """Creates a mask that separates the product from the background."""
    if image.mode == "RGBA":
        alpha = image.getchannel("A")
        if alpha.getextrema() != (255, 255):
            return alpha

    samples = _corner_samples(image)
    bg_color = _median_color(samples)
    rgb = image.convert("RGB")
    pixels = list(rgb.getdata())
    mask_data = []
    for r, g, b in pixels:
        dist = ((r - bg_color[0]) ** 2 + (g - bg_color[1]) ** 2 + (b - bg_color[2]) ** 2) ** 0.5
        mask_data.append(255 if dist > 25 else 0)
    mask = Image.new("L", image.size)
    mask.putdata(mask_data)
    mask = mask.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(2))

    coverage = sum(1 for value in mask_data if value > 0) / len(mask_data)
    if coverage < 0.05 or coverage > 0.95:
        return None
    return mask


def resize_fill(image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    src = image.convert("RGBA")
    sw, sh = src.size
    tw, th = target_size
    scale = max(tw / sw, th / sh)
    nw, nh = int(sw * scale), int(sh * scale)
    resized = src.resize((nw, nh), Image.LANCZOS)
    left = (nw - tw) // 2
    top = (nh - th) // 2
    return resized.crop((left, top, left + tw, top + th))


def resize_fill_with_mask(image: Image.Image, mask: Image.Image, target_size: Tuple[int, int]) -> Tuple[Image.Image, Image.Image]:
    src = image.convert("RGBA")
    sw, sh = src.size
    tw, th = target_size
    scale = max(tw / sw, th / sh)
    nw, nh = int(sw * scale), int(sh * scale)
    resized_img = src.resize((nw, nh), Image.LANCZOS)
    resized_mask = mask.resize((nw, nh), Image.LANCZOS)
    left = (nw - tw) // 2
    top = (nh - th) // 2
    crop_box = (left, top, left + tw, top + th)
    return resized_img.crop(crop_box), resized_mask.crop(crop_box)


def generate_gradient(size: Tuple[int, int], start: Tuple[int, int, int], end: Tuple[int, int, int]) -> Image.Image:
    w, h = size
    base = Image.new("RGB", (w, h), start)
    top = Image.new("RGB", (w, h), end)
    mask = Image.new("L", (w, h))
    draw = ImageDraw.Draw(mask)
    for y in range(h):
        value = int(255 * (y / h))
        draw.line([(0, y), (w, y)], fill=value)
    return Image.composite(top, base, mask).convert("RGBA")


def generate_background(size: Tuple[int, int], style: str) -> Image.Image:
    if style == "studio":
        bg = generate_gradient(size, (245, 245, 245), (220, 220, 220))
    elif style == "cafe":
        bg = generate_gradient(size, (255, 235, 220), (200, 160, 140))
    elif style == "tech":
        bg = generate_gradient(size, (10, 40, 80), (30, 90, 140))
    else:
        bg = generate_gradient(size, (240, 240, 255), (200, 210, 255))

    draw = ImageDraw.Draw(bg)
    if style in {"cafe", "studio"}:
        for i in range(12):
            radius = size[0] // 8 + i * 12
            alpha = 12 if style == "studio" else 20
            ellipse_color = (255, 255, 255, alpha)
            draw.ellipse(
                [
                    size[0] * 0.2 - radius,
                    size[1] * 0.1 - radius,
                    size[0] * 0.2 + radius,
                    size[1] * 0.1 + radius,
                ],
                outline=ellipse_color,
                width=2,
            )
    if style == "tech":
        grid_color = (255, 255, 255, 25)
        step = max(40, size[0] // 15)
        for x in range(0, size[0], step):
            draw.line([(x, 0), (x, size[1])], fill=grid_color, width=1)
        for y in range(0, size[1], step):
            draw.line([(0, y), (size[0], y)], fill=grid_color, width=1)

    return bg


def compose_ad_image(image: Image.Image, mask: Optional[Image.Image], size: Tuple[int, int], style: str) -> Image.Image:
    background = generate_background(size, style)
    if mask is None:
        return resize_fill(image, size)
    product, resized_mask = resize_fill_with_mask(image, mask, size)
    return Image.composite(product, background, resized_mask)


def build_caption(title: str, price: Optional[str]) -> str:
    if price:
        return f"Upgrade your routine with {title} for {price}. Ready to make it yours?"
    return f"Upgrade your routine with {title}. Ready to make it yours?"


def build_hashtags(title: str) -> Dict[str, List[str]]:
    title_lower = title.lower()
    tech_tags = ["#innovation", "#technology", "#ai"]
    if any(word in title_lower for word in ["robot", "automation", "ai", "smart"]):
        instagram = ["#robotics", "#technology", "#innovation", "#future", "#smarttech"]
        tiktok = ["#robotics", "#tech", "#innovation", "#futuristic", "#gadgets"]
    else:
        instagram = ["#newarrival", "#productdesign", "#onlineshopping", "#musthave", "#innovation"]
        tiktok = ["#producttok", "#shopnow", "#trending", "#musthave", "#innovation"]

    return {
        "instagram": instagram[:5],
        "tiktok": tiktok[:5],
        "linkedin": tech_tags[:3],
        "facebook": ["#newproduct", "#shopping", "#innovation"],
        "x": ["#newlaunch", "#innovation", "#tech"],
    }


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def is_safe_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def generate_ad_assets(
    source_url: Optional[str],
    image_path: Optional[str],
    title_override: Optional[str],
    price_override: Optional[str],
    output_dir: str,
    background_style: str,
) -> GenerationResult:
    if not source_url and not image_path:
        raise ValueError("Provide a product URL or an image upload.")

    product_data = ProductData(title=None, price=None, image_urls=[])
    if source_url:
        if not is_safe_url(source_url):
            raise ValueError("URL must begin with http:// or https:// and include a valid host.")
        html = fetch_page(source_url)
        if html:
            product_data = parse_product_data(html, source_url)
        elif not image_path:
            raise ValueError("Product page is inaccessible and no fallback image was provided.")

    title = title_override or product_data.title or "Product"
    price = price_override or product_data.price

    best_img: Optional[Image.Image] = None
    if product_data.image_urls:
        best_img = choose_best_image(product_data.image_urls)

    if not best_img and image_path:
        try:
            best_img = Image.open(image_path).convert("RGBA")
        except OSError as exc:
            raise ValueError(f"Unable to open local image: {exc}") from exc

    if not best_img:
        raise ValueError("No suitable image found from URL or uploaded file.")

    mask = create_product_mask(best_img)

    ensure_dir(output_dir)
    for name, size in OUTPUT_SIZES.items():
        composed = compose_ad_image(best_img, mask, size, background_style)
        composed.save(os.path.join(output_dir, name))

    metadata = {
        "title": title,
        "price": price,
        "caption": build_caption(title, price),
        "hashtags": build_hashtags(title),
    }
    metadata_path = os.path.join(output_dir, "ad_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as file_obj:
        json.dump(metadata, file_obj, indent=2)

    return GenerationResult(output_dir=output_dir, metadata_path=metadata_path, metadata=metadata)


def create_web_app() -> "Flask":
    if Flask is None:
        raise RuntimeError("Flask is not installed. Install flask to use web mode.")

    app = Flask(__name__, template_folder="templates")
    app.config["SECRET_KEY"] = "shop2post-dev-key"
    app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
    app.config["UPLOAD_FOLDER"] = os.path.join("web_runs", "uploads")
    app.config["OUTPUT_FOLDER"] = os.path.join("web_runs", "outputs")
    ensure_dir(app.config["UPLOAD_FOLDER"])
    ensure_dir(app.config["OUTPUT_FOLDER"])

    @app.route("/", methods=["GET", "POST"])
    def index():
        if request.method == "POST":
            product_url = (request.form.get("url") or "").strip()
            title_override = (request.form.get("title") or "").strip() or None
            price_override = (request.form.get("price") or "").strip() or None
            background_style = (request.form.get("background_style") or "studio").strip()
            image_file = request.files.get("image")

            upload_path = None
            if image_file and image_file.filename:
                safe_name = secure_filename(image_file.filename)
                if not safe_name or not allowed_file(safe_name):
                    flash("Unsupported file type. Use png, jpg, jpeg, or webp.")
                    return redirect(url_for("index"))
                run_id = uuid.uuid4().hex
                upload_dir = os.path.join(app.config["UPLOAD_FOLDER"], run_id)
                ensure_dir(upload_dir)
                upload_path = os.path.join(upload_dir, safe_name)
                image_file.save(upload_path)

            run_id = uuid.uuid4().hex
            output_dir = os.path.join(app.config["OUTPUT_FOLDER"], run_id)

            try:
                result = generate_ad_assets(
                    source_url=product_url or None,
                    image_path=upload_path,
                    title_override=title_override,
                    price_override=price_override,
                    output_dir=output_dir,
                    background_style=background_style,
                )
            except ValueError as exc:
                flash(str(exc))
                return redirect(url_for("index"))

            image_files = sorted([name for name in os.listdir(result.output_dir) if name.endswith(".png")])
            return render_template(
                "index.html",
                result={
                    "run_id": run_id,
                    "images": image_files,
                    "metadata": result.metadata,
                },
            )

        return render_template("index.html", result=None)

    @app.route("/outputs/<run_id>/<path:filename>")
    def download_output(run_id: str, filename: str):
        safe_run = secure_filename(run_id)
        safe_name = secure_filename(filename)
        out_dir = os.path.join(app.config["OUTPUT_FOLDER"], safe_run)
        return send_from_directory(out_dir, safe_name, as_attachment=False)

    return app


def run_cli(args: argparse.Namespace) -> int:
    try:
        result = generate_ad_assets(
            source_url=args.url,
            image_path=args.image_path,
            title_override=args.title,
            price_override=args.price,
            output_dir=args.output,
            background_style=args.background_style,
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    for name, size in OUTPUT_SIZES.items():
        print(f"Generated: {os.path.join(result.output_dir, name)} ({size[0]}x{size[1]})")
    print(f"Metadata saved: {result.metadata_path}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ad images from a product page or local image.")
    parser.add_argument("--url", help="Product page URL to scrape", default=None)
    parser.add_argument("--image-path", help="Local image path", default=None)
    parser.add_argument("--title", help="Override product title for captions", default=None)
    parser.add_argument("--price", help="Override price for captions", default=None)
    parser.add_argument("--output", help="Output directory", default="out_ads")
    parser.add_argument(
        "--background-style",
        help="Background style (studio, gradient, cafe, tech)",
        default="studio",
        choices=["studio", "gradient", "cafe", "tech"],
    )
    parser.add_argument("--web", action="store_true", help="Run the HTML wrapper web app")
    parser.add_argument("--host", default="0.0.0.0", help="Host for web mode")
    parser.add_argument("--port", type=int, default=8000, help="Port for web mode")
    args = parser.parse_args()

    if args.web:
        app = create_web_app()
        app.run(host=args.host, port=args.port, debug=False)
        return

    raise SystemExit(run_cli(args))


if __name__ == "__main__":
    main()
