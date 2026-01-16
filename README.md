# Shop2Post

Shop2Post is a lightweight CLI that turns any e-commerce product page (or local product photo) into polished social-media ad visuals. The utility preserves the product appearance, replaces the background with clean, on-brand environments, and exports platform-ready PNGs with no text overlays.

## Features
- Scrape product title, price, and hero imagery from Shopify, WooCommerce, Amazon, and other product pages.
- Automatic hero image selection based on size and aspect ratio scoring.
- Simple product isolation to preserve the product pixels while replacing the background.
- Ready-to-post sizes for Instagram, TikTok, Facebook, and LinkedIn.
- JSON metadata output with caption + platform-specific hashtag suggestions.

## Requirements
- Python 3.9+
- `requests`
- `beautifulsoup4`
- `pillow`

Install dependencies:

```bash
pip install requests beautifulsoup4 pillow
```

## Usage

### From a product URL

```bash
python shop2post_app.py \
  --url "https://example.com/product/page" \
  --output out_ads \
  --background-style studio
```

### From a local image

```bash
python shop2post_app.py \
  --image-path /path/to/image.jpg \
  --title "Product Name" \
  --price "$199" \
  --output out_ads \
  --background-style gradient
```

### Output

The CLI creates PNGs in the output directory:

- `ad_instagram_portrait.png` (1080×1350)
- `ad_instagram_story.png` (1080×1920)
- `ad_instagram_square.png` (1080×1080)
- `ad_facebook_landscape.png` (1200×628)

It also saves `ad_metadata.json` containing:

- `title`
- `price`
- `caption`
- `hashtags` (platform-specific recommendations)

## Notes
- If the product page is inaccessible or yields no usable images, pass `--image-path` to supply a local photo.
- The background replacement uses a fast, heuristic mask to keep the product intact. For complex backgrounds, provide a transparent PNG for the best result.
