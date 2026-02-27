# Shop2Post

Shop2Post generates polished social-media ad visuals from e-commerce pages or local product images.
It preserves product appearance, replaces/extends backgrounds, exports platform-ready PNGs, and writes editable caption metadata.

## Features
- Scrapes product title, price, and candidate images from HTML meta tags and JSON-LD (`og:title`, `product:price:amount`, structured `Product`).
- Selects the best hero image using resolution + aspect scoring.
- Attempts product isolation with alpha passthrough or corner-color background masking.
- Generates ad assets (no text overlays) in these sizes:
  - Instagram portrait: `1080x1350`
  - Instagram story / TikTok: `1080x1920`
  - Square (Instagram / X / LinkedIn): `1080x1080`
  - Facebook landscape: `1200x628`
- Produces `ad_metadata.json` with title, price, caption, and platform hashtag suggestions.
- Includes an optional browser wrapper (`--web`) for form-based use.

## Requirements
- Python 3.9+
- `requests`
- `beautifulsoup4`
- `pillow`
- `flask` (optional, for browser wrapper)

Install dependencies:

```bash
pip install requests beautifulsoup4 pillow flask
```

## CLI Usage

### From a product URL

```bash
python shop2post_app.py \
  --url "https://example.com/product/page" \
  --output out_ads \
  --background-style studio
```

### From a local image with overrides

```bash
python shop2post_app.py \
  --image-path /path/to/image.jpg \
  --title "Product Name" \
  --price "$199" \
  --output out_ads \
  --background-style gradient
```

## Browser Usage (HTML Wrapper)

```bash
python shop2post_app.py --web --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000`.

The web wrapper supports:
- Product URL input
- Optional image upload fallback
- Title/price overrides
- Background style selection
- Inline previews and metadata rendering

## Outputs

In the chosen output directory:

- `ad_instagram_portrait.png`
- `ad_instagram_story.png`
- `ad_instagram_square.png`
- `ad_facebook_landscape.png`
- `ad_metadata.json`

Example metadata format:

```json
{
  "title": "Example Product",
  "price": "$99",
  "caption": "Upgrade your routine with Example Product for $99. Ready to make it yours?",
  "hashtags": {
    "instagram": ["#newarrival", "#productdesign", "#onlineshopping", "#musthave", "#innovation"],
    "tiktok": ["#producttok", "#shopnow", "#trending", "#musthave", "#innovation"],
    "linkedin": ["#innovation", "#technology", "#ai"],
    "facebook": ["#newproduct", "#shopping", "#innovation"],
    "x": ["#newlaunch", "#innovation", "#tech"]
  }
}
```

## Error handling and safety
- URL validation accepts only `http(s)` URLs.
- Scraper handles redirects/timeouts and returns actionable errors.
- Uploads are extension-filtered (`png/jpg/jpeg/webp`) and filename-sanitized.
- If scraping fails, use `--image-path` (CLI) or upload an image (web mode).
