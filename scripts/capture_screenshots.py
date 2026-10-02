"""Capture screenshots of the running MECH UI via Playwright.

Assumes:
  * backend  http://127.0.0.1:8000  (python backend/main.py, PYTHONPATH=repo root)
  * frontend http://localhost:5173   (npx vite)

Writes PNGs into docs/images/ui/.
"""

import json
import os
import sys

OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "images", "ui"
)
BASE = "http://localhost:5173"

# (hash route, output name, extra wait seconds)
PAGES = [
    ("explorer", "01-model-explorer", 6),
    ("transformer", "02-transformer-visualizer", 4),
    ("network", "03-network", 4),
    ("steering", "04-steering-lab", 4),
    ("benchmark", "05-benchmark-dashboard", 4),
    ("society", "06-research-society", 5),
    ("workspace", "07-campaign-workspace", 4),
    ("models", "08-models", 3),
    ("settings", "09-settings", 2),
    ("plugins", "10-plugin-sdk", 2),
]

console_errors = []


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    from playwright.sync_api import sync_playwright

    captured = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": 1600, "height": 1000})
        page = ctx.new_page()
        page.on(
            "console",
            lambda m: console_errors.append(f"{m.type}: {m.text}")
            if m.type == "error"
            else None,
        )

        for route, name, wait in PAGES:
            url = f"{BASE}/#{route}"
            print(f"-> {route}")
            try:
                page.goto(url, wait_until="networkidle", timeout=45000)
            except Exception:
                page.goto(url, wait_until="domcontentloaded", timeout=45000)
            page.wait_for_timeout(wait * 1000)
            path = os.path.join(OUT, f"{name}.png")
            page.screenshot(path=path, full_page=False)
            captured.append(path)
            title = page.title()
            print(f"   saved {name}.png  (title={title!r})")

        browser.close()

    print(f"\n{len(captured)} screenshots -> {os.path.relpath(OUT)}")
    if console_errors:
        print(f"\nconsole errors ({len(console_errors)}):")
        for err in dict.fromkeys(console_errors):
            print("  ", err[:200])
    else:
        print("\nno console errors")

    with open(os.path.join(OUT, "_console_errors.json"), "w", encoding="utf-8") as fh:
        json.dump(sorted(set(console_errors)), fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
