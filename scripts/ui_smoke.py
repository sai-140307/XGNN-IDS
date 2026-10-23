"""End-to-end check of the deployed console: start Streamlit, drive the UI in a
headless browser, save screenshots to docs/screenshots/.

    python scripts/ui_smoke.py          # needs: pip install playwright && playwright install chromium
"""
from __future__ import annotations

import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "screenshots"
PORT = 8599


def wait_http(url: str, timeout: float = 120) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(url, timeout=3)
            return
        except Exception:
            time.sleep(1)
    raise TimeoutError(url)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    log: list[str] = []
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py", "--server.port", str(PORT),
                             "--server.headless", "true", "--browser.gatherUsageStats", "false"], cwd=ROOT)
    ok = True
    try:
        wait_http(f"http://127.0.0.1:{PORT}/_stcore/health")
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=2)
            page.on("console", lambda m: log.append(f"console.{m.type}: {m.text}"))
            page.on("pageerror", lambda e: log.append(f"PAGEERROR: {e}"))
            page.goto(f"http://127.0.0.1:{PORT}/", wait_until="domcontentloaded")
            frame = page.frame_locator("iframe").first
            frame.locator("#headline h1").wait_for(timeout=180_000)
            page.wait_for_timeout(4000)
            page.screenshot(path=OUT / "console-map.png")
            log.append("map: " + frame.locator("#headline h1").inner_text())

            frame.locator(".inc").first.click()
            page.wait_for_timeout(3000)
            page.screenshot(path=OUT / "console-incident.png")

            frame.locator(".samples button").first.click()
            frame.locator(".verdict .cls").first.wait_for(timeout=120_000)
            frame.locator("#attrChart svg").wait_for(timeout=120_000)
            page.wait_for_timeout(3000)
            page.screenshot(path=OUT / "console-explanation.png")
            log.append("explanation: " + frame.locator(".verdict .cls").first.inner_text())

            frame.locator('.views button[data-view="eval"]').click()
            page.wait_for_timeout(2500)
            page.screenshot(path=OUT / "console-evaluation.png")
            frame.locator('.views button[data-view="method"]').click()
            page.wait_for_timeout(1500)
            page.screenshot(path=OUT / "console-method.png")
            browser.close()
    except Exception as exc:  # keep whatever screenshots exist, report the failure
        ok = False
        log.append(f"FAILED: {exc!r}")
    finally:
        proc.terminate()
    errors = [line for line in log if line.startswith(("PAGEERROR", "FAILED"))]
    (OUT / "ui_smoke_log.txt").write_text("\n".join(log) + "\n")
    print("\n".join(log))
    return 0 if ok and not errors else 1


if __name__ == "__main__":
    sys.exit(main())
