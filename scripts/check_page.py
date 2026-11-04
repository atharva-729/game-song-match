"""Checkpoints 4-5 in a real browser: screenshots at 380/1280 px in light/dark, the test tone through the
file input (result renders, playhead moves, moments filled), a 4-minute mp3 timed, no console errors.
usage: python scripts/check_page.py [live-url]   (default: serves site/ locally)
"""
import json, subprocess, sys, threading, time
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
SITE, SHOTS = ROOT / "site", ROOT / "logs/screens"
SHOTS.mkdir(parents=True, exist_ok=True)
tone, mp3 = ROOT / "data/test_tone.wav", ROOT / "data/test_tone_4min.mp3"
if not tone.exists():
    subprocess.run([sys.executable, str(ROOT / "scripts/make_test_tone.py"), str(tone)], check=True)
if not mp3.exists():
    long = ROOT / "data/test_tone_4min.wav"
    subprocess.run([sys.executable, str(ROOT / "scripts/make_test_tone.py"), str(long), "240"], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(long), "-b:a", "128k", str(mp3)], check=True)

sys.path.insert(0, str(ROOT / "scripts"))
from serve import make_server  # Range support, so audio seeking behaves as on GitHub Pages
LIVE = sys.argv[1].rstrip("/") + "/" if len(sys.argv) > 1 else None
srv = None if LIVE else make_server()
if srv:
    threading.Thread(target=srv.serve_forever, daemon=True).start()
URL = (LIVE or f"http://127.0.0.1:{srv.server_port}/") + "index.html"
PFX = "live_" if LIVE else ""

report, errors = {"screens": [], "runs": []}, []
with sync_playwright() as p:
    browser = p.chromium.launch(args=["--autoplay-policy=no-user-gesture-required"])
    for width in (380, 1280):
        for scheme in ("light", "dark"):
            ctx = browser.new_context(viewport={"width": width, "height": 900}, color_scheme=scheme)
            page = ctx.new_page()
            page.on("console", lambda m, w=width, s=scheme: m.type == "error" and errors.append(f"{w}/{s}: {m.text}"))
            page.on("pageerror", lambda e, w=width, s=scheme: errors.append(f"{w}/{s}: {e}"))
            page.goto(URL)
            page.wait_for_selector("#result:not(.hidden)", timeout=30000)
            page.wait_for_function("document.querySelectorAll('#moments li').length > 0", timeout=30000)
            scroll_w = page.evaluate("document.documentElement.scrollWidth")
            shot = SHOTS / f"{PFX}page_{width}_{scheme}.png"
            page.screenshot(path=str(shot), full_page=True)
            report["screens"].append({"file": str(shot.relative_to(ROOT)), "width": width, "scheme": scheme,
                                      "horizontal_overflow": scroll_w > width})

            if width == 1280 and scheme == "dark" or width == 380 and scheme == "light":
                # demo cards: each one plays real audio and the scoreboard follows the playhead
                for k in range(page.locator(".demo").count()):
                    page.locator(".demo").nth(k).click()
                    page.wait_for_function("(a => !a.paused && a.currentTime > 0.05)(document.getElementById('audio'))", timeout=30000)
                    page.wait_for_timeout(2500)
                    page.click("#play")
                    report["demos"] = report.get("demos", []) + [{
                        "width": width, "demo": page.locator(".demo").nth(k).get_attribute("data-key"),
                        "title": page.inner_text("#r-title"),
                        "audio_t": round(page.evaluate("document.getElementById('audio').currentTime"), 2),
                        "scoreboard": " | ".join(page.inner_text(s) for s in ("#jt-l", "#jt-clock", "#jt-r")).replace("\n", " ")}]
                page.screenshot(path=str(SHOTS / f"{PFX}demo_{width}_{scheme}.png"), full_page=True)
                # seeking needs HTTP Range on the server (GitHub Pages and serve.py have it, http.server doesn't)
                page.evaluate("(() => { const a = document.getElementById('audio'); a.currentTime = a.duration * 0.6; })()")
                page.wait_for_timeout(1500)
                report["seek_ok"] = report.get("seek_ok", True) and page.evaluate(
                    "(() => { const a = document.getElementById('audio'); return a.currentTime > a.duration * 0.55; })()")
                for f in (tone, mp3):
                    page.set_input_files("#file", str(f))
                    page.wait_for_function("document.getElementById('status').textContent.startsWith('Matched in')", timeout=60000)
                    status = page.inner_text("#status")
                    page.click("#play")
                    t_a = page.evaluate("document.getElementById('audio').currentTime")
                    page.wait_for_timeout(2500)
                    t_b = page.evaluate("document.getElementById('audio').currentTime")
                    page.click("#play")
                    run = {"file": f.name, "width": width, "status": status, "title": page.inner_text("#r-title"),
                           "moments": page.evaluate("document.querySelectorAll('#moments li').length"),
                           "runners": page.evaluate("document.querySelectorAll('.runner').length"),
                           "audio_advanced_s": round(t_b - t_a, 2), "share": page.inner_text("#share-text")}
                    report["runs"].append(run)
                    page.screenshot(path=str(SHOTS / f"{PFX}song_{width}_{scheme}_{f.stem}.png"), full_page=True)
            ctx.close()
    # the previous page design stays reachable at v1.html
    ctx = browser.new_context(viewport={"width": 1280, "height": 900})
    page = ctx.new_page()
    page.on("console", lambda m: m.type == "error" and errors.append(f"v1: {m.text}"))
    page.on("pageerror", lambda e: errors.append(f"v1: {e}"))
    page.goto(URL.replace("index.html", "v1.html"))
    page.wait_for_function("document.querySelectorAll('#moments li').length > 0", timeout=30000)
    report["v1_ok"] = True
    ctx.close()
    browser.close()
if srv:
    srv.shutdown()
report["console_errors"] = errors
(ROOT / f"logs/{PFX}page_check.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
print(json.dumps(report, indent=1))
ok = not errors and all(r["moments"] == 5 and r["audio_advanced_s"] > 1 for r in report["runs"]) \
     and all(d["audio_t"] > 1 and d["scoreboard"].strip(" |") for d in report.get("demos", [])) \
     and not any(s["horizontal_overflow"] for s in report["screens"]) and report.get("seek_ok") \
     and (not LIVE or all(LIVE in r["share"] for r in report["runs"]))
sys.exit(0 if ok else 1)
