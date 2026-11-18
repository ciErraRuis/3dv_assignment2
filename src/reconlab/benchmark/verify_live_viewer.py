"""Browser acceptance of full coverage, real media decode, and same-page comparison."""

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener

os.environ.setdefault(
    "PLAYWRIGHT_BROWSERS_PATH", "/mnt/afs/260010098/data/envs/report-browser-binaries"
)
os.environ.setdefault(
    "FONTCONFIG_FILE", "/mnt/afs/260010098/data/envs/report-browser-libs/fonts.conf"
)
sys.path.insert(0, "/mnt/afs/260010098/data/envs/report-browser")
from playwright.sync_api import sync_playwright  # noqa: E402

URL = "http://127.0.0.1:8766"
OUTPUT = Path("/mnt/afs/260010098/data/3dv_assignment/matching10_sparse6_r01/viewer")


def main():
    opener = build_opener(ProxyHandler({}))
    with opener.open(URL) as response:
        html = response.read()
    assert len(html) < 1024 * 1024 and b"data:image" not in html
    with opener.open(URL + "/api/index") as response:
        data = json.load(response)
    assert len(data["objects"]) == 10 and len(data["methods"]) == 14 and len(data["rows"]) == 140
    assert len(data["gaussians"]) == 4
    errors, requests, checks = [], [], []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"]
        )
        page = browser.new_page(viewport={"width": 1500, "height": 1100})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("request", lambda request: requests.append(request.url))
        page.goto(URL, wait_until="networkidle")
        page.wait_for_function("document.querySelectorAll('canvas[data-decoded=true]').length===2")
        assert page.locator("#object option").count() == 10
        assert page.locator("#left .method-select option").count() == 14
        assert page.locator("#matrix button").count() == 14
        initial_media = {u for u in requests if "/media/" in u}
        assert len(initial_media) <= 10 and not any(".gif" in u for u in initial_media)
        for obj in data["objects"]:
            page.select_option("#object", obj["uid"])
            page.wait_for_function(
                "document.querySelectorAll('canvas[data-decoded=true]').length===2"
            )
            assert page.locator("#matrix button").count() == 14
            assert page.locator("#pair option").count() == 15
            checks.append({"kind": "object", "uid": obj["uid"], "decoded_canvases": 2})
        page.select_option("#object", data["objects"][0]["uid"])
        page.select_option("#pair", "view_001__view_017")
        for method in data["methods"]:
            page.click(f"#matrix [data-method='{method['id']}']")
            page.wait_for_function(
                "name=>document.querySelector('#left h3')?.textContent.includes(name)",
                arg=method["label"],
            )
            page.wait_for_function(
                "document.querySelector('#left canvas[data-decoded=true]')||document.querySelector('#left .empty')"
            )
            assert page.url.rstrip("/") == URL
            checks.append(
                {
                    "kind": "method",
                    "method": method["id"],
                    "canvas": bool(page.locator("#left canvas[data-decoded=true]").count()),
                }
            )
        page.select_option("#left .method-select", "gluemap")
        page.select_option("#right .method-select", "romav2")
        page.select_option("#mode", "render")
        page.wait_for_function(
            "document.querySelectorAll('img.gif').length===2&&[...document.querySelectorAll('img.gif')].every(i=>i.complete&&i.naturalWidth===512)"
        )
        media_url = page.locator("img.gif").first.get_attribute("src")
        with opener.open(Request(URL + media_url, headers={"Range": "bytes=0-127"})) as response:
            assert response.status == 206 and len(response.read()) == 128
        gif = page.locator("img.gif").first
        gif.scroll_into_view_if_needed()
        first = gif.screenshot()
        page.wait_for_timeout(500)
        second = gif.screenshot()
        assert hashlib.sha256(first).digest() != hashlib.sha256(second).digest(), (
            "GIF did not animate"
        )
        page.select_option("#mode", "geometry")
        page.wait_for_function(
            "document.querySelectorAll('iframe').length===2&&[...document.querySelectorAll('iframe')].every(f=>f.contentDocument?.querySelector('canvas')?.width>0)"
        )
        for uid in [data["objects"][0]["uid"], data["objects"][-1]["uid"]]:
            page.select_option("#object", uid)
            page.select_option("#branch", "gt")
            page.wait_for_function(
                "document.querySelectorAll('img.gif').length===2&&[...document.querySelectorAll('img.gif')].every(i=>i.complete&&i.naturalWidth===512)"
            )
            assert page.locator("#matrix button").count() == 4
            assert page.locator("#object option").count() == 10
            checks.append({"kind": "gt_gifs", "uid": uid, "decoded": 2})
        page.select_option("#branch", "frontend")
        page.select_option("#object", data["objects"][0]["uid"])
        page.select_option("#left .method-select", "sift")
        page.select_option("#right .method-select", "romav2")
        page.select_option("#pair", "view_001__view_017")
        page.wait_for_function("document.querySelectorAll('canvas[data-decoded=true]').length===2")
        page.select_option('#right select[aria-label="实际模型输入阶段"]', "coarse_800")
        page.wait_for_function(
            "document.querySelector('#right canvas')?.dataset.stage==='coarse_800'&&document.querySelector('#right .input-note').textContent.includes('coarse_800')"
        )
        assert page.locator("#right canvas").evaluate("(c)=>[c.width,c.height]") == [1680, 800]
        checks.append(
            {
                "kind": "actual_model_input_stage",
                "stage": "coarse_800",
                "caption_matches_pixels": True,
            }
        )
        page.screenshot(path=str(OUTPUT / "browser_preview.png"), full_page=True)
        assert not errors, errors
        browser.close()
    result = {
        "passed": True,
        "checked_at": time.time(),
        "url": URL,
        "html_bytes": len(html),
        "objects": 10,
        "methods": 14,
        "frontend_slots": 140,
        "gt_methods": 4,
        "initial_media_requests": len(initial_media),
        "initial_gif_requests": 0,
        "byte_range_verified": True,
        "gif_animation_verified": True,
        "same_document_navigation": True,
        "geometry_iframes_decoded": 2,
        "user_port_forward_verified": False,
        "checks": checks,
        "errors": errors,
    }
    (OUTPUT / "BROWSER_CHECK.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(
        json.dumps({k: v for k, v in result.items() if k != "checks"}, ensure_ascii=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
