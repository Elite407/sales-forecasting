"""
Automated Dashboard Screenshot Capture Pipeline
Uses Headless Chrome and Chrome DevTools Protocol (CDP) to capture
1920x1080 high-resolution screenshots for all 7 Streamlit dashboard tabs.
"""

import os
import sys
import time
import json
import base64
import urllib.request
import subprocess
import asyncio
import tempfile
import websockets

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ASSETS_DIR = os.path.join(REPO_ROOT, "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

STREAMLIT_PORT = 8501
CDP_PORT = 9222
HEALTH_URL = f"http://127.0.0.1:{STREAMLIT_PORT}/_stcore/health"
APP_URL = f"http://localhost:{STREAMLIT_PORT}"
CDP_JSON_URL = f"http://127.0.0.1:{CDP_PORT}/json"

TAB_CONFIG = [
    {"index": 0, "name": "Overview", "slug": "dashboard_overview.png", "wait": 4},
    {"index": 1, "name": "Forecast Explorer", "slug": "dashboard_forecast_explorer.png", "wait": 5},
    {"index": 2, "name": "Category Trends", "slug": "dashboard_category_trends.png", "wait": 5},
    {"index": 3, "name": "Classical Models", "slug": "dashboard_classical_models.png", "wait": 5},
    {"index": 4, "name": "SHAP Explainer", "slug": "dashboard_shap_explainer.png", "wait": 7},
    {"index": 5, "name": "Model Comparison", "slug": "dashboard_model_comparison.png", "wait": 4},
    {"index": 6, "name": "GenAI Analyst", "slug": "dashboard_genai_analyst.png", "wait": 4},
]

def find_chrome():
    candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("Neither Google Chrome nor Microsoft Edge could be found.")

def kill_ports(ports):
    for port in ports:
        try:
            cmd = ["powershell", "-Command", f"Get-NetTCPConnection -LocalPort {port} -ErrorAction SilentlyContinue | ForEach-Object {{ Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }}"]
            subprocess.run(cmd, capture_output=True, timeout=5)
        except Exception:
            pass

def wait_for_http(url, timeout=60):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(1)
    return False

async def cdp_session():
    with urllib.request.urlopen(CDP_JSON_URL, timeout=5) as resp:
        targets = json.loads(resp.read().decode("utf-8"))

    ws_url = None
    for t in targets:
        if t.get("type") == "page" and "webSocketDebuggerUrl" in t:
            ws_url = t["webSocketDebuggerUrl"]
            break

    if not ws_url:
        raise RuntimeError(f"Could not find a page target in CDP targets: {targets}")

    print(f"Connecting to CDP WebSocket: {ws_url}")
    async with websockets.connect(ws_url, max_size=50 * 1024 * 1024) as ws:
        msg_id = 0

        async def send_cmd(method, params=None):
            nonlocal msg_id
            msg_id += 1
            req_id = msg_id
            payload = {"id": req_id, "method": method}
            if params:
                payload["params"] = params
            await ws.send(json.dumps(payload))
            while True:
                resp_raw = await ws.recv()
                resp = json.loads(resp_raw)
                if resp.get("id") == req_id:
                    if "error" in resp:
                        raise RuntimeError(f"CDP error in {method}: {resp['error']}")
                    return resp.get("result", {})

        await send_cmd("Page.enable")
        await send_cmd("DOM.enable")

        # Set 1920x1080 viewport
        await send_cmd("Emulation.setDeviceMetricsOverride", {
            "width": 1920,
            "height": 1080,
            "deviceScaleFactor": 1,
            "mobile": False,
        })

        # Navigate to Streamlit App URL
        print(f"Navigating page to {APP_URL}...")
        await send_cmd("Page.navigate", {"url": APP_URL})

        # Wait for Streamlit to render tabs
        print("Waiting for Streamlit tabs to mount in DOM...")
        tabs_ready = False
        for i in range(60):
            eval_res = await send_cmd("Runtime.evaluate", {
                "expression": "document.querySelectorAll('button[data-baseweb=\"tab\"]').length",
                "returnByValue": True,
            })
            val = eval_res.get("result", {}).get("value", 0)
            if val >= 7:
                tabs_ready = True
                print(f"Detected {val} tabs mounted in DOM (after {i+1}s).")
                break
            if (i + 1) % 5 == 0:
                print(f"  ...waiting ({i+1}s elapsed, current tab count = {val})")
            await asyncio.sleep(1)

        if not tabs_ready:
            raise TimeoutError("Timed out waiting for 7 tabs to mount in Streamlit DOM.")

        # Brief extra delay for first tab components
        await asyncio.sleep(3)

        results = []
        for tab in TAB_CONFIG:
            idx = tab["index"]
            name = tab["name"]
            slug = tab["slug"]
            wait_sec = tab["wait"]

            print(f"\n--- Capturing Tab {idx}: {name} ---")
            # Click the tab button
            click_expr = f"""
            (() => {{
                const tabs = document.querySelectorAll('button[data-baseweb="tab"]');
                if (tabs[{idx}]) {{
                    tabs[{idx}].click();
                    window.scrollTo(0, 0);
                    return true;
                }}
                return false;
            }})()
            """
            eval_click = await send_cmd("Runtime.evaluate", {
                "expression": click_expr,
                "returnByValue": True,
            })
            print(f"Clicked tab {idx}: {eval_click.get('result', {}).get('value')}")

            # Wait for content rendering
            print(f"Waiting {wait_sec}s for plots and widgets to render...")
            await asyncio.sleep(wait_sec)

            # Scroll to top to ensure clean layout
            await send_cmd("Runtime.evaluate", {"expression": "window.scrollTo(0, 0);"})
            await asyncio.sleep(0.5)

            # Capture screenshot
            shot_res = await send_cmd("Page.captureScreenshot", {
                "format": "png",
            })
            b64_data = shot_res.get("data")
            img_bytes = base64.b64decode(b64_data)

            out_path = os.path.join(ASSETS_DIR, slug)
            with open(out_path, "wb") as f:
                f.write(img_bytes)

            file_size = os.path.getsize(out_path)
            print(f"Saved: {out_path} ({file_size:,} bytes)")
            results.append((name, slug, file_size))

        return results

def main():
    chrome_exe = find_chrome()
    print(f"Found browser executable: {chrome_exe}")

    print("Cleaning up any existing processes on ports 8501 and 9222...")
    kill_ports([STREAMLIT_PORT, CDP_PORT])
    time.sleep(1)

    streamlit_cmd = [
        sys.executable, "-m", "streamlit", "run",
        os.path.join(REPO_ROOT, "app", "app.py"),
        f"--server.port={STREAMLIT_PORT}",
        "--server.headless=true",
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false"
    ]

    print("Launching Streamlit server (DEVNULL output)...")
    st_proc = subprocess.Popen(
        streamlit_cmd,
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    chrome_proc = None
    temp_dir = tempfile.mkdtemp()
    try:
        print(f"Polling Streamlit health at {HEALTH_URL}...")
        if not wait_for_http(HEALTH_URL, timeout=60):
            raise TimeoutError("Streamlit server failed to start within 60 seconds.")
        print("Streamlit server is healthy and responding.")

        chrome_cmd = [
            chrome_exe,
            "--headless=new",
            f"--remote-debugging-port={CDP_PORT}",
            f"--user-data-dir={temp_dir}",
            "--window-size=1920,1080",
            "--hide-scrollbars",
            "--disable-gpu",
            "--disable-extensions",
            "--no-first-run",
            "--no-default-browser-check",
            "about:blank"
        ]
        print(f"Launching headless Chrome on port {CDP_PORT}...")
        chrome_proc = subprocess.Popen(chrome_cmd)

        print(f"Polling CDP endpoint at {CDP_JSON_URL}...")
        if not wait_for_http(CDP_JSON_URL, timeout=30):
            raise TimeoutError("Headless Chrome CDP endpoint failed to respond within 30 seconds.")
        print("CDP endpoint is active.")

        # Run async CDP session
        results = asyncio.run(cdp_session())

        print("\n================ Screenshot Capture Summary ================")
        all_ok = True
        for name, slug, size in results:
            status = "✓ OK" if size > 10000 else "✗ FAILED"
            print(f"  {name:20} -> {slug:32} ({size:>8,} bytes) [{status}]")
            if size <= 10000:
                all_ok = False

        if all_ok:
            print("\nAll 7 dashboard screenshots captured successfully!")
            return 0
        else:
            print("\nError: One or more screenshots failed or are too small.")
            return 1

    finally:
        print("\nCleaning up processes...")
        if chrome_proc:
            chrome_proc.terminate()
            try:
                chrome_proc.wait(timeout=5)
            except Exception:
                chrome_proc.kill()
        if st_proc:
            st_proc.terminate()
            try:
                st_proc.wait(timeout=5)
            except Exception:
                st_proc.kill()
        kill_ports([STREAMLIT_PORT, CDP_PORT])
        print("Cleanup completed.")

if __name__ == "__main__":
    sys.exit(main())
