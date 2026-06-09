#!/usr/bin/env python3
"""
OpenCareer Web GUI Launcher

Starts the FastAPI backend server and opens the browser to the Vite dev server.
Usage: python launch_web.py [--port 8080] [--no-browser]
"""

import argparse
import logging
import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

# Windows encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Ensure demo/ is on path
_demo_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_demo_dir))

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv(_demo_dir / ".env")
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("launch_web")


def _npm_cmd(cmd: str) -> str:
    """On Windows, npm/npx are .cmd files; Python subprocess needs the extension."""
    if sys.platform == "win32":
        return cmd + ".cmd"
    return cmd


def start_frontend(web_dir: Path) -> subprocess.Popen | None:
    """Start the Vite dev server."""
    if not (web_dir / "node_modules").exists():
        logger.info("Installing frontend dependencies (npm install)...")
        subprocess.run(
            [_npm_cmd("npm"), "install"],
            cwd=str(web_dir),
            check=True,
        )

    logger.info("Starting Vite dev server...")
    return subprocess.Popen(
        [_npm_cmd("npx"), "vite", "--host"],
        cwd=str(web_dir),
        stdout=sys.stdout,
        stderr=sys.stderr,
    )


def main():
    parser = argparse.ArgumentParser(description="OpenCareer Web GUI Launcher")
    parser.add_argument("--port", type=int, default=8080, help="Backend API port (default: 8080)")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser")
    parser.add_argument("--no-frontend", action="store_true", help="Don't start frontend dev server")
    args = parser.parse_args()

    web_dir = _demo_dir / "web"
    if not args.no_frontend and not (web_dir / "package.json").exists():
        logger.warning("Frontend not found (demo/web/package.json missing). "
                       "Skipping frontend. Use --no-frontend to suppress this warning.")
        args.no_frontend = True

    print("\n" + "=" * 60)
    print("  OpenCareer Web GUI")
    print("=" * 60)

    # Start frontend
    vite_proc = None
    if not args.no_frontend:
        vite_proc = start_frontend(web_dir)
        time.sleep(2)

    # Start backend
    import uvicorn
    host = os.getenv("WEB_API_HOST", "0.0.0.0")
    port = args.port or int(os.getenv("WEB_API_PORT", "8080"))
    api_url = f"http://{host.replace('0.0.0.0', 'localhost')}:{port}"

    print(f"\n  Backend API:  {api_url}")
    print(f"  Frontend:     http://localhost:5173")
    print(f"  Health check: {api_url}/api/health")
    print("\n  Press Ctrl+C to stop all servers\n")
    print("=" * 60 + "\n")

    # Open browser
    if not args.no_browser:
        webbrowser.open("http://localhost:5173")

    try:
        uvicorn.run(
            "web_api.server:app",
            host=host,
            port=port,
            reload=True,
            log_level="info",
        )
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        if vite_proc:
            vite_proc.terminate()
            vite_proc.wait()


if __name__ == "__main__":
    main()
