"""Record the dashboard replay as a video for X (1280x720 or 1080x1080), plus a final-frame PNG.

    pip install playwright && playwright install chromium
    python -m analysis.record_replay                       # runs/replay.webm + runs/final_frame.png
    python -m analysis.record_replay --square --seconds 20
    ffmpeg -i runs/replay.webm -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p runs/replay.mp4   # X wants mp4
"""
import argparse, os, shutil
from playwright.sync_api import sync_playwright

VIDEO_CSS = """
.controls, header .sub, footer, #final { display: none !important; }
body { padding: 16px 24px !important; }
header h1 { font-size: 26px !important; margin-bottom: 4px !important; }
.vs { margin-top: 6px !important; }
.banner { padding: 5px 12px !important; margin-bottom: 10px !important; }
.board { margin: 12px 0 10px !important; }
.card { padding: 12px 16px !important; }
.card .big { margin: 4px 0 6px !important; } .card .big .v { font-size: 34px !important; }
.stats div:nth-child(n+4) { display: none; }
.panel { padding: 10px 14px !important; } #feed { max-height: 360px !important; }
#chart { max-height: 335px; } .legend { margin-bottom: 0 !important; } #ylab { display: none; }
"""
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", default=os.path.join(ROOT, "runs", "dashboard.html"))
    ap.add_argument("--seconds", type=int, default=20, choices=[10, 20, 40])
    ap.add_argument("--square", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "runs"))
    args = ap.parse_args()
    size = {"width": 1080, "height": 1080} if args.square else {"width": 1280, "height": 720}
    tmp = os.path.join(args.out, "_video")
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=os.environ.get("CHROME_PATH") or None)
        ctx = b.new_context(viewport=size, device_scale_factor=1, record_video_dir=tmp, record_video_size=size)
        pg = ctx.new_page()
        pg.goto("file://" + os.path.abspath(args.html))
        pg.add_style_tag(content=VIDEO_CSS)  # compact frame: title, scoreboard, chart + feed
        pg.evaluate("window.dispatchEvent(new Event('resize'))")
        pg.evaluate(f"document.querySelector(\"#speed button[data-s='{args.seconds}']\").click()")
        pg.wait_for_timeout(800)
        pg.evaluate("document.querySelector('#play').click()")
        pg.wait_for_timeout(args.seconds * 1000 + 2500)
        pg.screenshot(path=os.path.join(args.out, "final_frame.png"))
        video = pg.video.path()
        ctx.close(); b.close()
    dst = os.path.join(args.out, "replay.webm")
    shutil.move(video, dst)
    shutil.rmtree(tmp, ignore_errors=True)
    print("wrote", dst, "and final_frame.png")


if __name__ == "__main__":
    main()
