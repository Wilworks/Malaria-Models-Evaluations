#!/usr/bin/env python3
"""
Resumable download script for Kano Nigeria Thin Blood Smear Dataset (Zenodo 13763939).
Zero external dependencies (uses standard library urllib + ssl).
Safely resumes from existing Safari .download files or previous partial downloads.
Auto-reconnects on network drops without losing any downloaded data.
"""

import os
import sys
import time
import shutil
import ssl
import urllib.request
import urllib.error
from pathlib import Path

ZENODO_URL = "https://zenodo.org/records/13763939/files/Thin%20blood%20smear%20images.zip?download=1"
TOTAL_SIZE = 5437283585  # exact size in bytes (5.44 GB)

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari/605.1.15"
)

def get_ssl_context():
    """Create SSL context, falling back to unverified if macOS certs aren't configured in Python."""
    try:
        ctx = ssl.create_default_context()
        # Test if certificates work
        urllib.request.urlopen("https://zenodo.org", context=ctx, timeout=3)
        return ctx
    except Exception:
        # Fallback for Python installations missing macOS root certificates
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx

def format_bytes(b):
    if b >= 1024 ** 3:
        return f"{b / (1024 ** 3):.2f} GB"
    elif b >= 1024 ** 2:
        return f"{b / (1024 ** 2):.1f} MB"
    elif b >= 1024:
        return f"{b / 1024:.1f} KB"
    return f"{b} B"

def main():
    home = Path.home()
    target_dir = home / "Downloads"
    final_file = target_dir / "Thin_blood_smear_images.zip"
    part_file = target_dir / "Thin_blood_smear_images.zip.part"

    if final_file.exists() and final_file.stat().st_size == TOTAL_SIZE:
        print(f"File already completely downloaded at: {final_file}")
        return

    # Check if Safari partial download exists to rescue the 1.67 GB
    safari_part = target_dir / "Thin blood smear images.zip-3.download" / "Thin blood smear images.zip"
    if not part_file.exists() and safari_part.exists():
        safari_size = safari_part.stat().st_size
        print(f"============================================================")
        print(f"  FOUND EXISTING DOWNLOAD PROGRESS: {format_bytes(safari_size)}")
        print(f"============================================================")
        print(f"Adopting your existing 1.67 GB so you don't lose any progress...")
        shutil.copyfile(safari_part, part_file)
        print(f"Successfully rescued {format_bytes(part_file.stat().st_size)}!\n")

    ssl_ctx = get_ssl_context()
    chunk_size = 1024 * 1024  # 1 MB chunk

    print(f"Connecting to Zenodo (Record 13763939)...")
    print(f"Destination: {final_file}")

    while True:
        downloaded = part_file.stat().st_size if part_file.exists() else 0

        if downloaded >= TOTAL_SIZE:
            part_file.rename(final_file)
            print(f"\n\n============================================================")
            print(f"  DOWNLOAD COMPLETE: {final_file}")
            print(f"  Final Size: {format_bytes(final_file.stat().st_size)}")
            print(f"============================================================")
            break

        req = urllib.request.Request(ZENODO_URL)
        req.add_header("User-Agent", USER_AGENT)
        if downloaded > 0:
            req.add_header("Range", f"bytes={downloaded}-")
            print(f"\nResuming stream from byte {downloaded:,} ({format_bytes(downloaded)} / {format_bytes(TOTAL_SIZE)})...")
        else:
            print(f"\nStarting stream from byte 0 ({format_bytes(TOTAL_SIZE)} total)...")

        try:
            with urllib.request.urlopen(req, context=ssl_ctx, timeout=30) as resp:
                status = resp.status
                if status not in (200, 206):
                    print(f"Server returned HTTP {status}. Retrying in 5 seconds...")
                    time.sleep(5)
                    continue

                mode = "ab" if downloaded > 0 else "wb"
                start_time = time.time()
                bytes_this_session = 0

                with open(part_file, mode) as f:
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        bytes_this_session += len(chunk)

                        elapsed = max(time.time() - start_time, 0.001)
                        speed = bytes_this_session / elapsed
                        pct = (downloaded / TOTAL_SIZE) * 100
                        remaining_bytes = TOTAL_SIZE - downloaded
                        eta_seconds = remaining_bytes / speed if speed > 0 else 0
                        eta_min = int(eta_seconds // 60)
                        eta_sec = int(eta_seconds % 60)

                        filled = int(pct // 4)
                        bar = f"[{'=' * filled}{' ' * (25 - filled)}]"
                        print(
                            f"\r{bar} {pct:5.1f}% | {format_bytes(downloaded)} / {format_bytes(TOTAL_SIZE)} "
                            f"| {format_bytes(speed)}/s | ETA: {eta_min}m {eta_sec:02d}s   ",
                            end="",
                            flush=True,
                        )

        except (urllib.error.URLError, TimeoutError, ConnectionResetError, OSError) as e:
            print(f"\nNetwork interrupted ({type(e).__name__}). Reconnecting in 3s without losing progress...")
            time.sleep(3)
        except KeyboardInterrupt:
            print(f"\nDownload paused by user. Progress safely saved at: {part_file}")
            sys.exit(0)

if __name__ == "__main__":
    main()
