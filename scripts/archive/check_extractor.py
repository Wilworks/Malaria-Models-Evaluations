import subprocess
import shutil
from pathlib import Path

rar_path = Path.home() / "Downloads" / "Thick_Ghana.part2.rar"
print(f"File exists: {rar_path.exists()}, Size: {rar_path.stat().st_size / (1024*1024):.2f} MB")

# Check available extractors
tools = [
    r"C:\Program Files\7-Zip\7z.exe",
    r"C:\Program Files (x86)\7-Zip\7z.exe",
    r"C:\Program Files\WinRAR\WinRAR.exe",
    r"C:\Program Files\WinRAR\UnRAR.exe",
    "7z",
    "unrar",
    "tar"
]

found = []
for t in tools:
    p = shutil.which(t) or (Path(t) if Path(t).exists() else None)
    if p:
        found.append(str(p))
        print(f"[FOUND] Extractor: {p}")

if not found:
    print("[Notice] No standard CLI archive tool found in common locations.")
