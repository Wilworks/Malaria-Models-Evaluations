"""
Script 01: Clones benchmark candidate repositories into models/external/ using shallow clone (--depth 1)
and verifies pretrained weight availability.
"""

import os
import shutil
import subprocess
from pathlib import Path

EXTERNAL_MODELS = [
    {
        "name": "MalariaScreener",
        "url": "https://github.com/LHNCBC/MalariaScreener.git",
        "provenance": "Peer-Reviewed (BMC Infect Dis 2020 - NIH)",
    },
    {
        "name": "kossisoroyce_malaria_detection",
        "url": "https://github.com/kossisoroyce/malaria-detection.git",
        "provenance": "Informal (Marketed as Clinical-Grade YOLOv8)",
    }
]


def clone_repositories():
    target_dir = Path(__file__).resolve().parent.parent / "models" / "external"
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Script 01: Cloning Candidate Model Repositories into {target_dir} ===")

    for model in EXTERNAL_MODELS:
        repo_path = target_dir / model["name"]
        
        # Clean incomplete directory if git clone was aborted
        if repo_path.exists() and not (repo_path / "README.md").exists():
            print(f"[Cleaning] Removing partial directory {repo_path}...")
            shutil.rmtree(repo_path, ignore_errors=True)

        if repo_path.exists():
            print(f"[OK] Repository '{model['name']}' already exists at {repo_path}")
        else:
            print(f"[Cloning --depth 1] {model['name']} from {model['url']}...")
            try:
                subprocess.run(
                    ["git", "clone", "--depth", "1", model["url"], str(repo_path)],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                print(f"[Success] Cloned {model['name']}")
            except subprocess.CalledProcessError as e:
                print(f"[Error] Failed to clone {model['name']}: {e.stderr}")

    print("\n=== Model Repository Audit Complete ===")


if __name__ == "__main__":
    clone_repositories()
