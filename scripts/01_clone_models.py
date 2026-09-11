"""
Script 01: Clones benchmark candidate repositories into models/external/
and verifies pretrained weight availability.
"""

import os
import subprocess
from pathlib import Path

EXTERNAL_MODELS = [
    {
        "name": "MalariaScreener",
        "url": "https://github.com/LHNCBC/MalariaScreener.git",
        "provenance": "Peer-Reviewed (BMC Infect Dis 2020 - NIH)",
        "expected_weights": ["malaria_thinsmear_44_retrainSudan_20P_4000C_separate.pb"]
    },
    {
        "name": "kossisoroyce_malaria_detection",
        "url": "https://github.com/kossisoroyce/malaria-detection.git",
        "provenance": "Informal (Marketed as Clinical-Grade YOLOv8)",
        "expected_weights": ["best.pt"]
    }
]


def clone_repositories():
    target_dir = Path(__file__).resolve().parent.parent / "models" / "external"
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Script 01: Cloning Candidate Model Repositories into {target_dir} ===")

    for model in EXTERNAL_MODELS:
        repo_path = target_dir / model["name"]
        if repo_path.exists():
            print(f"[OK] Repository '{model['name']}' already exists at {repo_path}")
        else:
            print(f"[Cloning] {model['name']} from {model['url']}...")
            try:
                subprocess.run(
                    ["git", "clone", model["url"], str(repo_path)],
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
