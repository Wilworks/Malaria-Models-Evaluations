#!/usr/bin/env bash
# ==============================================================================
# WAM-Bench: The West African Malaria AI Benchmark
# A Multi-Center Clinical Evaluation Across West African Blood Smears
# Author: Wilfred Ayine Asumboya (University of Ghana, Legon)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "================================================================================"
echo "  WAM-BENCH: WEST AFRICAN MALARIA AI CLINICAL BENCHMARK"
echo "  Department of Biomedical Engineering, University of Ghana, Legon"
echo "================================================================================"

# 1. Verify Python 3
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed or not found in system PATH."
    echo "Please install Python 3.10+ (e.g., sudo apt install python3 python3-venv python3-pip)"
    exit 1
fi

PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
echo "[INFO] Host Python: Python ${PY_VER}"

# 2. Virtual Environment Setup
if [ ! -d ".venv" ]; then
    echo "[INFO] Initializing Python virtual environment (.venv)..."
    python3 -m venv .venv
    echo "[INFO] Virtual environment created successfully."
fi

# 3. Activate Virtual Environment
source .venv/bin/activate
echo "[INFO] Active Environment: $(which python)"

# 4. Dependency Sync
if [ -f "requirements.txt" ]; then
    echo "[INFO] Verifying package dependencies..."
    pip install --quiet --upgrade pip
    pip install --quiet -r requirements.txt
    echo "[INFO] Dependencies up to date."
fi

# 5. Hand off to Pipeline Orchestrator
echo "[INFO] Launching Benchmark Pipeline..."
python run_pipeline.py "$@"
