#!/usr/bin/env bash
# ==============================================================================
# Master Reproducibility Pipeline Launcher for Linux / macOS
# Cross-Domain Generalization & Deployment Safety of Malaria AI Models
# Princess Marie Louise Children's Hospital Cohort (N = 4,056 Micrographs)
# Author: Wilfred Ayine Asumboya (University of Ghana, Legon)
# ==============================================================================

set -e

# Terminal ANSI Color Palette
CYAN='\033[38;2;0;229;255m'
MINT='\033[38;2;0;255;170m'
WHITE='\033[38;2;245;247;250m'
GRAY='\033[38;2;120;128;140m'
AMBER='\033[38;2;255;179;0m'
RESET='\033[0m'
BOLD='\033[1m'

echo -e "\n${CYAN}════════════════════════════════════════════════════════════════════════════════${RESET}"
echo -e "${BOLD}${WHITE}  MALARIA AI CLINICAL BENCHMARK — LINUX RUNTIME BOOTSTRAP${RESET}"
echo -e "${GRAY}  Dept. of Biomedical Engineering, University of Ghana, Legon${RESET}"
echo -e "${CYAN}════════════════════════════════════════════════════════════════════════════════${RESET}\n"

# 1. Verify Python 3 installation
if ! command -v python3 &> /dev/null; then
    echo -e "${AMBER}[ ✖ FAIL ] python3 is not installed or not found in system PATH!${RESET}"
    echo -e "Please install Python 3.10+ (e.g., sudo apt update && sudo apt install python3 python3-venv python3-pip)"
    exit 1
fi

PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo -e "  ${MINT}[ ✔ PASS ]${RESET} Detected host Python runtime: ${BOLD}${WHITE}Python ${PY_VER}${RESET}"

# 2. Automatically create virtual environment if missing
if [ ! -d ".venv" ]; then
    echo -e "  ${CYAN}[ ⚡ READY ]${RESET} Creating dedicated virtual environment: ${BOLD}.venv${RESET}..."
    python3 -m venv .venv
    echo -e "  ${MINT}[ ✔ PASS ]${RESET} Virtual environment successfully initialized."
fi

# 3. Activate virtual environment
source .venv/bin/activate
echo -e "  ${MINT}[ ✔ PASS ]${RESET} Virtual environment active: ${BOLD}${WHITE}$(which python)${RESET}"

# 4. Hand off execution to the aesthetic Python telemetry console
python run_pipeline.py "$@"
