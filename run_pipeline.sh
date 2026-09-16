#!/usr/bin/env bash
# ==============================================================================
# Master Reproducibility Pipeline Launcher for Linux / macOS
# Cross-Domain Generalization & Deployment Safety of Malaria AI Models
# Princess Marie Louise Children's Hospital Cohort (N = 4,056 Micrographs)
# Author: Wilfred Ayine Asumboya (University of Ghana, Legon)
# ==============================================================================

set -e

# Terminal ANSI Color Palette
CYAN='\033[1;96m'
MINT='\033[1;92m'
WHITE='\033[1;97m'
GRAY='\033[90m'
AMBER='\033[1;93m'
MAGENTA='\033[1;95m'
BLUE='\033[1;94m'
RESET='\033[0m'
BOLD='\033[1m'

# Animated Braided Spinner
spinner() {
    local pid=$1
    local msg=$2
    local delay=0.08
    local spinstr='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
    while kill -0 "$pid" 2>/dev/null; do
        local temp=${spinstr#?}
        printf "  ${CYAN}[%c]${RESET} ${WHITE}%s${RESET}\r" "$spinstr" "$msg"
        spinstr=$temp${spinstr%"$temp"}
        sleep $delay
    done
    printf "                                                                               \r"
}

echo -e "\n${CYAN}╭──────────────────────────────────────────────────────────────────────────────╮${RESET}"
echo -e "${CYAN}│${RESET}  ${BOLD}${WHITE}MALARIA AI CLINICAL BENCHMARK — LINUX AUTOMATED BOOTSTRAP${RESET}                   ${CYAN}│${RESET}"
echo -e "${CYAN}│${RESET}  ${GRAY}Department of Biomedical Engineering, University of Ghana, Legon${RESET}            ${CYAN}│${RESET}"
echo -e "${CYAN}╰──────────────────────────────────────────────────────────────────────────────╯${RESET}\n"

# 1. Verify Python 3 installation
if ! command -v python3 &> /dev/null; then
    echo -e "  ${AMBER}[ ✖ FAIL ]${RESET} Python 3 is not installed or not found in system PATH!"
    echo -e "  Please install Python 3.10+ (e.g., sudo apt update && sudo apt install -y python3 python3-venv python3-pip)"
    exit 1
fi

PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
echo -e "  ${MINT}[ ✔ PASS ]${RESET} Detected host Python runtime: ${BOLD}${WHITE}Python ${PY_VER}${RESET} (64-bit)"

# 2. Automatically create virtual environment if missing
if [ ! -d ".venv" ]; then
    echo -e "  ${CYAN}[ ⚡ BOOT ]${RESET} Creating dedicated virtual environment: ${BOLD}.venv${RESET}..."
    (python3 -m venv .venv) &
    PID=$!
    spinner $PID "Initializing isolated Python virtual environment (.venv)..."
    wait $PID
    echo -e "  ${MINT}[ ✔ PASS ]${RESET} Isolated virtual environment successfully initialized."
fi

# 3. Activate virtual environment
source .venv/bin/activate
echo -e "  ${MINT}[ ✔ PASS ]${RESET} Environment active: ${BOLD}${CYAN}$(which python)${RESET}"

# 4. Verify/Install dependencies with animated spinner
if [ -f "requirements.txt" ]; then
    echo -e "  ${BLUE}[ ℹ SYNC ]${RESET} Verifying locked dependencies..."
    (pip install --quiet -r requirements.txt) &
    PID=$!
    spinner $PID "Checking deep learning libraries (PyTorch, TensorFlow, Rich, OpenCV)..."
    wait $PID
    echo -e "  ${MINT}[ ✔ PASS ]${RESET} All packages verified and ready for execution."
fi

echo -e "\n  ${MAGENTA}▶ Starting Aesthetic Telemetry Console...${RESET}\n"
sleep 0.5

# 5. Hand off execution to the aesthetic Python telemetry console
python run_pipeline.py "$@"
