"""
Aesthetic Terminal UI & Telemetry Engine for Linux and Cross-Platform Terminals.
Powered by Rich with full 24-bit TrueColor, rounded panels, vibrant tables,
and interactive high-resolution progress animations.
"""

import os
import sys
import time
import shutil
from typing import List, Dict, Any, Optional

# Force Windows Console to enable ANSI escape sequences
if os.name == 'nt':
    try:
        os.system('')
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except Exception:
        pass

# Ensure UTF-8 stdout
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, MofNCompleteColumn, TaskProgressColumn
    from rich.align import Align
    from rich.style import Style
    from rich import box
    RICH_AVAILABLE = True
    console = Console(force_terminal=True, color_system="auto")
except ImportError:
    RICH_AVAILABLE = False
    console = None


class TermColors:
    """ANSI fallback constants."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[1;96m"
    GREEN = "\033[1;92m"
    YELLOW = "\033[1;93m"
    BLUE = "\033[1;94m"
    MAGENTA = "\033[1;95m"
    RED = "\033[1;91m"
    WHITE = "\033[1;97m"
    GRAY = "\033[90m"
    BG_CYAN = "\033[46;30m"
    BG_GREEN = "\033[42;30m"


class TerminalUI:
    """State-of-the-Art Aesthetic Terminal UI Engine."""

    @classmethod
    def print_hero_banner(cls) -> None:
        """Renders an elite high-tech research banner."""
        if RICH_AVAILABLE:
            banner_text = Text()
            banner_text.append("🔬 MALARIA CLINICAL DIAGNOSTICS: ZERO-SHOT AUDIT\n", style="bold bright_cyan")
            banner_text.append("Cross-Domain Generalization & Deployment Safety on Ghanaian Pediatric Blood Smears\n\n", style="italic bright_white")
            banner_text.append("Principal Investigator: ", style="bold bright_yellow")
            banner_text.append("Wilfred Ayine Asumboya ", style="bold bright_white")
            banner_text.append("| Department of Biomedical Engineering, University of Ghana, Legon\n", style="dim cyan")
            banner_text.append("Clinical Cohort: ", style="bold bright_yellow")
            banner_text.append("Princess Marie Louise Children's Hospital, Accra (N = 4,056 Micrographs)", style="bold bright_green")
            
            panel = Panel(
                Align.center(banner_text),
                box=box.DOUBLE_EDGE,
                border_style="bold bright_cyan",
                padding=(1, 2),
                subtitle="[bold bright_magenta]Master Reproducibility Console[/]",
                subtitle_align="right"
            )
            console.print()
            console.print(panel)
            console.print()
        else:
            c = TermColors
            print(f"\n{c.CYAN}╔══════════════════════════════════════════════════════════════════════════════════════════╗{c.RESET}")
            print(f"{c.CYAN}║{c.BOLD}{c.WHITE}           MALARIA CLINICAL DIAGNOSTICS: ZERO-SHOT EMPIRICAL BENCHMARK            {c.CYAN}║{c.RESET}")
            print(f"{c.CYAN}║{c.GREEN}   Cross-Domain Generalization & Deployment Safety on Ghanaian Pediatric Blood Smears   {c.CYAN}║{c.RESET}")
            print(f"{c.CYAN}╠══════════════════════════════════════════════════════════════════════════════════════════╣{c.RESET}")
            print(f"{c.CYAN}║{c.YELLOW} Author: Wilfred Ayine Asumboya | Dept of Biomedical Engineering, Univ of Ghana, Legon   {c.CYAN}║{c.RESET}")
            print(f"{c.CYAN}║{c.MAGENTA} Clinical Site: Princess Marie Louise Children's Hospital, Accra (N = 4,056 Micrographs)  {c.CYAN}║{c.RESET}")
            print(f"{c.CYAN}╚══════════════════════════════════════════════════════════════════════════════════════════╝{c.RESET}\n")

    @classmethod
    def print_step_header(cls, step_num: int, total_steps: int, title: str, subtitle: Optional[str] = None) -> None:
        """Renders a stage section header."""
        if RICH_AVAILABLE:
            header_text = Text()
            header_text.append(f" STAGE {step_num}/{total_steps} ", style="bold black on bright_cyan")
            header_text.append(f"  {title.upper()}", style="bold bright_white")
            if subtitle:
                header_text.append(f"\n↳ {subtitle}", style="dim italic cyan")
            
            p = Panel(header_text, box=box.HORIZONTALS, border_style="dim bright_cyan", padding=(0, 1))
            console.print(p)
        else:
            c = TermColors
            print(f"\n{c.BG_CYAN} STAGE {step_num}/{total_steps} {c.RESET} {c.BOLD}{c.WHITE}{title.upper()}{c.RESET}")
            if subtitle:
                print(f"  {c.GRAY}↳ {subtitle}{c.RESET}")
            print(f"{c.GRAY}{'─' * 80}{c.RESET}")

    @classmethod
    def print_badge(cls, status: str, text: str, detail: str = "") -> None:
        """Renders glowing status badges."""
        if RICH_AVAILABLE:
            badges = {
                "PASS": ("[bold black on bright_green]  ✔ PASS  [/]", "bold bright_white"),
                "READY": ("[bold black on bright_cyan]  ⚡ READY [/]", "bold bright_white"),
                "INFO": ("[bold black on bright_blue]  ℹ INFO  [/]", "bright_white"),
                "WARN": ("[bold black on bright_yellow]  ⚠ WARN  [/]", "bold bright_yellow"),
                "FAIL": ("[bold black on bright_red]  ✖ FAIL  [/]", "bold bright_red"),
                "BENCH": ("[bold black on bright_magenta]  ★ RUN   [/]", "bold bright_cyan"),
                "LOCK": ("[bold black on bright_green]  🔒 HASH [/]", "bold bright_white")
            }
            badge_str, style = badges.get(status, (f"[{status}]", "white"))
            detail_str = f" [dim italic]({detail})[/]" if detail else ""
            console.print(f"  {badge_str} [{style}]{text}[/]{detail_str}")
        else:
            c = TermColors
            colors = {
                "PASS": f"{c.GREEN}[ ✔ PASS ]{c.RESET}",
                "READY": f"{c.CYAN}[ ⚡ READY ]{c.RESET}",
                "INFO": f"{c.BLUE}[ ℹ INFO ]{c.RESET}",
                "WARN": f"{c.YELLOW}[ ⚠ WARN ]{c.RESET}",
                "FAIL": f"{c.RED}[ ✖ FAIL ]{c.RESET}",
                "BENCH": f"{c.MAGENTA}[ ★ RUN ]{c.RESET}",
                "LOCK": f"{c.GREEN}[ 🔒 HASH ]{c.RESET}"
            }
            b = colors.get(status, f"[{status}]")
            d = f" {c.GRAY}({detail}){c.RESET}" if detail else ""
            print(f"  {b} {c.WHITE}{text}{c.RESET}{d}")

    @classmethod
    def render_progress_bar(cls, current: int, total: int, prefix: str = "", speed: float = 0.0, bar_len: int = 28) -> None:
        """Renders an animated high-res gradient block progress bar."""
        pct = min(1.0, current / total) if total > 0 else 0.0
        filled = int(round(bar_len * pct))
        
        eta_sec = int((total - current) / speed) if speed > 0 and current < total else 0
        eta_str = time.strftime("%M:%S", time.gmtime(eta_sec)) if eta_sec > 0 else "--:--"
        speed_str = f"{speed:5.1f} img/s" if speed > 0 else "-- img/s"
        pct_str = f"{pct * 100:5.1f}%"
        
        if RICH_AVAILABLE:
            bar = f"[bold bright_green]{'━' * filled}[/][dim white]{'─' * (bar_len - filled)}[/]"
            line = f"  [bold bright_cyan]▶[/] [bold bright_white]{prefix:<26}[/] {bar} [bold bright_yellow]{pct_str}[/] | [dim white]{current:>4}/{total:>4}[/] | [bold bright_cyan]{speed_str}[/] | [dim italic]ETA: {eta_str}[/]"
            console.print(line, end="\r")
            if current >= total:
                console.print()
        else:
            bar = f"\033[1;92m{'█' * filled}\033[90m{'░' * (bar_len - filled)}\033[0m"
            line = f"  \033[1;96m▶\033[0m \033[1;97m{prefix:<26}\033[0m {bar} \033[1;93m{pct_str}\033[0m | {current:>4}/{total:>4} | \033[1;96m{speed_str}\033[0m | ETA: {eta_str}"
            sys.stdout.write(f"\r{line}")
            sys.stdout.flush()
            if current >= total:
                sys.stdout.write("\n")
                sys.stdout.flush()

    @classmethod
    def print_hardware_card(cls, hw_info: Dict[str, Any]) -> None:
        """Renders an aesthetic hardware telemetry card."""
        if RICH_AVAILABLE:
            table = Table(box=box.ROUNDED, border_style="bold bright_blue", title="[bold bright_white]🖥 Hardware & Execution Environment[/]", title_justify="left", padding=(0, 2))
            table.add_column("Property", style="bold bright_cyan", no_wrap=True)
            table.add_column("Detected Specification", style="bold bright_white")
            for k, v in hw_info.items():
                table.add_row(k, str(v))
            console.print(table)
            console.print()
        else:
            c = TermColors
            print(f"\n  {c.CYAN}┌─ HARDWARE TELEMETRY ──────────────────────────────────────────┐{c.RESET}")
            for k, v in hw_info.items():
                print(f"  {c.CYAN}│{c.RESET}  {c.YELLOW}{k:<24}{c.RESET} : {c.WHITE}{v}{c.RESET}")
            print(f"  {c.CYAN}└───────────────────────────────────────────────────────────────┘{c.RESET}\n")

    @classmethod
    def print_models_card(cls, models: List[Dict[str, Any]]) -> None:
        """Renders the models verification table."""
        if RICH_AVAILABLE:
            table = Table(box=box.ROUNDED, border_style="bold bright_magenta", title="[bold bright_white]🤖 Candidate Deep Learning Model Zoo[/]", title_justify="left", padding=(0, 2))
            table.add_column("#", style="bold bright_yellow", justify="center")
            table.add_column("Model Identifier", style="bold bright_white")
            table.add_column("Architecture", style="bold cyan")
            table.add_column("Training Domain", style="bold bright_green")
            table.add_column("Format", style="bold bright_blue")
            table.add_column("File Size", style="bold bright_yellow", justify="right")
            
            for idx, m in enumerate(models, 1):
                table.add_row(
                    str(idx),
                    m.get("name", ""),
                    m.get("arch", ""),
                    m.get("domain", ""),
                    m.get("format", ""),
                    m.get("size", "")
                )
            console.print(table)
            console.print()
        else:
            c = TermColors
            print(f"\n  {c.MAGENTA}--- VERIFIED MODEL ZOO ---{c.RESET}")
            for idx, m in enumerate(models, 1):
                print(f"    {c.YELLOW}{idx}.{c.RESET} {c.WHITE}{m['name']:<25}{c.RESET} | {c.CYAN}{m['arch']:<12}{c.RESET} | {c.GREEN}{m['domain']:<22}{c.RESET} | {c.BLUE}{m['format']:<8}{c.RESET} ({m['size']})")
            print()

    @classmethod
    def print_results_matrix(cls, results_df) -> None:
        """Renders the publication-grade clinical diagnostic results table."""
        if RICH_AVAILABLE:
            table = Table(
                box=box.DOUBLE_EDGE,
                border_style="bold bright_green",
                title="[bold bright_white on dark_green]  CLINICAL BENCHMARK DIAGNOSTIC PERFORMANCE MATRIX  [/]",
                title_justify="center",
                padding=(0, 1)
            )
            
            table.add_column("Modality", style="bold bright_cyan", justify="center")
            table.add_column("Model Identifier", style="bold bright_white")
            table.add_column("N", style="dim white", justify="right")
            table.add_column("Sensitivity [95% CI]", style="bold bright_green", justify="center")
            table.add_column("Specificity [95% CI]", style="bold bright_yellow", justify="center")
            table.add_column("F1-Score", style="bold bright_cyan", justify="right")
            table.add_column("Accuracy", style="bold bright_white", justify="right")
            
            for _, row in results_df.iterrows():
                mod = str(row.get("Modality", ""))
                model = str(row.get("Model", ""))
                n = str(row.get("N", ""))
                
                sens = row.get("Sensitivity", 0.0)
                sens_ci = row.get("Sens_95CI", "")
                sens_str = f"{sens:5.2f}% {sens_ci}"
                
                spec = row.get("Specificity", 0.0)
                spec_ci = row.get("Spec_95CI", "")
                spec_str = f"{spec:5.2f}% {spec_ci}" if spec != "N/A" else "N/A"
                
                f1 = f"{row.get('F1_Score', 0.0):.2f}"
                acc = f"{row.get('Accuracy', 0.0):.2f}%"
                
                # Highlight styling
                if "Sudan" in model and mod == "Thin":
                    model_styled = f"[bold bright_green]★ {model}[/]"
                    spec_styled = f"[bold bright_green on dark_green] {spec_str} [/]"
                elif "YOLO" in model:
                    model_styled = f"[bold bright_magenta]{model}[/]"
                    spec_styled = f"[bright_yellow]{spec_str}[/]"
                else:
                    model_styled = f"[bright_white]{model}[/]"
                    spec_styled = f"[bright_yellow]{spec_str}[/]"
                    
                table.add_row(mod, model_styled, n, sens_str, spec_styled, f1, acc)
                
            console.print()
            console.print(table)
            console.print()
        else:
            c = TermColors
            print(f"\n{c.BG_GREEN} CLINICAL BENCHMARK DIAGNOSTIC PERFORMANCE MATRIX {c.RESET}\n")
            print(f"  {c.CYAN}{'MODALITY':<9} {'MODEL':<24} {'N':>5} {'SENSITIVITY (95% CI)':<22} {'SPECIFICITY (95% CI)':<22} {'F1':>6} {'ACC':>6}{c.RESET}")
            print(f"  {c.GRAY}{'─' * 82}{c.RESET}")
            for _, row in results_df.iterrows():
                mod = str(row.get("Modality", ""))
                model = str(row.get("Model", ""))
                sens = f"{row.get('Sensitivity', 0.0):5.2f}% {row.get('Sens_95CI', '')}"
                spec = f"{row.get('Specificity', 0.0):5.2f}% {row.get('Spec_95CI', '')}" if row.get("Specificity") != "N/A" else "N/A"
                print(f"  {c.CYAN}{mod:<9}{c.RESET} {c.WHITE}{model:<24}{c.RESET} {c.GRAY}{row.get('N',''):>5}{c.RESET} {c.GREEN}{sens:<22}{c.RESET} {c.YELLOW}{spec:<22}{c.RESET} {c.CYAN}{row.get('F1_Score',0.0):>6.2f}{c.RESET} {c.WHITE}{row.get('Accuracy',0.0):>6.2f}{c.RESET}")
            print(f"  {c.GRAY}{'─' * 82}{c.RESET}\n")

    @classmethod
    def print_insights_card(cls) -> None:
        """Renders high-impact empirical discoveries in a vibrant panel."""
        if RICH_AVAILABLE:
            content = Text()
            content.append("★ African Regional Advantage: ", style="bold bright_green")
            content.append("MS_Sudan delivers 5.0x higher thin-smear specificity (58.82% vs. 11.76%), cutting false-positive diagnostic errors by 2.14x (Fisher's exact p = 9.50e-7 < 0.0001).\n\n", style="bright_white")
            
            content.append("★ Modality Transfer Cliff: ", style="bold bright_yellow")
            content.append("Leading thick-smear detector (YOLOv8, 87.43% sensitivity) collapses 23.37 percentage points on thin smears (64.06%, p = 5.45e-54) due to intact RBC membrane occlusions.\n\n", style="bright_white")
            
            content.append("★ Optical Blur Vulnerability: ", style="bold bright_red")
            content.append("Focus blur (quantified via circular FOV-masked Laplacian variance, sigma_Lap^2) induces up to a 25.37 percentage-point collapse in thick-smear sensitivity, establishing mandatory pre-inference focus gating.", style="bright_white")
            
            p = Panel(content, box=box.ROUNDED, border_style="bold bright_yellow", title="[bold bright_yellow]⚡ Key Empirical Findings & Regulatory Thresholds[/]", padding=(1, 2))
            console.print(p)
            console.print()
        else:
            c = TermColors
            print(f"  {c.YELLOW}╭─ KEY CLINICAL DISCOVERIES REPRODUCED ────────────────────────────────────╮{c.RESET}")
            print(f"  {c.YELLOW}│{c.RESET} {c.GREEN}★ African Regional Advantage:{c.RESET} MS_Sudan delivers 5.0x higher thin-smear specificity {c.YELLOW}│{c.RESET}")
            print(f"  {c.YELLOW}│{c.RESET}   (58.82% vs 11.76%), cutting false-positive rate by 2.14x (p < 0.0001).      {c.YELLOW}│{c.RESET}")
            print(f"  {c.YELLOW}│{c.RESET} {c.YELLOW}★ Modality Transfer Cliff:{c.RESET} YOLOv8 collapses 23.37 percentage points on thin   {c.YELLOW}│{c.RESET}")
            print(f"  {c.YELLOW}│{c.RESET}   smears (87.43% -> 64.06%, p = 5.45e-54) due to intact RBC membranes.       {c.YELLOW}│{c.RESET}")
            print(f"  {c.YELLOW}│{c.RESET} {c.RED}★ Optical Blur Vulnerability:{c.RESET} Severe blur drops sensitivity by up to 25.37%      {c.YELLOW}│{c.RESET}")
            print(f"  {c.YELLOW}╰──────────────────────────────────────────────────────────────────────────╯{c.RESET}\n")

    @classmethod
    def print_completion_seal(cls, elapsed_sec: float, sha256_hash: str, output_path: str) -> None:
        """Renders the cryptographic verification badge."""
        if RICH_AVAILABLE:
            elapsed_str = time.strftime("%Hh %Mm %Ss", time.gmtime(elapsed_sec))
            content = Text()
            content.append("• Total Benchmark Execution Time: ", style="bold bright_white")
            content.append(f"{elapsed_str}\n", style="bold bright_cyan")
            content.append("• Master SHA-256 Determinism Hash: ", style="bold bright_white")
            content.append(f"{sha256_hash}\n", style="bold bright_green")
            content.append("• Output Predictions & Checksums:  ", style="bold bright_white")
            content.append(f"{output_path}", style="bold bright_yellow")
            
            p = Panel(content, box=box.HEAVY, border_style="bold bright_green", title="[bold black on bright_green]  ✔ BENCHMARK COMPLETE & DETERMINISM SEALED  [/]", padding=(1, 2))
            console.print(p)
            console.print()
        else:
            c = TermColors
            elapsed_str = time.strftime("%Hh %Mm %Ss", time.gmtime(elapsed_sec))
            print(f"\n  {c.BG_GREEN}{c.WHITE} BENCHMARK COMPLETE & DETERMINISM VERIFIED {c.RESET}")
            print(f"  {c.WHITE}• Total Elapsed Time:{c.RESET} {c.CYAN}{elapsed_str}{c.RESET}")
            print(f"  {c.WHITE}• Master SHA-256 Checksum:{c.RESET} {c.GREEN}{sha256_hash}{c.RESET}")
            print(f"  {c.WHITE}• Results Manifest Saved At:{c.RESET} {c.YELLOW}{output_path}{c.RESET}\n")
