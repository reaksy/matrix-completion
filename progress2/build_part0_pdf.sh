#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

/usr/local/texlive/2026/bin/universal-darwin/pdflatex -interaction=nonstopmode -file-line-error part0_initial_experiments.tex
/usr/local/texlive/2026/bin/universal-darwin/pdflatex -interaction=nonstopmode -file-line-error part0_initial_experiments.tex
