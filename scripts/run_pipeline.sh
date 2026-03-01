#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

CONFIG="${1:-configs/default.yaml}"

cd "$PROJECT_ROOT"
python -m saegenrec.dataset process "$CONFIG" "${@:2}"
