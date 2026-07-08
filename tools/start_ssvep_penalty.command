#!/bin/zsh
set -e

cd "$(dirname "$0")/.."
python tools/run_ssvep_penalty.py --config config.yaml --host 127.0.0.1 --port 5005 --open-game
