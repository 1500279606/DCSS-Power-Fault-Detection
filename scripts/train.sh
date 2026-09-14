#!/usr/bin/env bash
set -euo pipefail
python experiments/train.py --config configs/default.yaml --data "${1:?usage: scripts/train.sh DATA.npz}"

