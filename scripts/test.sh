#!/usr/bin/env bash
set -euo pipefail
python experiments/test.py --config configs/default.yaml --data "${1:?usage: scripts/test.sh DATA.npz}"

