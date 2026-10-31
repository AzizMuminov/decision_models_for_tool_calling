#!/usr/bin/env bash
# BFCL v4 data ships inside the pinned bfcl-eval wheel. When2Call is fetched from NVIDIA's repo.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
if [ ! -d data/When2Call ]; then
  git clone --depth 1 https://github.com/NVIDIA/When2Call data/When2Call
fi
git -C data/When2Call rev-parse HEAD > data/When2Call.commit
wc -l data/When2Call/data/test/when2call_test_mcq.jsonl
