#!/usr/bin/env bash
# gate before a commit: SQTT tests, ruff and mypy on the encoder; any failure exits non-zero
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
out=$(DEV=MOCK+AMD timeout 900 python -m pytest -q test/amd/test_sqtt_timing.py test/amd/test_sqtt_heldout.py test/amd/test_sqtt_encoder.py 2>&1 | tail -1)
echo "$out"; [[ $out == *passed* && $out != *failed* ]] || { echo "FAIL pytest: $out"; exit 1; }
timeout 120 uvx ruff@0.16.4 check test/mockgpu/amd/sqtt_enc.py test/amd/test_sqtt_heldout.py || { echo "FAIL ruff"; exit 1; }
timeout 300 uvx mypy@2.3.1 test/mockgpu/amd/sqtt_enc.py > /dev/null || { echo "FAIL mypy"; exit 1; }
echo "check OK"
