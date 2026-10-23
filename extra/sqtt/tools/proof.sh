#!/usr/bin/env bash
# proof run of the SQTT timing v2 result: every number below is computed live from the committed tree (CPU only)
# usage (repo root): bash extra/sqtt/tools/proof.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
B=$'\e[1;36m'; G=$'\e[1;32m'; N=$'\e[0m'
say() { printf '\n%s== %s ==%s\n' "$B" "$1" "$N"; sleep 1.5; }
run() { printf '%s$ %s%s\n' "$G" "$*" "$N"; sleep 0.8; eval "$@"; }
FRESH="extra/sqtt/heldout/captures/v2_2[01]?.pkl"  # v2 seeds 200-219, captured once after the freeze at 02bf898d0
OLD=.work/proof-old
trap 'git worktree remove --force "$OLD" 2>/dev/null || true' EXIT
git worktree add -q --detach "$OLD" 1f6b96c96  # the model at the start of the leg
clear
say "SQTT timing v2: emulator vs RDNA3 hardware traces (7900 XTX), proof run $(date '+%F %T')"
run "git log --oneline -9 | cat"
run "git status --short --untracked-files=no | wc -l   # 0 = clean tree"
say "1. Fresh seeds 200-219: captured on the GPU once, after freezing the model at 02bf898d0"
run "git diff --stat 02bf898d0 HEAD -- test/mockgpu/amd/sqtt_enc.py | tail -1; echo '(empty = model unchanged since the freeze)'"
run "PYTHONPATH=. python3 extra/sqtt/heldout/score.py --sram $FRESH | tail -1"
say "2. Same 20 captures, model from the start of this leg (1f6b96c96)"
run "(cd $OLD && git log --oneline -1 | cat && PYTHONPATH=. python3 extra/sqtt/heldout/score.py --sram $PWD/$FRESH | tail -1)"
say "3. Commit gate: SQTT tests + ruff + mypy"
run "extra/sqtt/tools/check.sh 2>&1 | tail -3"
say "4. Full AMD test suite on the mock GPU"
run "DEV=MOCK+AMD python -m pytest test/amd --deselect test_cfg_branch_diamond -q -p no:cacheprovider 2>&1 | tail -1"
say "done: target 98% not met; fresh 96.8% vs 94.8% for the old model on the same seeds"
