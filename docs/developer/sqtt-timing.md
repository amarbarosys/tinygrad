# SQTT timing model

The frozen SQTT timing model reaches 74.5% exactness on fresh SRAM traces, below the 98% lane gate. This page records the measured boundary and the rules that produced it.

## Current result

- Baseline seeds 1-11: 815/1108 exact, 73.6%.
- Final fit seeds 1-11: 863/1108 exact, 77.9%.
- Final mulheld fit: 714/986 exact, 72.4%.
- Fresh seeds 100-119: 1448/1944 SRAM exact, 74.5%.
- The fresh set was captured once after the model was frozen.

SRAM scoring excludes `SMEM_RD` and `WAVEEND`; those DRAM classes are out of scope for this lane.

## Timing rules

`LDS_TO_VALU_LOAD = 4` and `LDS_TO_VALU_STORE = 5` schedule a direct VALU successor from the LDS service start. The emulator records `last_lds_service` per wave. Changing either offset by plus or minus one cycle lowers a fit seed score.

The fresh multiply probe uses random `vvvvssllwnb` bodies and includes `v_mul_lo_u32`, `v_mul_hi_u32`, and `v_mul_hi_i32`. WGP3 is required on this machine because WGP0 and WGP1 produced waves without instruction tokens.

## Checks

- SQTT timing and heldout tests pass: 3 tests, 18 subtests.
- LDS mutant tests pass: 2 tests, 15 subtests.
- AMD suite: 1298 passed, 72 skipped, 1023 subtests passed, with the known unrelated `test_cfg_branch_diamond` failure.
- `mypy` and `ruff` were unavailable in the lane environment.

Detailed capture and packet-class results: `.work/reports/sqtt-timing-v2.md`.
