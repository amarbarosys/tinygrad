# Held-out SQTT timing: 91.2% of SRAM deltas exact on kernels the model never saw

On 20 random SRAM kernels (`randprobe_v2.py`, seeds 120-139) captured on a 7900 XTX after the timing rules were frozen,
the mockgpu emulator's RDNA3 SQTT timing (`test/mockgpu/amd/sqtt_enc.py`) predicts 1712 of 1878 SRAM instruction deltas
to the cycle (`score.py --sram`: the s_load, its wait, the instruction after it, the wave end and fetch stalls are DRAM
and not counted; with them 1748/2058). The 41 captures in `captures/` (seed1-11, mul1-10, v2_100-119) were all used
to fit and score 94.6%, 91.9% and 92.7%; the fresh 91.2% is the number to quote.

`randprobe.py` builds the kernel from a seed: VALU, SALU, LDS store and load, `s_waitcnt`, `s_nop` and barriers,
on 128 threads; `randprobe_v2.py` adds the 32-bit integer multiplies (VALUB_4 on gfx1100). `captures/` holds one real
trace per seed (gfx1100, `profile_standard` clocks, SQTT on WGP 3). `score.py` replays each kernel in the emulator,
trying both hidden dispatcher states (which SIMD gets wave 0, which SIMD wins the first LDS tie), and compares every
delta; `misses.py` prints every scored wave instruction by instruction.

```
$ PYTHONPATH=. python3 extra/sqtt/heldout/score.py --sram .work/sqtt/fresh2/seed*/profile_randprobe_v2.pkl
  ... 20 seeds, 83.0% to 98.7% each, mean abs error 0.03 to 0.46 cycles
  total: 1712/1878 exact (91.2%) SRAM only
$ PYTHONPATH=. python3 extra/sqtt/heldout/score.py --sram      # the 41 fitted captures
  total: 3505/3768 exact (93.0%) SRAM only
```

## What is still wrong on the fresh set

| packet | misses / scored | cause |
|---|---|---|
| LDS_WR_2, LDS_RD | 68 / 236 | issue +-1..4 after VALUB_4 -> VALU chains and after SALU, the multiply's pipe against three other waves |
| IMMEDIATE | 33 / 300 | LDS return order between SIMDs, barrier release +-1 |
| VALUINST | 24 / 598 | first VALU after a release or an LDS store |
| VALUB_4 | 20 / 216 | issue after a barrier or an LDS store |
| SGMEM_WR_2 | 11 / 38 | global store after VALUB_4 -> VALU -> free wait |
| BARRIER, SALU | 10 / 490 | arrival of the untraced SIMD's waves |

## The tests hold every capture at its score

```
DEV=MOCK+AMD python -m pytest test/amd/test_sqtt_heldout.py test/amd/test_sqtt_timing.py
```

`test/amd/test_sqtt_heldout.py` fails if any of the 41 captures loses an exact delta, and moves every measured constant
by one cycle either way to check that a named capture loses deltas (VALUB_4 exec and SALU hold, barrier window, busy and
serial counting, LDS-to-VALU offsets, LDS after VALU exec, multiply pipe occupancy).

## Capturing a new seed

Needs a 7900 XTX and [wavewait](https://github.com/amarbaro/wavewait)'s `tools/sqtt-capture`, with its
`wgp-probe.patch` applied to this checkout:

```
git apply .work/sqtt/wgp-probe.patch          # trace WGP 3; WGP 0 and 1 give waves without instruction tokens on this card
SQTT_SEED=140 SQTT_WGP_PROBE=3 TINYGRAD=$PWD sqtt-capture $PWD/extra/sqtt/heldout/randprobe_v2.py OUT
PYTHONPATH=. python3 extra/sqtt/heldout/score.py --sram OUT/profile_randprobe_v2.pkl
```
