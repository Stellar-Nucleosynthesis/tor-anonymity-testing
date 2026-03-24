# Tor Deanonymization Research Framework

A research framework for evaluating traffic correlation attacks against Tor,
built on top of the [Shadow](https://shadow.github.io/) discrete-event network
simulator and [tornettools](https://github.com/shadow/tornettools).

The framework runs end-to-end: from downloading relay descriptors and generating
a synthetic Tor network, through simulating traffic, to running correlation
attacks and rendering evaluation reports.

> **Research use only.** The modified Tor binary emits control-port events that
> expose internal circuit state. It must never be deployed on the live Tor network.

---

## Overview

```
download descriptors
        ↓
simulate (Shadow + tornettools + modified Tor)
        ↓
analyze (traffic correlation attack)
        ↓
report (metrics + plots + JSON)
```

Each step is a self-contained script. Steps can be run
individually or chained together.

---

## Requirements

- Python ≥ 3.10
- [Shadow](https://shadow.github.io/) simulator
- [tornettools](https://github.com/shadow/tornettools)
- Modified Tor binary (see [Tor Fork](https://github.com/Stellar-Nucleosynthesis/tor))
- Python packages: `numpy`, `scipy`, `matplotlib`

---

## Quick Start

```bash
# Step 1 — download relay descriptors for a given month
python download.py \
    --month 2024-09 \
    --output-dir data/

# Step 2 — generate and run simulations across 5 seeds
python simulate.py \
    --data-dir data/ \
    --output-dir runs/ \
    --seeds 5

# Step 3 — run the guard-exit attack and generate a report
python analyze.py \
    --sim-dirs runs/seed_* \
    --output-dir reports/baseline \
    guard-exit --guard-fraction 0.10 --exit-fraction 0.10
```

---

## Attacks

### Guard-Exit Correlation (`guard-exit`)

Simulates an adversary who controls a fraction of guard and exit relays. For
each compromised guard relay, the attack attempts to identify the matching exit
relay by correlating their per-circuit traffic timeseries.

**How it works:**

1. Parse all relay OnionTrace logs to build traffic profiles (`CIRC_BW OR_STAT`
   events from the modified Tor binary).
2. Build ground truth by joining `RESEARCH_ID_CHOSEN` events (client logs) with
   `RESEARCH_ID_UPDATED` events (relay logs) via the globally unique
   `research_id` assigned to each circuit.
3. For each (guard profile, exit profile) candidate pair, compute a
   cross-correlation score between their traffic timeseries.
4. The exit with the highest score above `--threshold` is nominated as the
   match for that guard observation.

---

## Metrics

All metrics treat deanonymization as a **top-1 identification problem**.

| Metric                 | Description                                                            |
|------------------------|------------------------------------------------------------------------|
| `success_rate`         | Fraction of all observed circuits correctly identified (unconditional) |
| `coverage`             | Fraction of circuits for which a prediction was made                   |
| `conditional_accuracy` | Fraction of predictions that were correct                              |
| `abstention_rate`      | `1 - coverage`; fraction of circuits where the attack abstained        |
| `score_separation`     | `mean(score                                                            | correct) − mean(score | incorrect)`, measures discriminability |
| `mean_score_correct`   | Mean correlation score on correct identifications                      |
| `mean_score_incorrect` | Mean correlation score on incorrect identifications                    |

---

## Reports and Plots

Running `analyze.py` produces a `scenario_summary.json` and a set of plots in
the output directory.

### Plot selection

Pass `--plots` with one or more names, or `--plots all`:

| Name                | Description                                                                                      |
|---------------------|--------------------------------------------------------------------------------------------------|
| `metrics_bar`       | Grouped bar chart: `conditional_accuracy`, `coverage`, `success_rate` per scenario               |
| `score_metrics_bar` | Grouped bar chart: `mean_score_correct`, `mean_score_incorrect`, `score_separation` per scenario |
| `seed_variance`     | Box plots of every key metric across seeds, one file per metric                                  |
| `score_dist`        | Overlaid histograms of correlation scores for correct vs. incorrect identifications              |
| `accuracy_coverage` | Accuracy/coverage trade-off curve with best operating point marked                               |

---

## Tor Fork

See [README-TOR.md](https://github.com/Stellar-Nucleosynthesis/tor/blob/main/README.md) for a full description of the modifications
made to the Tor source, including the `research_id` circuit identifier, the
`RELAY_COMMAND_UPDATE_RESEARCH_ID` cell, and all new control-port events.