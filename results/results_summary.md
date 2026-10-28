# Experimental Results Summary

User study run with an online questionnaire showing pre-recorded phosphene videos rendered by the pipeline in two modes:

- **RGB-only mode** (baseline): phosphene brightness driven by the grayscale camera image only.
- **Hybrid mode** (proposed): depth-modulated texture + LiDAR structural edges.

Raw (anonymized) answers: [`data/survey_responses.csv`](../data/survey_responses.csv) · Analysis: [`analysis/form_analysis.py`](../analysis/form_analysis.py)

## 1. Sample

- **89 participants**, predominantly young adults (≈75% aged 18–25), all untrained to phosphene vision.

## 2. Dynamic tracking (Test D) — key finding

| Mode | Preference |
|---|---|
| **Hybrid (depth edges)** | **79.8%** |
| RGB-only | 19.1% |
| Neutral | 1.1% |

The hybrid pipeline suppresses temporal flickering and "ghosting" artifacts, giving a stable signal for real-time interaction: in dynamic scenes, noise suppression matters more than raw detail.

![Movement](Graph_1_Movement_Final.png)

## 3. Static scene perception (Test B)

| Scenario | Hybrid | RGB-only |
|---|---|---|
| Chair (geometric clarity) | **63%** | 31% |
| Bed (obstacle detection) | **51%** | 35% |
| Door (high-contrast detail) | 27% | **64%** |

Trade-off between geometric clarity and detail preservation: aggressive denoising in hybrid mode can remove subtle texture cues in low-contrast static scenes. A natural next step is an adaptive filter that relaxes suppression when the user is stationary.

![Static](Graph_2_Static_Final.png)

## 4. Semantic efficacy (Test A)

| Target | Detection (shape) | Identification (name) |
|---|---|---|
| Laptop | 71% | 40.5% |
| Bottle | 57% | 51% |

The pulsating semantic highlight works well as an **alert mechanism**; naming the object remains hard at 32×32 resolution, yet ~40–50% success is notable for untrained subjects.

![Recognition](Graph_3_Recognition_Final.png)

## 5. User experience

- **Mental fatigue:** 3.45 / 5 — interpreting phosphene vision is cognitively demanding, motivating cleaner signals.
- **Perceived safety:** 2.39 / 5 — a realistic level of caution for a first-time experience.

![UX](Graph_4_UX_Final.png)

## Verdict

The depth-aware pipeline clearly outperforms RGB-only processing in dynamic tasks (≈80% preference) and supports basic object detection (≈70%). Static low-contrast detail remains a challenge, but the trade-off favours the proposed method for navigational safety and interaction stability.
