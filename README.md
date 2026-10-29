# Ad click prediction: Sprint 4 (baseline model + experimentation)

Predict P(click) per ad impression from user, device, time, placement, category and history, so ad spend goes to impressions likely to be clicked.
Tabular binary classification, 1,000 rows, **20 clicks (2%)**.

## 1. Result in 5 lines
- Chosen model: **random forest** (selected on train-only CV, see §6). Trivial baseline = constant prior.
- CV (train, 5x5 repeated stratified): PR-AUC **0.181** vs floor 0.020 (9x); ROC-AUC 0.928; log-loss 0.069 vs floor 0.098; Brier 0.0176 vs 0.0196.
- Held-out test, touched once: PR-AUC **0.120** vs floor 0.020 (bootstrap 95% CI 0.03 to 0.37); ROC-AUC 0.815 (CI 0.59 to 0.97); log-loss 0.084 vs 0.098; Brier 0.0189 vs 0.0196.
- **Test has only 5 clicks. Treat the test numbers as "clearly above chance", not as a precise estimate.**
- Headline caveat: the data contradicts the brief in three places (§3), and with 15 train clicks the *choice among models* is not statistically stable (§3, C4).

## 2. How to run (from the repository root)
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py                                   # data: --data PATH | $AD_CLICK_DATA | ./data/*.csv | ./*.csv
python main.py --stage predict --input new.csv   # scores new rows with the saved artifacts (no refit)
```
Place the CSV at `data/synthetic_data_87.csv` (or pass `--data`). Outputs (created at run time, ~15 s): `outputs/reports/*.json|csv`, `outputs/models/model.joblib`, `outputs/run.log`.
`eda.ipynb` imports `src/models.py`, so EDA and the pipeline use the same loading/contract code.
Per the Sprint 4 folder structure the repo contains only `README.md, eda.ipynb, requirements.txt, main.py, src/models.py`; all logic is in `src/models.py` (sections: config, contract, features, models, metrics, CV, orchestration, inference).

## 3. Caveats (read before trusting any number)
| # | Caveat | Status |
|---|---|---|
| C1 | `time_of_day` is 0-99.6 (537 rows > 23), not an hour. Kept as plain numeric, unit unknown; the brief's "mobile evenings" interaction **cannot be tested**. | confirmed by user |
| C2 | `previous_clicks_on_similar_ads` runs **opposite to the brief**: 17 of 20 clicks sit in the lowest quintile; none above 0.25. It is also fractional (995/1000), so it is scaled upstream or not a raw count. Kept as-is; models learn the direction. | confirmed by user |
| C3 | Placement levels are Top-of-page / In-content / Sidebar / Footer (brief: top / inline / sidebar); no "unknown" gender; 11 rows have age < 13. Kept and flagged, not dropped. | default, **UNCONFIRMED** |
| C4 | 15 train clicks: re-running the selection with other split seeds picked random forest 3 of 5 times (incl. the final seed) and logistic regression or its balanced variant otherwise. The model *family* ranking (trees > linear) held in 5/5; the single winner is not stable. | measured |
| C5 | Threshold rule (max F1 on train OOF, flagging at least 5% of impressions) is a **default**: costs and serving capacity were not given. Use the lift table (§7) to set the real cut-off. | default, **UNCONFIRMED** |
| C6 | Assumed `previous_clicks...` counts only clicks **before** the impression (data cannot verify). If it includes the current impression this is leakage and results are overstated. | assumption |
| C7 | Dataset looks synthetic; no timestamps or user ids, so the split is stratified random. Real deployment needs a temporal split. | note |

## 4. Success criteria (set before results) vs achieved
| Criterion | Target | Achieved |
|---|---|---|
| CV PR-AUC beats no-skill floor | >= 2x floor (0.04) | 0.181 (9x) pass |
| Probabilities no worse than constant prior (log-loss, Brier) | below floor | CV pass; test pass but Brier margin is tiny (0.0189 vs 0.0196) |
| Test touched once, floors beside every metric | yes | yes |
| Pipeline runs from repo root with only the Sprint 4 files | yes | yes (fresh-copy run, deterministic) |

## 5. Pipeline flow
1. Load, **validate contract** (columns, numeric types, previous clicks >= 0, binary target). Deviations from the brief are counted in `data_quality.json`, not silently fixed. Drop exact duplicates (0).
2. **Top-level stratified 75/25 train/test split, before anything is fit** (seed 42). Guard: no identical feature row in both sets. Stateless feature `log1p(previous_clicks)` is added before the split (row-wise, learns nothing).
3. Repeated stratified 5-fold x 5 CV **on train only**. Preprocessing (median/most-frequent imputer, scaler, one-hot with unseen -> zeros) is inside each `Pipeline`, so it is refit per fold.
4. Selection: gate on PR-AUC > floor, then the simplest model within 1 std of the best PR-AUC.
5. Threshold from train out-of-fold probabilities of the chosen model.
6. Final refit on train with a **different seed** (2026), one test evaluation (chosen model + trivial baseline only; no per-candidate test metrics), bootstrap CI.
7. Save one artifact set (`model.joblib`: pipeline, threshold, train prior, feature list, trained category levels, versions). `predict` loads it, validates, logs unseen levels/missing cells, and never calls `.fit`.

## 6. Models and EDA decisions
EDA findings F1-F9 with their decisions are in `eda.ipynb` (executed). Summary: 2% positives -> PR-AUC primary; previous clicks is the strongest signal and non-linear; category cells hold 0-12 clicks each, so no per-level claims.

| Model | Role | CV PR-AUC | CV log-loss | Verdict |
|---|---|---|---|---|
| prior_baseline | trivial floor | 0.020 | 0.098 | floor |
| logreg (C=0.3) | simple baseline | 0.090 | 0.083 | beats floor; linear cannot capture the near-zero effect |
| logreg_balanced | imbalance-remedy experiment | 0.127 | **0.338** | PR-AUC up, but probabilities 3.4x worse than floor: rejected for pricing; class weights not adopted |
| random_forest (min leaf 20) | **chosen**; contrast model | **0.181** | 0.069 | best on every metric |
| hist_gradient_boosting (depth 3) | complex | 0.127 | 0.078 | no gain over the forest |

- **Contrast (pre-registered from F7):** "trees beat the linear model". Held: best tree 0.181 vs 0.090 (+0.091); also true under 4 other split seeds.
- Parsimony rule: only the forest is within 1 std of the best, so it is chosen even though a simpler model exists. Its CV std (0.029) is across repeats of the same data and **understates** true uncertainty (C4).
- No hyper-parameter search: with 15 positives a search would fit noise. Settings are fixed and strongly regularised.

## 7. Decision view: what to do with the scores
Capacity and cost per impression were not given, so the table shows the trade-off instead of one answer.
| Serve to top | Train OOF: share of clicks caught | Test: share of clicks caught (5 clicks) |
|---|---|---|
| 5% | 47% (lift 9.2x) | 40% (2/5) |
| 10% | 73% | 40% |
| 20% | 93% (lift 4.7x) | 80% (4/5) |
| 30% | 100% | 80% |

Default threshold 0.107 (UNCONFIRMED): on test flags 16/250 impressions (6.4%), precision 12.5% (flag-all floor 2%), recall 40%.
What the model does **not** tell you: *why* a user clicks (observational, no causal claim); any per-category or per-hour effect (too few clicks); behaviour on a different population.
Monitoring/retrain: track weekly CTR and the share of flagged impressions; retrain when CTR drifts outside 0.5-5% or flagged-share moves by more than half; re-fit a threshold once a real cost or capacity exists.

## 8. Feedback on the previous submission -> what changed
| Feedback | Done |
|---|---|
| Runnable from the repo root | `python main.py` from root; every module referenced exists; paths resolve from `main.py`'s own location; test in a fresh copy of the 5 files + CSV (**you must still run the clean-venv steps in §2**: no network here to install) |
| Top-level split explicit and first | §5 step 2, `run_pipeline()`; asserts no shared feature rows |
| Calibration-aware metric | log-loss **and** Brier next to PR-AUC / ROC-AUC, each with its floor |
| Persist reports as files | `outputs/reports/`: data_quality, split, cv_results, model_selection (incl. threshold), test_results, lift tables, run_summary |
| Keep the modular design, connect all pieces | one module, clearly sectioned, because Sprint 4 allows only `src/models.py` |

## 9. Skipped on purpose (and why)
Hyper-parameter search, VIF/formal tests, subgroup evaluation (5 test clicks make any subgroup meaningless), serving-parity test (predict uses the same pipeline object), unit-test folder (not in the Sprint 4 structure), calibration upkeep (no drift data).

## 10. Self-review (8 checks, on the actual code/output)
1. Learned-from-data steps fit on train only: imputer/scaler/encoder are inside the CV `Pipeline`; threshold from train OOF. Pass.
2. Target absent from features: `FEATURES` excludes `clicked_ad`. Pass.
3. Label recipe: label is observed, not engineered. N/A.
4. Floor beside every metric: yes (§1, §6, `test_results.json`).
5. Test touched once: one `predict_proba` on test, chosen model only. Pass.
6. Split matches use: stratified random only because no time/user column (C7).
7. Each output tier maps to an action with volume vs capacity: §7, capacity unknown (C5).
8. README and code agree: numbers copied from `outputs/reports/`.
**Caught and changed during review:** the first "balanced" logistic looked better on PR-AUC; its log-loss showed the probabilities were distorted, so it is reported as rejected rather than chosen.
