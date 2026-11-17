# Demo and Viva Walkthrough (about 10 minutes)

The project is judged on how well you can **explain** it, not on model complexity. Learn the *story*, not the code.

## The story in 30 seconds (everyone should be able to say this)

> "We asked: how early can we predict a student's final grade? With only background information, we can barely beat guessing the average (R² 0.24). Once we know the second-period grade G2 it becomes easy (R² 0.81), but even copying G2 is almost as good. We also found that every student with a final grade of 0 has 0 recorded absences, so that feature is partly a dropout marker. For Part B we built a game, Overfit Lab, where you discover overfitting yourself: training error always improves, but only held-out data tells the truth."

## Suggested split of speaking (3 people)

| Who | Part | Time |
|---|---|---|
| **Ajim (EC24I2021)** | Problem, dataset, cleaning and EDA (Part A, steps 1-2) | 3 min |
| **Jashwanth (EC24I1019)** | Modelling, evaluation, error analysis, limitations (Part A, steps 3-4) | 4 min |
| **Charan Teja (EC24I2015)** | Live demo of the game and playtest evidence (Part B) | 3 min |

Everyone must still be able to answer questions on every part.

## Live script

### 1. Open (Ajim, 30 s)
Show the README. "One repo, two parts. Part A is `partA/student_grade_prediction.py`, Part B is `partB/index.html`."

### 2. Problem and data (Ajim, 1 min)
- Question: *can a school warn a student before the exam?*
- Data: UCI Student Performance, 395 maths students, 33 columns, real and public.
- Goal: beat the "predict the average" baseline, and measure what G1 and G2 add.

### 3. Cleaning and EDA (Ajim, 2 min) - open `partA/figures/`
- **fig01:** "Grades are bell-shaped, but 38 students scored zero."
- **fig02:** "Past failures are the clearest driver: average 11.3 with no failures, 5.7 with three."
- Reveal the quirk: "All 38 zero-score students have 0 absences. So 0 absences does not mean perfect attendance, it means probably dropped out. We kept them, because deleting the hardest cases would hide the real problem."
- Mention: no missing values or duplicates (we checked), log transform for skewed absences.

### 4. Modelling (Jashwanth, 2 min) - open `fig04_model_comparison.png`
- "Two scenarios: A has no grades, B has G1 and G2. Same split, same models."
- "Baselines first: predict the mean, and copy G2. A model must beat these."
- "Three models: ridge, random forest, gradient boosting, tuned with 5-fold cross-validation on the training set only. The 79-student test set was used once at the end."
- "No leakage: scaling and encoding are inside the pipeline, so they are fitted on training folds only."

### 5. Results and honesty (Jashwanth, 2 min)
- A: RMSE 4.55 &rarr; 3.95, R² 0.24. B: RMSE 1.98, R² 0.81.
- "But copy-G2 scores 2.07. The difference is smaller than the cross-validation spread, so we do not claim the forest is truly better."
- Show `fig06`: "In B, G2 dominates. In A, absences and failures lead, but absences is partly the dropout marker. Without it, R² falls to 0.15."
- Show `fig07`: "Most error comes from 5 zero-score students: RMSE 5.3 for them versus 1.5 for everyone else."
- Limitations: small data, sensitive attributes, correlation is not causation.

### 6. The game (Charan Teja, 3 min) - open `partB/index.html`
1. **Level 1.** Drag the slider to degree 12. "Training error is tiny, and the curve fits every dot." Lock in. "Test error is huge, the score is about 6. The model memorised noise. Training error can never warn us." Point at the chart: blue line only falls, red line turns up.
2. **Level 2.** "Now we hold some data back (orange squares). Validation error behaves like the hidden test." Pick the degree where the orange error is lowest, lock in, score near 100.
3. **Level 3.** "Model fixed at degree 9. How much data do we need?" Run 15 points: fail. Run 35 points: pass. "More data tames a flexible model."
4. **Level 4.** "Same idea on the real students from Part A: predict G3 from G1 using only 20 students. The best model is simple, about 2.8 grade points of error versus about 4.5 for just guessing the average."
5. Press **New data**: "Answers are not hard-coded. The curve is fitted live in the browser."
6. Show `docs/PLAYTEST.md`: "Here are classmates who tried it and what they said."

### 7. Close (anyone, 20 s)
"The same lesson links both parts: always judge a model on data it has not seen, and report honestly when a simple baseline is almost as good."

---

## Likely viva questions and short answers

**Why regression and not classification?** G3 is a number from 0 to 20. We also derived pass/fail (G3 &ge; 10) from the predictions to give a practical at-risk view.

**What is RMSE?** Typical size of the prediction error in grade points, punishing large misses more. RMSE 1.98 means predictions are typically about 2 marks off. MAE is the average absolute miss.

**What is R²?** The share of variation in grades explained by the model. 0 = no better than the average, 1 = perfect. Ours: 0.24 (A) and 0.81 (B).

**What is data leakage and how did you avoid it?** Information from test data leaking into training, giving falsely good scores. We split once, tuned only with cross-validation on the training set, and put scaling and encoding inside the pipeline so they are learned from training folds only. We chose the final model by CV, not by test score.

**Why is there a baseline?** To know whether the model adds anything. "Copy G2" showed that for scenario B it adds very little.

**Why did you choose random forest if gradient boosting scored slightly better on the test set?** We chose by cross-validation before looking at the test set. Choosing by test score would leak the test set into the decision. The two are statistically indistinguishable anyway.

**Why keep the zero-grade students?** They are real. Removing them would make the model look better but hide the hardest and most important cases.

**Why log-transform absences?** It is heavily skewed (max 75). The log reduces the pull of extreme values without deleting students.

**What is the absences problem?** All 38 students with G3 = 0 have 0 absences, probably because dropouts have no attendance record. The model can exploit that, which is not real "behaviour". We tested the effect: removing absences reduced R² from 0.24 to 0.15.

**Why is the test set only 79 students? Is it reliable?** It is small, so test numbers are noisy. That is why we also show cross-validation means and spreads, and avoid strong claims about small differences.

**Does "going out" cause lower grades?** No. This is correlation, not causation. Other factors (motivation, time, friends) could drive both.

**Is it ethical to predict "at risk" students?** Only to offer support. Sensitive attributes (sex, age, parents' education) can encode unfairness, so a real system would audit or drop them and never penalise students.

**What is overfitting?** A model so flexible it memorises noise in the training data: tiny training error, large error on new data. Underfitting is the opposite: too simple to capture the pattern.

**Bias-variance in one line?** Simple models are biased (too stiff) but stable. Complex models have low bias but high variance (they change wildly with each sample). Best is in between.

**Why does more data help in Level 3?** With more points a flexible curve cannot pass through all of them by chance, so it is forced toward the true pattern. The variance shrinks.

**How does the game compute things?** In `partB/engine.js`: seeded random data, least-squares polynomial fit, RMSE. The test error is computed live after lock-in; the score is compared with the best degree found by trying all of them.

**What would you do with more time?** Model dropout separately, add the Portuguese-course data, repeated cross-validation, and tune the at-risk threshold with teachers.

---

## Checklist before the demo
- [ ] `pip install -r requirements.txt`, then `python partA/student_grade_prediction.py` runs without errors on your laptop (do it once beforehand; it takes 1-3 min)
- [ ] `partB/index.html` opens by double-click in Chrome or Edge
- [ ] At least 3 real classmates have played; `docs/PLAYTEST.md` is filled in with their real names and comments
- [ ] Each of you has read the story in 30 seconds and the Q&A above
