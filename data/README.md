# Dataset

**File:** `student-mat.csv` (semicolon separated, 395 rows x 33 columns)

**Source:** UCI Machine Learning Repository, *Student Performance* dataset (mathematics course).
https://archive.ics.uci.edu/dataset/320/student+performance

**Citation:** P. Cortez and A. Silva. *Using Data Mining to Predict Secondary School Student Performance.*
In A. Brito and J. Teixeira (Eds.), Proceedings of 5th FUture BUsiness TEChnology Conference (FUBUTEC 2008), pp. 5-12, Porto, Portugal, April 2008.

**Licence:** Creative Commons Attribution 4.0 (CC BY 4.0), as stated on the UCI page.

**What it contains:** two Portuguese secondary schools; one row per student; background
(age, family, parents' education), behaviour (study time, going out, alcohol, absences, past
failures) and three grades: `G1` (period 1), `G2` (period 2) and `G3` (final, the prediction target), all on a 0-20 scale.

**Audit result (from `partA/results/summary.json`):** 0 missing values, 0 duplicate rows,
38 students with G3 = 0.
