"""
Part A - Can we predict a student's final grade, and how early?

Dataset : UCI "Student Performance" (maths course, 395 students, Cortez & Silva 2008)
Target  : G3 (final grade, 0-20)

Two scenarios are compared on purpose:
  A) EARLY WARNING : only background/behaviour features (no previous grades)
  B) WITH GRADES   : the same features + G1 and G2 (period grades)

Run from the repo root:   python partA/student_grade_prediction.py
Everything (figures, tables, summary.json) is written next to this file.
"""
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

SEED = 42  # one seed for every random step -> results are reproducible
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "student-mat.csv"
HERE = Path(__file__).resolve().parent
FIG, RES = HERE / "figures", HERE / "results"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)
PASS_MARK = 10  # G3 >= 10 counts as "pass"

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})
C_BLUE, C_ORANGE, C_GREY, C_RED = "#2b6cb0", "#dd6b20", "#8a94a6", "#c53030"


# ----------------------------------------------------------------------------
# 1. Load + audit
# ----------------------------------------------------------------------------
def load_and_audit():
    df = pd.read_csv(DATA, sep=";")
    audit = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_cols": int(df.select_dtypes("number").shape[1]),
        "categorical_cols": int(df.select_dtypes("object").shape[1]),
        "g3_zero_students": int((df.G3 == 0).sum()),
        "g3_zero_with_g2_zero": int(((df.G3 == 0) & (df.G2 == 0)).sum()),
        "g3_zero_with_g2_positive": int(((df.G3 == 0) & (df.G2 > 0)).sum()),
        # data quirk: students who scored 0 have no recorded absences (probably dropped out)
        "g3_zero_with_absences_zero": int(((df.G3 == 0) & (df.absences == 0)).sum()),
        "share_g3_zero_when_absences_zero": float((df[df.absences == 0].G3 == 0).mean()),
        "n_absences_zero": int((df.absences == 0).sum()),
        "absences_max": int(df.absences.max()),
        "absences_p95": float(df.absences.quantile(0.95)),
        "absences_above_30": int((df.absences > 30).sum()),
        "pass_rate": float((df.G3 >= PASS_MARK).mean()),
        "g3_mean": float(df.G3.mean()),
        "g3_std": float(df.G3.std()),
        "corr_G2_G3": float(df.G2.corr(df.G3)),
        "corr_G1_G3": float(df.G1.corr(df.G3)),
    }
    return df, audit


# ----------------------------------------------------------------------------
# 2. Feature engineering (each step has a reason)
# ----------------------------------------------------------------------------
def engineer(df):
    d = df.copy()
    # absences is heavily right-skewed (max 75) -> log1p tames outliers without deleting students
    d["log_absences"] = np.log1p(d["absences"])
    # weekday + weekend alcohol merged into one weekly-weighted score (5 weekdays, 2 weekend days)
    d["alcohol"] = (5 * d["Dalc"] + 2 * d["Walc"]) / 7
    # mother's and father's education are strongly correlated -> average them
    d["parent_edu"] = (d["Medu"] + d["Fedu"]) / 2
    # any past failure is the key jump, so give it its own flag
    d["has_failed_before"] = (d["failures"] > 0).astype(int)
    return d.drop(columns=["absences", "Dalc", "Walc", "Medu", "Fedu"])


def split_columns(X):
    cat = X.select_dtypes("object").columns.tolist()
    num = [c for c in X.columns if c not in cat]
    return cat, num


def make_preprocessor(X):
    cat, num = split_columns(X)
    # preprocessing lives INSIDE the pipeline, so inside cross-validation it is fitted
    # on the training folds only -> no data leakage
    return ColumnTransformer([
        ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), cat),
        ("num", StandardScaler(), num),
    ])


# ----------------------------------------------------------------------------
# 3. EDA plots
# ----------------------------------------------------------------------------
def eda_plots(df):
    # Fig 1: target distribution
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.hist(df.G3, bins=np.arange(-0.5, 21.5, 1), color=C_BLUE, edgecolor="white")
    ax.axvline(PASS_MARK - 0.5, color=C_RED, ls="--", lw=1.2)
    ax.text(PASS_MARK - 0.3, ax.get_ylim()[1] * 0.92, "pass mark (10)", color=C_RED, fontsize=9)
    ax.annotate(f"{(df.G3 == 0).sum()} students scored 0\n(dropouts / no-shows)",
                xy=(0, (df.G3 == 0).sum()), xytext=(2.5, (df.G3 == 0).sum() * 0.8),
                arrowprops=dict(arrowstyle="->", color=C_GREY), fontsize=9)
    ax.set(title="Final grade (G3) is bell-shaped, plus a spike of zeros",
           xlabel="G3 (0-20)", ylabel="students")
    fig.tight_layout()
    fig.savefig(FIG / "fig01_target_distribution.png")
    plt.close(fig)

    # Fig 2: four drivers
    fig, axs = plt.subplots(2, 2, figsize=(9, 6.2))
    d = df.copy()
    d["absence_band"] = pd.cut(d.absences, [-1, 0, 4, 10, 100], labels=["0", "1-4", "5-10", "11+"])
    specs = [("failures", "Past failures", None), ("studytime", "Weekly study time (1=<2h ... 4=>10h)", None),
             ("absence_band", "Absences", None), ("goout", "Going out (1=low ... 5=high)", None)]
    for ax, (col, label, _) in zip(axs.ravel(), specs):
        groups = sorted(d[col].dropna().unique(), key=lambda v: str(v) if col == "absence_band" else v)
        if col == "absence_band":
            groups = ["0", "1-4", "5-10", "11+"]
        data = [d.loc[d[col].astype(str) == str(g), "G3"] for g in groups]
        ax.boxplot(data, patch_artist=True,
                   boxprops=dict(facecolor="#bee3f8"), medianprops=dict(color=C_RED))
        ax.set_xticklabels([str(g) for g in groups])  # works on old and new matplotlib
        ax.set(xlabel=label, ylabel="G3")
    axs[0, 0].set_title("Past failures: the clearest drop")
    axs[0, 1].set_title("More study time: small gain")
    axs[1, 0].set_title("Absences: no simple pattern")
    axs[1, 1].set_title("Going out: slightly lower grades")
    fig.suptitle("What do final grades look like across student groups?", y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / "fig02_eda_drivers.png")
    plt.close(fig)

    # Fig 3: strongest correlations with G3 (numeric), Spearman is robust to outliers
    num = df.select_dtypes("number").drop(columns="G3")
    corr = num.apply(lambda s: s.corr(df.G3, method="spearman")).sort_values()
    top = pd.concat([corr.head(6), corr.tail(6)]).drop_duplicates()
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.barh(top.index, top.values, color=[C_RED if v < 0 else C_BLUE for v in top.values])
    ax.set(title="Correlation with final grade (Spearman)", xlabel="correlation with G3")
    fig.tight_layout()
    fig.savefig(FIG / "fig03_correlations.png")
    plt.close(fig)
    return corr


# ----------------------------------------------------------------------------
# 4. Modelling
# ----------------------------------------------------------------------------
def candidate_models(X):
    pre = make_preprocessor(X)
    return {
        "Ridge regression": (Pipeline([("pre", pre), ("m", Ridge())]),
                             {"m__alpha": [0.1, 1, 10, 30, 100, 300]}),
        "Random forest": (Pipeline([("pre", pre), ("m", RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1))]),
                          {"m__max_depth": [3, 5, None], "m__min_samples_leaf": [1, 3, 5, 10]}),
        "Gradient boosting": (Pipeline([("pre", pre), ("m", GradientBoostingRegressor(subsample=0.8, random_state=SEED))]),
                              {"m__n_estimators": [100, 200], "m__learning_rate": [0.03, 0.1], "m__max_depth": [2, 3]}),
    }


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def pass_fail_metrics(y_true, y_pred):
    true_fail, pred_fail = y_true < PASS_MARK, y_pred < PASS_MARK
    acc = float((true_fail == pred_fail).mean())
    tp = int((true_fail & pred_fail).sum())
    recall = tp / max(int(true_fail.sum()), 1)
    precision = tp / max(int(pred_fail.sum()), 1)
    return acc, float(recall), float(precision)


def run_scenario(name, X_tr, X_te, y_tr, y_te):
    cv = KFold(5, shuffle=True, random_state=SEED)
    rows, fitted, preds = [], {}, {}

    # --- baselines -----------------------------------------------------------
    dummy = DummyRegressor(strategy="mean").fit(X_tr, y_tr)
    cv_d = -cross_val_score(DummyRegressor(), X_tr, y_tr, cv=cv, scoring="neg_root_mean_squared_error")
    p = dummy.predict(X_te)
    a, r, pr = pass_fail_metrics(y_te.values, p)
    rows.append(dict(scenario=name, model="Baseline: predict the mean", cv_rmse=cv_d.mean(), cv_std=cv_d.std(),
                     test_rmse=rmse(y_te, p), test_mae=mean_absolute_error(y_te, p), test_r2=r2_score(y_te, p),
                     pass_fail_accuracy=a, fail_recall=r, fail_precision=pr, best_params=""))
    preds["Baseline: predict the mean"] = p

    if "G2" in X_tr.columns:  # naive "copy last grade" baseline, only meaningful when G2 is known
        p = X_te["G2"].values.astype(float)
        a, r, pr = pass_fail_metrics(y_te.values, p)
        cv_c = [rmse(y_tr.iloc[v], X_tr["G2"].iloc[v]) for _, v in cv.split(X_tr)]
        rows.append(dict(scenario=name, model="Baseline: copy G2", cv_rmse=np.mean(cv_c), cv_std=np.std(cv_c),
                         test_rmse=rmse(y_te, p), test_mae=mean_absolute_error(y_te, p), test_r2=r2_score(y_te, p),
                         pass_fail_accuracy=a, fail_recall=r, fail_precision=pr, best_params=""))
        preds["Baseline: copy G2"] = p

    # --- tuned models --------------------------------------------------------
    for mname, (pipe, grid) in candidate_models(X_tr).items():
        gs = GridSearchCV(pipe, grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
        gs.fit(X_tr, y_tr)
        best = gs.best_estimator_
        cv_scores = -cross_val_score(best, X_tr, y_tr, cv=cv, scoring="neg_root_mean_squared_error")
        p = best.predict(X_te)
        a, r, pr = pass_fail_metrics(y_te.values, p)
        rows.append(dict(scenario=name, model=mname, cv_rmse=cv_scores.mean(), cv_std=cv_scores.std(),
                         test_rmse=rmse(y_te, p), test_mae=mean_absolute_error(y_te, p), test_r2=r2_score(y_te, p),
                         pass_fail_accuracy=a, fail_recall=r, fail_precision=pr,
                         best_params=str({k.replace("m__", ""): v for k, v in gs.best_params_.items()})))
        fitted[mname], preds[mname] = best, p
    return pd.DataFrame(rows), fitted, preds


# ----------------------------------------------------------------------------
# 5. Figures for results
# ----------------------------------------------------------------------------
def fig_model_comparison(res):
    models = ["Baseline: predict the mean", "Baseline: copy G2", "Ridge regression", "Random forest", "Gradient boosting"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, sc, title in zip(axs, ["A: early warning (no grades)", "B: with G1 and G2"],
                             ["A) Early warning: no previous grades", "B) With G1 and G2 known"]):
        sub = res[res.scenario == sc].set_index("model").reindex(models).dropna(subset=["test_rmse"])
        cols = [C_GREY if m.startswith("Baseline") else C_BLUE for m in sub.index]
        bars = ax.bar(range(len(sub)), sub.test_rmse, color=cols)
        ax.errorbar(range(len(sub)), sub.cv_rmse, yerr=sub.cv_std, fmt="o", color=C_ORANGE, ms=4, capsize=3,
                    label="CV RMSE (train folds)")
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels([m.replace("Baseline: ", "").replace(" regression", "").replace("predict the mean", "mean")
                            for m in sub.index], rotation=20, ha="right")
        for b, v in zip(bars, sub.test_rmse):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.08, f"{v:.2f}", ha="center", fontsize=9)
        ax.set_title(title)
    axs[0].set_ylabel("RMSE on held-out test set (grade points, lower = better)")
    axs[1].legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig04_model_comparison.png")
    plt.close(fig)


def fig_pred_vs_actual(y_te, predA, predB, nameA, nameB):
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 4.3), sharex=True, sharey=True)
    for ax, p, title in zip(axs, [predA, predB], [f"A) No grades ({nameA})", f"B) With G1, G2 ({nameB})"]):
        ax.scatter(y_te, p, s=22, alpha=0.7, color=C_BLUE, edgecolor="white", linewidth=0.4)
        ax.plot([0, 20], [0, 20], "--", color=C_RED, lw=1)
        ax.set(title=title, xlabel="actual G3", xlim=(-1, 21), ylim=(-1, 21))
    axs[0].set_ylabel("predicted G3")
    fig.tight_layout()
    fig.savefig(FIG / "fig05_pred_vs_actual.png")
    plt.close(fig)


def fig_importance(modelA, XteA, y_te, modelB, XteB):
    out = {}
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, model, X, title in zip(axs, [modelA, modelB], [XteA, XteB],
                                   ["A) No grades: what matters", "B) With G1/G2: what matters"]):
        r = permutation_importance(model, X, y_te, n_repeats=30, random_state=SEED,
                                   scoring="neg_root_mean_squared_error")
        imp = pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False).head(8)
        ax.barh(imp.index[::-1], imp.values[::-1], color=C_BLUE)
        ax.set(title=title, xlabel="increase in RMSE when feature is shuffled")
        out[title[:1]] = imp.round(3).to_dict()
    fig.tight_layout()
    fig.savefig(FIG / "fig06_feature_importance.png")
    plt.close(fig)
    return out


def error_analysis(y_te, predA, predB, XteB):
    zero = (y_te.values == 0)
    rows = []
    for label, p in [("A: early warning", predA), ("B: with G1, G2", predB)]:
        rows.append(dict(scenario=label, group="students with G3 = 0", n=int(zero.sum()), rmse=rmse(y_te[zero], p[zero]),
                         mean_error=float((p[zero] - y_te.values[zero]).mean())))
        rows.append(dict(scenario=label, group="students with G3 > 0", n=int((~zero).sum()), rmse=rmse(y_te[~zero], p[~zero]),
                         mean_error=float((p[~zero] - y_te.values[~zero]).mean())))
    tab = pd.DataFrame(rows)
    tab.to_csv(RES / "error_analysis_by_group.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    w = 0.35
    for i, (lab, col) in enumerate([("A: early warning", C_GREY), ("B: with G1, G2", C_BLUE)]):
        sub = tab[tab.scenario == lab]
        ax.bar(np.arange(2) + (i - 0.5) * w, sub.rmse, w, label=lab, color=col)
        for x, v in zip(np.arange(2) + (i - 0.5) * w, sub.rmse):
            ax.text(x, v + 0.1, f"{v:.1f}", ha="center", fontsize=9)
    ax.set_xticks(range(2))
    ax.set_xticklabels([f"G3 = 0  (n={int(zero.sum())})", f"G3 > 0  (n={int((~zero).sum())})"])
    ax.set(title="Where the errors come from: students who scored zero", ylabel="RMSE on test set")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig07_error_analysis.png")
    plt.close(fig)

    big = pd.DataFrame({"G1": XteB["G1"].values, "G2": XteB["G2"].values, "actual_G3": y_te.values,
                        "pred_A": predA.round(1), "pred_B": predB.round(1)})
    big["abs_err_B"] = (big.pred_B - big.actual_G3).abs().round(1)
    big = big.sort_values("abs_err_B", ascending=False).head(8)
    big.to_csv(RES / "biggest_misses_scenarioB.csv", index=False)
    return tab, big


# ----------------------------------------------------------------------------
def main():
    df, audit = load_and_audit()
    print("AUDIT", json.dumps(audit, indent=1))
    corr = eda_plots(df)

    X_all = engineer(df.drop(columns="G3"))
    y = df["G3"]
    XA = X_all.drop(columns=["G1", "G2"])  # scenario A
    XB = X_all  # scenario B
    # ONE split for both scenarios; test set is untouched until the final evaluation
    idx_tr, idx_te = train_test_split(df.index, test_size=0.2, random_state=SEED)
    y_tr, y_te = y.loc[idx_tr], y.loc[idx_te]

    resA, fitA, predsA = run_scenario("A: early warning (no grades)", XA.loc[idx_tr], XA.loc[idx_te], y_tr, y_te)
    resB, fitB, predsB = run_scenario("B: with G1 and G2", XB.loc[idx_tr], XB.loc[idx_te], y_tr, y_te)
    res = pd.concat([resA, resB], ignore_index=True)
    res.round(3).to_csv(RES / "model_comparison.csv", index=False)
    print(res.round(3).drop(columns="best_params").to_string())

    fig_model_comparison(res)
    bestA = resA[~resA.model.str.startswith("Baseline")].sort_values("cv_rmse").iloc[0].model
    bestB = resB[~resB.model.str.startswith("Baseline")].sort_values("cv_rmse").iloc[0].model  # chosen by CV, not test
    fig_pred_vs_actual(y_te, predsA[bestA], predsB[bestB], bestA, bestB)
    imp = fig_importance(fitA[bestA], XA.loc[idx_te], y_te, fitB[bestB], XB.loc[idx_te])
    tab, big = error_analysis(y_te, predsA[bestA], predsB[bestB], XB.loc[idx_te])
    print(tab.round(2).to_string())
    print(big.to_string())

    # Sensitivity check: how much of scenario A depends on the 'absences' dropout quirk?
    XA2 = XA.drop(columns=["log_absences"])
    resA2, _, _ = run_scenario("A2: early warning, absences removed", XA2.loc[idx_tr], XA2.loc[idx_te], y_tr, y_te)
    resA2.round(3).to_csv(RES / "sensitivity_no_absences.csv", index=False)
    print(resA2.round(3).drop(columns="best_params").to_string())

    summary = {"audit": audit, "split": {"train": len(idx_tr), "test": len(idx_te), "seed": SEED},
               "best_model_A_by_cv": bestA, "best_model_B_by_cv": bestB,
               "top_importance": imp, "spearman_top": corr.round(3).to_dict()}
    (RES / "summary.json").write_text(json.dumps(summary, indent=2))
    print("best by CV:", bestA, "|", bestB)


if __name__ == "__main__":
    main()
