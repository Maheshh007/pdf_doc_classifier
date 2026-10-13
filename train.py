"""Step 4/6: train + evaluate document classifier.
Experiments
  A) stratified 80/20 split on everything (in-distribution)
  B) train on SYNTHETIC only -> test on ALL real documents (honest domain-transfer check)
  C) leave-one-out over REAL documents: each real doc is predicted by a model that never saw it
Real documents get a higher sample weight (REAL_WEIGHT) because the synthetic data is easier than reality.
Final model = best pipeline refit on all data, saved with joblib (vectoriser + classifier in one Pipeline).
"""
import os, json
import numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(ROOT, "data", "dataset.csv"))
df["clean_text"] = df.clean_text.fillna("")
LABELS = ["invoice", "bill", "receipt"]
REAL_WEIGHT = 5.0

def fit_weighted(model, X, y, weights):
    """Fit a pipeline passing sample weights to its final estimator."""
    step = model.steps[-1][0]
    return model.fit(X, y, **{f"{step}__sample_weight": weights})

weights = np.where(df.source == "real", REAL_WEIGHT, 1.0)

def tfidf(): return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
CANDIDATES = {
    "tfidf+LogReg": lambda: make_pipeline(tfidf(), LogisticRegression(max_iter=2000, C=10, class_weight="balanced")),
    "tfidf+LinearSVC": lambda: make_pipeline(tfidf(), CalibratedClassifierCV(LinearSVC(C=1, class_weight="balanced"), cv=3)),
    "tfidf+NaiveBayes": lambda: make_pipeline(tfidf(), MultinomialNB(alpha=.3)),
}

# ---- cross-validation model selection --------------------------------
cv = StratifiedKFold(5, shuffle=True, random_state=0)
cv_res = {n: cross_val_score(m(), df.clean_text, df.label, cv=cv, scoring="f1_macro") for n, m in CANDIDATES.items()}
for n, s in cv_res.items(): print(f"CV macro-F1 {n:18s} {s.mean():.3f} ± {s.std():.3f}")
best = max(cv_res, key=lambda n: cv_res[n].mean())
# Differences this small are noise; prefer LogReg (best-behaved probabilities for the app's confidence threshold)
if cv_res["tfidf+LogReg"].mean() >= cv_res[best].mean() - 0.01: best = "tfidf+LogReg"
print("best:", best)

report = {"cv_macro_f1": {n: [round(s.mean(), 4), round(s.std(), 4)] for n, s in cv_res.items()}, "selected": best}

# ---- A) hold-out split -----------------------------------------------
Xtr, Xte, ytr, yte, _, ite = train_test_split(df.clean_text, df.label, df.index, test_size=.2, stratify=df.label, random_state=42)
m = CANDIDATES[best](); m.fit(Xtr, ytr); pred = m.predict(Xte)
report["holdout"] = dict(n_test=len(yte), accuracy=round(accuracy_score(yte, pred), 4), macro_f1=round(f1_score(yte, pred, average="macro"), 4))
print("\n== A) hold-out\n", classification_report(yte, pred, digits=3))
cm = confusion_matrix(yte, pred, labels=LABELS)
ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(cmap="Blues"); plt.title("Hold-out confusion matrix"); plt.savefig(os.path.join(ROOT, "reports", "confusion_holdout.png"), dpi=130, bbox_inches="tight"); plt.close()
errs = df.loc[ite][pred != yte.values][["file", "label"]].assign(predicted=pred[pred != yte.values])
errs.to_csv(os.path.join(ROOT, "reports", "holdout_errors.csv"), index=False)

# ---- B) synthetic -> real transfer -----------------------------------
syn, real = df[df.source != "real"], df[df.source == "real"]   # "syn" = synthetic + public/generated
m2 = CANDIDATES[best](); m2.fit(syn.clean_text, syn.label); p2 = m2.predict(real.clean_text)
report["synthetic_to_real"] = dict(n_test=len(real), accuracy=round(accuracy_score(real.label, p2), 4),
                                   per_class={c: [int((p2[(real.label == c).values] == c).sum()), int((real.label == c).sum())] for c in LABELS if (real.label == c).any()})
print("== B) train synthetic only -> test real docs: acc", report["synthetic_to_real"]["accuracy"], "(correct/total per class:", report["synthetic_to_real"]["per_class"], ")")
real.assign(predicted=p2)[["file", "label", "predicted", "extraction_method", "n_words"]].to_csv(os.path.join(ROOT, "reports", "real_transfer_synthetic_only.csv"), index=False)

# ---- C) leave-one-out over real documents ----------------------------
loo_pred = []
for i in real.index:
    keep = df.index != i
    mi = CANDIDATES[best](); fit_weighted(mi, df.clean_text[keep], df.label[keep], weights[keep])
    loo_pred.append(mi.predict([df.clean_text[i]])[0])
loo_pred = np.array(loo_pred)
report["real_leave_one_out"] = dict(n=len(real), accuracy=round(accuracy_score(real.label, loo_pred), 4),
                                    per_class={c: [int((loo_pred[(real.label == c).values] == c).sum()), int((real.label == c).sum())] for c in LABELS if (real.label == c).any()})
print("== C) leave-one-out on real docs: acc", report["real_leave_one_out"]["accuracy"], report["real_leave_one_out"]["per_class"])
real.assign(predicted=loo_pred)[["file", "label", "predicted", "extraction_method", "n_words"]].to_csv(os.path.join(ROOT, "reports", "real_leave_one_out.csv"), index=False)

# ---- top features ------------------------------------------------------
final = fit_weighted(CANDIDATES["tfidf+LogReg"](), df.clean_text, df.label, weights)
vec, lr = final[0], final[1]; names = np.array(vec.get_feature_names_out())
report["top_terms"] = {c: names[np.argsort(-lr.coef_[i])[:12]].tolist() for i, c in enumerate(lr.classes_)}

# ---- save final model (selected pipeline refit on ALL data) -----------
final = fit_weighted(CANDIDATES[best](), df.clean_text, df.label, weights)
os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
joblib.dump(final, os.path.join(ROOT, "models", "doc_classifier.joblib"))
json.dump(report, open(os.path.join(ROOT, "reports", "metrics.json"), "w"), indent=2)
print("saved model + metrics")
