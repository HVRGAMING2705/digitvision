"""DigitVision — handwritten digit classifier.
Trains multinomial logistic regression (softmax) on the sklearn digits
dataset (1,797 8x8 grayscale images, pixel values 0..16) and exports the
weights (coef_ 10x64 + intercept_ 10) as JSON so the browser demo
reproduces sklearn's predict_proba exactly, with no backend.
"""
import json

import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# ---- Dataset: 8x8 grayscale digits, pixel values 0..16 ----
digits = load_digits()
X = digits.data / 16.0          # scale to 0..1 (same normalization in JS)
y = digits.target
print(f"dataset: {X.shape[0]} images, {X.shape[1]} features, classes {sorted(set(y))}")


def make_model():
    return LogisticRegression(
        solver="lbfgs", C=1.0, max_iter=2000, random_state=42)


# Stratified 5-fold CV (honest metric), then final model on ALL data.
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_accs = []
for tri, tei in skf.split(X, y):
    c = make_model()
    c.fit(X[tri], y[tri])
    fold_accs.append(accuracy_score(y[tei], c.predict(X[tei])))
acc = float(np.mean(fold_accs))
print(f"5-fold CV accuracy: {acc:.4f} (folds={['%.4f' % a for a in fold_accs]})")

clf = make_model()
clf.fit(X, y)
print(f"trained on {X.shape[0]} samples; train accuracy: {accuracy_score(y, clf.predict(X)):.4f}")

# Export: coef_ (10 x 64), intercept_ (10), plus a few labeled test digits
# for the demo's "load a sample" buttons.
rng = np.random.RandomState(42)
samples = []
for d in range(10):
    idx = rng.choice(np.flatnonzero(y == d))
    samples.append({
        "digit": int(d),
        "pixels": [round(float(v), 4) for v in digits.data[idx] / 16.0],
    })

weights = {
    "coef": [[round(float(v), 6) for v in row] for row in clf.coef_],
    "intercept": [round(float(v), 6) for v in clf.intercept_],
    "accuracy": round(float(acc), 4),
    "fold_accuracies": [round(float(a), 4) for a in fold_accs],
    "n_samples": int(X.shape[0]),
    "samples": samples,
}

with open("weights.json", "w") as f:
    json.dump(weights, f)
print(f"exported weights.json ({clf.coef_.shape[0]} classes x {clf.coef_.shape[1]} features)")

# ---- Parity check: re-implement the JS-side softmax math in pure Python
#      and verify it matches sklearn's predict_proba to 1e-5 ----
COEF = np.array(weights["coef"])
B = np.array(weights["intercept"])


def js_predict_proba(x):
    z = COEF @ x + B
    e = np.exp(z - z.max())
    return e / e.sum()


ok = True
for i in rng.choice(len(X), 8, replace=False):
    mine = js_predict_proba(X[i])
    sk = clf.predict_proba(X[i:i + 1])[0]
    diff = float(np.abs(mine - sk).max())
    ok = ok and diff < 1e-5
    print(f"parity check: true={y[i]} pred={mine.argmax()} diff={diff:.2e}")
print("JS parity:", "PASS" if ok else "FAIL")

# Sanity: classify the demo samples with the final model
for s in samples:
    x = np.array(s["pixels"])
    p = js_predict_proba(x)
    print(f"sample {s['digit']}: predicted {p.argmax()} (prob {p.max():.3f})")
