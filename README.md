# DigitVision ✍️

ML-powered **handwritten digit classifier** — draw a digit 0–9 and a real trained model predicts it, right in the browser.

**Live demo:** https://digitvision-hvrgaming2705s-projects.vercel.app

## How it works

1. **Data** — 1,797 8×8 grayscale digit images from scikit-learn's digits dataset (classes 0–9), pixel values scaled to 0–1.
2. **Model** — Softmax regression (multinomial logistic regression, scikit-learn). One 64-pixel weight map per digit class + bias. **5-fold cross-validation accuracy: 97.0%.**
3. **Browser demo** — `train.py` exports the 10 weight vectors, 10 biases, and one sample digit per class to `weights.json`. Your drawing is downsampled to 8×8 and JavaScript re-runs the exact same math as sklearn's `predict_proba` (linear scores + softmax) — verified to match Python's probabilities to 8 decimals. No backend, no API calls.

## Reproduce

```bash
pip install scikit-learn numpy
python train.py   # retrains and regenerates weights.json
```

## Files

- `train.py` — dataset loading, training, evaluation, weight export (+ JS parity check)
- `weights.json` — exported model (10 × 64 weights + 10 biases + sample digits)
- `index.html` — the interactive demo site (draw-on-canvas + probability bars)
- `make_ppt.py` — builds the project presentation (python-pptx)
- `vercel.json` — static hosting config + security headers

Built by [Vikranth Reddy](https://github.com/HVRGAMING2705) · B.Tech CSE (AI & ML), Woxsen University
