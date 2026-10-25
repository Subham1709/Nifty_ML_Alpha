# Web app

Live: https://subham-nifty-ml-alpha.streamlit.app

```
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py
```

The app does not train anything. `app/precompute.py` runs the six (validation, model)
combinations once, on the real data and on the random-walk placebo, with the same code as
`scripts/02_run_experiment.py`, and saves the out-of-sample predictions to `app/assets/`.
The app rebuilds the long-short portfolios from them with `src/alpha/backtest.py`, so the
trading cost and quintile size can be changed live. At the defaults (15 bps, quintiles)
it reproduces `results/summary.md` and `results/placebo/summary.md` exactly.

Rebuild the assets after changing the models or data: `python app/precompute.py` (needs the
full `requirements.txt`).

Deploy on Streamlit Community Cloud: repo `Subham1709/Nifty_ML_Alpha`, branch `main`,
main file `app/streamlit_app.py`.
