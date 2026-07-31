# Snowball Risk Studio

A polished Streamlit dashboard for the European Snowball model in the supplied `Snowball_Pricing(2).ipynb`.

## What it does

- Day-0 Monte Carlo PV / NPV
- Target coupon that makes PV equal a selected issue price
- Delta / Gamma / Price curves across underlying levels and valuation times
- Delta heatmap across spot and time
- Observation-date Delta before/after and Delta jump analysis
- KO barrier schedule and first-KO probabilities
- Maturity-safe / maturity-loss outcome probabilities
- Target coupon sensitivity to dividend yield and volatility
- Random-seed stability check
- CSV downloads and portable ZIP analysis package
- External-facing styling and model-scope notes

## Model basis

The core model follows the supplied notebook's Cell 3 / Cell 4 logic:

- Risk-neutral GBM with drift `r - q`
- Monthly KO observations after the lock-in period
- Linear KO step-down
- European KI checked only at maturity
- Deterministic `r`, `q`, and `vol`
- Common-random-number central finite-difference Greeks

The numerical engine preserves the notebook's legacy `RandomState` random-number sequence convention for reproducibility.

## Project structure

```text
snowball_visual_tool/
├── app.py
├── snowball_engine.py
├── requirements.txt
├── README.md
├── .gitignore
└── .streamlit/
    └── config.toml
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\\Scripts\\activate         # Windows PowerShell

pip install -r requirements.txt
streamlit run app.py
```

## Deploy for sharing

The project is designed to be pushed to GitHub and deployed as a Streamlit app. Use `app.py` as the entrypoint and `requirements.txt` for dependencies.

After deployment, the app runs in the browser. Other people only need the app URL; they do not need Python, your notebook, or your local Terminal environment.

## Performance guidance

- Exploration: 5,000–10,000 paths
- Final analysis: 30,000+ paths
- Delta heatmap is optional because it is more expensive than a single curve
- Keep the curve grid moderate when running on hosted infrastructure

## Disclaimer

This dashboard is for research, education and model demonstration. It is not formal investment, trading, risk-management or valuation advice.
