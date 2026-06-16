# CAR-T Infection Risk Prescreening Calculator

Streamlit Community Cloud upload package for the research-use CAR-T day-100 infection risk stratification calculator.

## Required Streamlit Cloud Settings

- Repository: `Lab-Leon/cart-infection-risk-calculator`
- Branch: `main`
- Main file path: `app/streamlit_app.py`
- Python version: `3.11`
- Secrets: leave blank

## Local Test

```powershell
python -m streamlit run app/streamlit_app.py --server.port 8501
```

## Included Files

- `app/streamlit_app.py`
- `code/preprocessing_helpers.py`
- `models/final_model_bundle.joblib`
- `models/final_model_metadata.json`
- `processed/publication_tables.xlsx`
- `requirements.txt`
- `.streamlit/config.toml`

No raw CIBMTR source data, patient-level files, manuscript files, figures, cache files or local logs are included.

## Intended Use

This calculator is for research-use risk stratification and reviewer assessment only. It is not intended to guide patient-care decisions without prospective validation and local recalibration.
