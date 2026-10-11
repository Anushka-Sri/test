# Moderation Model Lab (Streamlit)

Web app for testing the HateXplain hate / offensive / normal baselines
(TF-IDF + Logistic Regression, class-weighted LR, Naive Bayes).

Run locally:

    pip install -r requirements.txt
    streamlit run app.py

Deploy: push this folder to GitHub, then create an app at share.streamlit.io
with main file `app.py` and Python 3.13.

The trained models are in `artifacts/`. scikit-learn is pinned to 1.9.1
because the saved models must be loaded with the version that trained them.
