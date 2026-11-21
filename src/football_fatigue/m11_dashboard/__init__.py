"""M11 Step 8: interactive identity-recovery dashboard over the frozen Step 6 model (development matches only).

    python -m football_fatigue.m11_dashboard                 # launch the Streamlit app
    python -m football_fatigue.m11_dashboard verify --match 3829   # batch reproduction check, prints only

The package reads existing artifacts and writes nothing:

- :mod:`.data` discovers development matches, loads matcher-visible task B instances, public encounter metadata and, kept
  apart, the provider reference;
- :mod:`.inference` loads the checksummed frozen model and runs the unchanged Step 6 feature, calibration, assignment and
  decision code on one instance;
- :mod:`.pitch` draws the RAW tracking on a pitch (Plotly); :mod:`.research` reads the saved scoring summaries;
- ``app.py`` is the Streamlit user interface.

Every agreement shown is provider-reference agreement, not verified identity accuracy.
"""
