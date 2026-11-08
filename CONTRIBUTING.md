# Contributing

Anyone is welcome: open an issue, or fork the repository and send a pull request.

1. Fork, then `git clone` your fork and `pip install -r requirements.txt pytest`.
2. Make the change on a branch. Keep the model's numbers unchanged unless the pull request is about the
   model; `pytest -q` checks the decoder against the original checkpoint.
3. Run `pytest -q` and, for app changes, `streamlit run streamlit_app.py`.
4. Open the pull request with a short description of what changed and how you checked it.
   GitHub Actions runs the tests on every pull request.

## Ideas

- Live: test `live_lsl.py` with a real headset and EmotivPRO's LSL output; a live page in the app.
- Other headsets: map their channel names to the 30 FLEX2 channels (or retrain).
- A model for more participants, or one that generalises to new pictures.
- Better explanations in the app; translations.

Do not add real EEG recordings of people to this repository.
