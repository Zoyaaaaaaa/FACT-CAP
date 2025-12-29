Quick Streamlit instructions

1. (Optional) Create a virtualenv and activate it:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install requirements (this will install PyTorch — expect a large download):

```bash
pip install -r requirements-streamlit.txt
```

3. Run the app from the repository root:

```bash
streamlit run streamlit_app.py
```

4. In the UI upload an image and click "Analyze Image". The app calls `scripts/predict.py` with the uploaded file and `deepfake_cnn_cpuv1.pth` in the repo root.

Notes:
- `scripts/predict.py` contains a placeholder model class. If your saved model uses a custom class, update `scripts/predict.py` to match the original architecture.
- If you want a lightweight dev path without installing PyTorch locally, consider running the Streamlit app in a container that has CUDA/CPU PyTorch preinstalled.
