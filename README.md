# IMDb Sentiment (fastai)

Tiny text classifier built with **fastai**. Train on **Kaggle GPU**, export the model, run a **Streamlit** app locally (CPU), and optionally deploy to **Streamlit Community Cloud**.

---

## Quickstart

### 1) Train on Kaggle

- Enable **GPU** and **Internet** in your Kaggle notebook.
- First cell:

```python
!pip -q install --upgrade fastai==2.7.15 nbdev gradio streamlit
import torch, random, numpy as np
random.seed(42); np.random.seed(42); torch.manual_seed(42)
print("CUDA available:", torch.cuda.is_available())
```

- Use `notebooks/01_train_imdb_sentiment.md` (copy those cells into Kaggle).
- Export the trained model:

```python
learn.export('imdb_sentiment_fastai.pkl')
```

- Download the exported file and place it locally at: `models/imdb_sentiment_fastai.pkl`.

---

### 2) Run locally (CPU)

> Create a CPU-only environment with Conda using the provided `fastai-2025-cpu.yml`.

```bash
conda env create -f fastai-2025-cpu.yml
conda activate fastai-2025-cpu
streamlit run app.py
```

---

### 3) Optional: Gradio (local)

```bash
python app_gradio.py
```

---

### 4) Deploy to Streamlit Community Cloud

- Push this repo to GitHub with `models/imdb_sentiment_fastai.pkl` included.
- In Streamlit Cloud, set **Main file** to `app.py`.
- If you prefer pip-only installs on the cloud runner, ensure `requirements.txt` is present.

---

## Project Structure

```
imdb-sentiment-fastai/
├─ app.py                     # Streamlit app (CPU)
├─ app_gradio.py              # Optional Gradio mirror
├─ fastai-cpu.yml             # Conda/Mamba env for local CPU
├─ requirements.txt           # For Streamlit Cloud (pip)
├─ notebooks/
│  └─ 01_train_imdb_sentiment.md  # Kaggle-ready cells to copy
├─ models/
│  └─ imdb_sentiment_fastai.pkl   # exported on Kaggle, then added here
├─ scripts/
│  └─ verify_env.py
├─ .streamlit/
│  └─ config.toml
├─ .gitignore
└─ README.md
```

---

## Notes & Tips

- Training happens on **Kaggle GPU**; the local environment is **CPU-only** for running demos and light dev.
- If you **don’t** want to commit the model artifact to the repo, add `models/*.pkl` to `.gitignore` and fetch the file from a hosted URL at app startup instead.
- To verify your local environment quickly:

```bash
python scripts/verify_env.py
```

---

## Live Demo (optional)

Once deployed, add your link here:

**Streamlit:** https://YOUR-STREAMLIT-APP-URL  
**Gradio (Hugging Face Spaces):** https://huggingface.co/spaces/YOUR-SPACE

---
