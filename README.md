# Machine Learning Projects

Nine ML/DL projects covering classical ML, deep learning, NLP, audio,
graph neural networks, reinforcement learning, and MLOps serving.

## Important: these were reviewed but not executed

Unlike the companion `AI-portfolio-reviewed` repo (where every project
was actually run and tested), the environment used to review this
batch had none of the required heavy libraries installed (torch,
transformers, xgboost, lightgbm, mlflow, langchain, chromadb,
gymnasium) and no internet access to install them. Every file was
read carefully and several real bugs were fixed by reasoning through
the code (see below), but **nothing here has been execution-verified
the way the other repo was. Run each project yourself in a real
Python environment before relying on it in an interview.**

## What was fixed

Several projects claimed to do more than their code actually did:

- **`mlops-house-price-predictor/app.py`** — the `/predict` endpoint
  ignored the trained model completely and returned a hardcoded mock
  formula (`sum(features) * 50000`). Fixed to load and use the actual
  model saved by `train_mlflow.py`.
- **`computer-vision-defect-detector/src/train.py`** — built a model
  and immediately saved it with zero actual training (no loop, no
  data, no loss). Fixed to run a real multi-epoch loop on synthetic
  data (no real images ship with this repo).
- **`time-series-stock-forecaster/src/train.py`** — ran exactly one
  gradient step on random noise and called it "training." Fixed to a
  real 20-epoch loop on synthetic random-walk sequences.
- **`rl-trading-agent/src/agent.py`** — had no learning step at all,
  only the action-selection (inference) path. Added a complete DQN
  training loop: replay buffer, target network, Bellman update, plus
  a tiny synthetic trading environment so it runs end to end.
- **`nlp-sentiment-analyzer/`** — had no training script at all;
  `predict.py` loaded a checkpoint file nothing in the repo ever
  created. Added `src/train.py` with a complete (if tiny) training loop.
- **`audio-keyword-spotting-engine/`** — had only the preprocessing
  and model classes, no script tying them together. Added
  `src/train.py` running the full pipeline on synthetic waveforms.
- **`churn-prediction-system/`** — referenced `data/dataset.csv`,
  which didn't exist anywhere in the repo. Added
  `src/make_sample_data.py` to generate a synthetic dataset with a
  learnable churn signal; the preprocessing pipeline was verified to
  run against it successfully (pandas/scikit-learn were available to test).
- **`graph-fraud-detection-gnn/requirements.txt`** — listed
  `torch-geometric` as a dependency, but the code only uses plain
  PyTorch (it's a hand-rolled, simplified GCN layer). Removed the
  unused dependency and added a README note clarifying this isn't
  using the real PyTorch Geometric library.

All of the above are now in the code, not just "mocked" or missing —
but on **synthetic data**, since no real datasets ship with this
repo. Read each project's README for exactly what's synthetic and
what you'd need to swap in for production.

## Projects

| Project | Area | Status |
|---|---|---|
| `churn-prediction-system` | Classical ML (XGBoost + FastAPI) | Preprocessing verified to run; needs xgboost installed for training |
| `mlops-house-price-predictor` | MLOps (LightGBM + MLflow + FastAPI) | Serving bug fixed; needs mlflow/lightgbm installed |
| `computer-vision-defect-detector` | Deep learning (ResNet18 transfer learning) | Training loop fixed (synthetic data); needs torch/torchvision |
| `time-series-stock-forecaster` | Deep learning (LSTM) | Training loop fixed (synthetic data); needs torch |
| `nlp-sentiment-analyzer` | NLP (DistilBERT fine-tuning) | Training script added; needs torch/transformers + internet |
| `audio-keyword-spotting-engine` | Audio (Mel-Spectrogram + CNN) | Training script added; needs torch/torchaudio |
| `graph-fraud-detection-gnn` | Graph neural networks (hand-rolled GCN) | Requirements fixed; needs torch |
| `rl-trading-agent` | Reinforcement learning (DQN) | Full training loop added; needs torch |
| `rag-document-qa-pipeline` | RAG (LangChain + ChromaDB) | Compatibility notes added; needs langchain/chromadb + internet |

## License

MIT — feel free to reuse any part of this for learning purposes.
