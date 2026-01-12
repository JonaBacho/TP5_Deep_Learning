# 🧠 TP5: Sequence Modeling & Attention Mechanisms - MLOps Pipeline

[![TP5 Training](https://github.com/JonaBacho/TP5_Deep_Learning/actions/workflows/tp5-sequence-training.yml/badge.svg)](https://github.com/JonaBacho/TP5_Deep_Learning/actions/workflows/tp5-sequence-training.yml)
[![MLflow](https://img.shields.io/badge/MLflow-2.9.2-blue.svg)](https://mlflow.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**École Nationale Supérieure Polytechnique de Yaoundé**  
Département de Génie Informatique - 5GI  
Instructeurs: Louis Fippo

Ce projet implémente le **TP5 sur les mécanismes d'attention et le traitement de séquences**. Il explore l'évolution des RNNs vers les architectures avec attention, incluant un défi de recherche sur l'amélioration d'un modèle latent temporel (TAP - ArXiv:2102.05095) pour mieux gérer les dépendances long-terme.

---

## 🎯 Objectifs d'Apprentissage

- **Implémenter** et visualiser le Scaled Dot-Product Attention
- **Hybrider** RNNs (LSTM/GRU) avec des couches d'Attention pour seq2seq
- **Comprendre** l'architecture des Temporal Latent Space Models
- **Rechercher** et proposer des améliorations architecturales pour dépendances long-terme
- **Pratiquer** l'écriture scientifique pour conférences AI (format NeurIPS/ICLR)

---

## 📁 Structure du Projet

```text
.
├── config/
│   └── mlflow_config.py              # Configuration MLflow centralisée
├── src/
│   ├── attention_mechanism.py        # Exercise 1: Attention de base
│   ├── lstm_attention_seq2seq.py     # Exercise 2: Seq2Seq + Attention
│   ├── tap_improvement.py            # Exercise 3: Improved TAP
│   ├── app.py                        # API Flask (optionnel)
│   ├── auto_promote.py               # Promotion automatique
│   └── promote_model.py              # Gestion manuelle des stages
├── paper/                            # Article scientifique
│   ├── main.tex                      # Paper LaTeX
│   ├── figures/                      # Diagrammes architectures
│   └── references.bib                # Bibliographie
├── attention_results/                # Visualisations attention
├── seq2seq_results/                  # Résultats Seq2Seq
├── tests/
│   └── test_model.py                 # Tests unitaires
├── .github/workflows/
│   ├── tp5-sequence-training.yml     # Pipeline principal
│   └── deploy.yml                    # Déploiement API (optionnel)
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🚀 Exercices du TP5

### Part 1: Mastering Basic Attention (2h)

#### Théorie: Why Attention?

**Questions théoriques**:
1. **Scaled Dot-Product Attention**: Formule mathématique avec Q, K, V
   ```
   Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
   ```
   - Pourquoi le scaling factor `1/sqrt(d_k)` est nécessaire?
   - Réponse: Évite les valeurs extrêmes dans softmax quand d_k est grand

2. **Self-Attention vs Cross-Attention**:
   - Self-Attention: Q, K, V de la MÊME séquence
   - Cross-Attention: Q d'une séquence, K, V d'une AUTRE

#### Exercise 1: Basic Attention Layer

Implémentation d'une couche d'attention custom dans Keras.

**Architecture**:
```
Input → GRU(return_sequences=True) → SimpleAttention → Dense → Output
```

**Expérience MLflow**: `TP5-Exercise1-BasicAttention`

**Métriques**:
- `test_accuracy`: Classification accuracy
- `attention_span`: Nombre de time steps avec poids significatifs (>0.05)
- Visualisations des poids d'attention

**Dataset**: Séquences synthétiques (3 classes) avec patterns temporels

### Part 2: Seq2Seq with Memory (3h)

#### Exercise 2: LSTM-Attention for Time Series

Modèle hybride pour prédiction de séries temporelles.

**Architecture**:
```
Encoder: Bidirectional LSTM
Decoder: LSTM + Bahdanau (Additive) Attention
```

**Flux**:
1. Encoder encode la séquence d'entrée
2. Decoder génère la séquence de sortie
3. À chaque step, Attention se focalise sur parties pertinentes de l'input

**Expérience MLflow**: `TP5-Exercise2-LSTM-Attention-Seq2Seq`

**Métriques**:
- `test_loss` (MSE)
- `test_mae`
- `attention_span`: Portée de l'attention
- `avg_attention_position`: Position moyenne focalisée

**Dataset**: Séries temporelles synthétiques (combinaison sinus + trends)
- Input: 50 time steps
- Output: 10 time steps (prédiction)

### Part 3: Research Challenge - TAP Improvement (6h+)

#### Context: ArXiv 2102.05095

**Paper**: "Temporal Latent Space Modeling for Video Generation" (TAP)

**Problème**: TAP peut être amélioré pour maintenir la cohérence sur fenêtres temporelles très longues

#### Challenge: Long-Term Consistency

Proposer une modification architecturale pour mieux gérer les dépendances long-terme.

**Améliorations Implémentées**:

1. **Temporal Transformer Block**
   - Multi-head attention pour dépendances long-range
   - Remplace transitions temporelles standard

2. **Memory-Augmented Module**
   - Module de mémoire externe (32 slots)
   - Stocke "keyframes" importants
   - Inspiré des Memory Networks

3. **Hierarchical Temporal Encoder**
   - 3 niveaux temporels:
     - Niveau 1: Court-terme (frames individuels)
     - Niveau 2: Moyen-terme (segments)
     - Niveau 3: Long-terme (séquence complète)
   - Pooling multi-échelle

**Architecture Improved TAP**:
```
Input Sequence
    ↓
Hierarchical Temporal Encoder (3 levels)
    ↓
Temporal Transformer (Multi-head Attention)
    ↓
Memory Module (Keyframe Storage)
    ↓
Latent Space Projection
    ↓
Decoder (LSTM + Dense)
    ↓
Reconstruction
```

**Expérience MLflow**: `TP5-Exercise3-ImprovedTAP`

**Métriques**:
- `test_loss` (MSE reconstruction)
- `test_mae`
- Analyse qualitative de la cohérence temporelle

**Dataset**: Moving MNIST-like (simplifié)
- Objets en mouvement sur 16 frames
- Test des dépendances long-terme

#### Submission: Scientific Paper

**Format**: 4 pages (+ références) style NeurIPS/ICLR

**Sections requises**:
1. **Abstract**: Problème et amélioration proposée
2. **Introduction**: Motivation pour dépendances long-terme
3. **Proposed Method**: Description mathématique et architecturale
4. **Experiments**: Comparaison TAP original vs Improved
5. **Ablation Study**: Impact de chaque module
6. **Conclusion**: Limitations et travaux futurs

**Contrainte**: Pas de transformers pré-entraînés "black-box"

---

## 🛠️ Installation

### Prérequis
- Python 3.10+
- TensorFlow 2.15
- LaTeX (pour compiler le paper)
- Accès à un serveur MLflow

### Installation Locale

```bash
git clone <votre-repo-url>
cd TP5_Sequence_Attention

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Éditer .env avec credentials MLflow
```

---

## 🔌 Utilisation

### Exécution via GitHub Actions

```bash
# Push → déclenche automatiquement
git push origin main

# Ou manuel depuis GitHub UI
# Actions → "TP5 - Sequence Modeling & Attention Training"
# Choisir: all, exercise1, exercise2, exercise3
```

**Durée**:
- Exercise 1: ~15 min
- Exercise 2: ~25 min
- Exercise 3: ~35 min
- Total: ~35-40 min (parallèle)

### Exécution Locale

```bash
# Exercise 1: Basic Attention
python src/attention_mechanism.py

# Exercise 2: Seq2Seq Attention
python src/lstm_attention_seq2seq.py

# Exercise 3: Improved TAP
python src/tap_improvement.py
```

---

## 🤖 Pipeline CI/CD

### 1. TP5 - Sequence Training (`tp5-sequence-training.yml`)

**Jobs**:
1. **exercise1-basic-attention** (15 min)
   - GRU + Attention
   - Visualisations poids d'attention
   
2. **exercise2-lstm-attention** (25 min)
   - Seq2Seq Bi-LSTM + Bahdanau Attention
   - Heatmap attention temporelle

3. **exercise3-tap-improvement** (35 min)
   - Improved TAP architecture
   - Transformer + Memory + Hierarchical

4. **promote-best-model**
   - Compare les 3 exercices
   - Promeut le meilleur modèle

5. **summary**
   - Rapport consolidé
   - Artifacts: logs + visualisations

---

## 📊 Visualisation des Résultats

### Dans MLflow UI

```bash
# Expériences créées:
- TP5-Exercise1-BasicAttention             (1 run)
- TP5-Exercise2-LSTM-Attention-Seq2Seq     (1 run)
- TP5-Exercise3-ImprovedTAP                (1 run)
- TP5-Theory-ScaledAttention               (1 run - analyse théorique)
```

**Métriques Exercise 1**:
- `test_accuracy`
- `average_attention_span`

**Métriques Exercise 2**:
- `test_loss`, `test_mae`
- `attention_span`
- `avg_attention_position`

**Métriques Exercise 3**:
- `test_loss`, `test_mae`
- `trainable_parameters`

**Visualisations**:
- Poids d'attention (heatmaps)
- Attention weights par time step
- Reconstruction vidéo (Moving MNIST)

---

## ⚙️ Configuration

### Variables d'Environnement

```bash
MLFLOW_TRACKING_URI=http://your-mlflow-server:5000
MLFLOW_TRACKING_USERNAME=your_username
MLFLOW_TRACKING_PASSWORD=your_password

MODEL_NAME=sequence-attention-model

MIN_ACCURACY=0.75  # Pour promotion

MLFLOW_S3_ENDPOINT_URL=https://your-s3
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
```

---

## 📈 Résultats Attendus

### Exercise 1: Basic Attention
```
Dataset: 1000 synthetic sequences (50 time steps)
Architecture: GRU + Attention
Test Accuracy: ~0.85-0.90
Attention Span: ~15-20 time steps
Training Time: ~10 min (20 epochs)
```

### Exercise 2: Seq2Seq Attention
```
Task: Time series forecasting (50→10 steps)
Architecture: Bi-LSTM Encoder + LSTM Decoder + Bahdanau Attention
Test MAE: ~0.15-0.20
Attention Span: ~25-30 time steps
Training Time: ~20 min (30 epochs)
Observation: Attention se focalise sur patterns pertinents
```

### Exercise 3: Improved TAP
```
Task: Video reconstruction (16 frames)
Improvements: Transformer + Memory + Hierarchical
Test MSE: ~0.01-0.02
Test MAE: ~0.08-0.12
Parameters: ~1-2M
Training Time: ~30 min (50 epochs)

Benefits:
- Better long-term consistency
- Reduced error accumulation
- Keyframe memory helps periodic motions
```

---

## 📝 Scientific Paper

### Structure Template (LaTeX)

```latex
\documentclass{article}
\usepackage{neurips_2024}
\usepackage{tikz, amsmath, graphicx}

\title{Improving Temporal Latent Space Models with \\
       Memory-Augmented Hierarchical Attention}

\author{Your Name \\ ENSPY, Université de Yaoundé I}

\begin{document}

\maketitle

\begin{abstract}
Long-term temporal consistency remains a challenge...
\end{abstract}

\section{Introduction}
...

\section{Related Work}
\subsection{Temporal Latent Space Models}
\subsection{Attention Mechanisms}
\subsection{Memory Networks}

\section{Proposed Method}
\subsection{Hierarchical Temporal Encoding}
\subsection{Temporal Transformer}
\subsection{Memory-Augmented Module}

\section{Experiments}
\subsection{Experimental Setup}
\subsection{Quantitative Results}
\subsection{Qualitative Analysis}
\subsection{Ablation Study}

\section{Conclusion}

\bibliographystyle{plain}
\bibliography{references}

\end{document}
```

### Figures à Inclure

1. Architecture diagram (TikZ)
2. Attention weights heatmaps
3. Reconstruction quality comparisons
4. Ablation study results (bar charts)

---

## 🧪 Tests

```bash
pytest tests/test_model.py

# Test Attention Layer
python -c "
from src.attention_mechanism import SimpleAttention
import tensorflow as tf
layer = SimpleAttention()
x = tf.random.normal((2, 10, 64))
context, weights = layer(x)
print(f'Context: {context.shape}')
print(f'Weights: {weights.shape}')
"

# Test Seq2Seq
python -c "
from src.lstm_attention_seq2seq import build_seq2seq_attention_model
model, _, _ = build_seq2seq_attention_model(50, 10)
model.summary()
"
```

---

## 🐛 Troubleshooting

### Out of Memory (Seq2Seq)

```python
# Réduire batch size et séquences
model.fit(..., batch_size=8)  # au lieu de 32
```

### TAP Training Slow

```bash
# Réduire epochs ou latent_dim
python src/tap_improvement.py
# Modifier epochs=30 au lieu de 50
# Modifier latent_dim=64 au lieu de 128
```

### LaTeX Paper Compilation

```bash
cd paper/
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

---

## 📚 Ressources

- [Énoncé TP5 (PDF)](./TP5_DL_5GI_2025_EN.pdf)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Neural Machine Translation (Bahdanau)](https://arxiv.org/abs/1409.0473)
- [TAP Paper (ArXiv:2102.05095)](https://arxiv.org/abs/2102.05095)
- [Memory Networks](https://arxiv.org/abs/1410.3916)
- [NeurIPS LaTeX Template](https://neurips.cc/Conferences/2024/PaperInformation/StyleFiles)

---

## 👥 Auteurs

**ENSPY - Université de Yaoundé I**  
FOMEKONG TAMDJI JONATHAN BACHELARD
Département de Génie Informatique - Promotion 5GI 2025

**Instructeurs**:
- Louis Fippo - louis.fippo@univ-yaounde1.cm

---

## ⚖️ Licence

Distribué sous la licence MIT. Voir [LICENSE](LICENSE) pour plus d'informations.