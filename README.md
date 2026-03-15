# 🎬 Sentiment Analysis — LSTM + Debugging Journey

## Overview
NLP project that classifies IMDB movie reviews as positive 
or negative using an LSTM neural network. Includes a full 
debugging case study showing how to fix unstable training.

## Results
| Version        | Accuracy | Issue                        |
|----------------|----------|------------------------------|
| First attempt  | 58.98%   | Unstable training (fixed!)   |
| Fixed version  | 85.12%   | Target achieved ✅           |

## What Makes This Project Unique
This project documents a REAL debugging journey:
- First model failed at 58.98% (barely above random)
- Diagnosed the problem: learning rate too high + model too small
- Fixed with: lower lr, bigger layers, early stopping
- Result: jumped to 85.12% ✅

This demonstrates real ML engineering skills —
not just running code that works, but fixing code that breaks.

## Model Architecture
```
Input → Embedding(10000, 128) → LSTM(128)
    → Dropout(0.3) → Dense(64, ReLU) → Dense(1, Sigmoid)
```

## Key Concepts Demonstrated
- Text tokenization and padding
- Word embeddings (learned representations)
- LSTM for sequential memory
- Early stopping (stopped at epoch 12, restored best weights)
- Learning rate tuning (0.001 → 0.0005)
- Training curve analysis and overfitting detection

## Debugging Case Study
```
Problem:  val_accuracy peaked at 77.8% (epoch 2)
          then CRASHED to 61.8% (epoch 3)
          
Diagnosis: Learning rate too high → unstable training
           Model too small → underfitting
           
Fix 1: lr 0.001 → 0.0005
Fix 2: LSTM units 64 → 128
Fix 3: Embedding dim 64 → 128  
Fix 4: Added EarlyStopping(patience=3)

Result: Stable training, 85.12% accuracy ✅
```

## Tech Stack
Python, TensorFlow, Keras, NumPy, Matplotlib

## How to Run
1. Open `sentiment_lstm.ipynb` in Google Colab
2. Runtime → Run All (GPU recommended)
3. IMDB dataset downloads automatically

## Sample Predictions
```
"This movie was absolutely brilliant"  → POSITIVE (62.8%)
"Terrible film, complete waste"        → NEGATIVE (95.5%)
"Started great but became boring"      → NEGATIVE (87.2%)
```
```

**requirements.txt:**
```
tensorflow
numpy
matplotlib
```

**Structure:**
```
sentiment-analysis-lstm/
├── README.md
├── requirements.txt
└── sentiment_lstm.ipynb
