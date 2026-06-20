## 📌 Project Overview

This project focuses on building, experimenting, and optimizing a Convolutional Neural Network (CNN) for the CIFAR-10 image classification problem.

The goal of this project was not only to achieve higher accuracy but to understand how different CNN architectural choices influence model performance.

The project explores:

- Custom TensorFlow data pipelines
- CNN architecture design
- Batch Normalization
- Max Pooling vs Strided Convolutions
- Functional API vs Model Subclassing
- Feature map visualization
- Controlled ablation experiments

The model improved from a baseline accuracy of:

```
58%
```

to:

```
72.65% Validation Accuracy
```

---

# 📂 Project Structure

```
cifar_cnn_classifier/

│
├── data/
│   ├── raw/
│   │   ├── images/
│   │   ├── sample.jpg
│   │   └── trainLabels.csv
│   │
│   └── processed/
│
├── notebooks/
│   └── 01_model_building.ipynb
│
├── plots/
│
├── src/
│   ├── model_class.py
│   ├── model_function.py
│   └── utils.py
│
├── README.md
├── RETROSPECTIVE.md
└── reqs.txt
```

---

# ⚙️ Data Pipeline

A custom data pipeline was created instead of using pre-built CIFAR loaders.

The pipeline loads:

```
images/
      |
      ↓
trainLabels.csv
      |
      ↓
TensorFlow Dataset
```

## Features

### Dynamic Label Mapping

The CSV file maps image IDs to class labels.

Example:

```
id,label

1,airplane
2,automobile
3,bird
```

---

## Stratified Dataset Split

The dataset is divided into:

```
80% Training
20% Validation
```

while maintaining equal representation of all 10 CIFAR classes.

---

## Performance Optimization

TensorFlow optimizations used:

```python
dataset.cache()

dataset.prefetch(
    tf.data.AUTOTUNE
)
```

Benefits:

- Faster training
- Reduced disk bottleneck
- Better GPU utilization

---

# 🏗️ Model Development

The project contains two CNN implementations:

```
src/

├── model_class.py

└── model_function.py
```

---

# 1. CNN Model Subclassing

File:

```
src/model_class.py
```

Initial implementation using:

```python
tf.keras.Model
```

Advantages:

- More control
- Object-oriented structure

Challenges:

- Handling training flags
- BatchNorm behaviour
- Layer connection debugging

---

# 2. Functional API CNN

File:

```
src/model_function.py
```

The final implementation uses the Keras Functional API.

Architecture:

```
Input Image

↓

Double Conv Block
32 Filters

↓

Double Conv Block
64 Filters

↓

Double Conv Block
128 Filters

↓

Dense Classifier

↓

10 Classes
```

The architecture was designed to easily modify:

- Batch Normalization
- Pooling
- Strides
- Padding

---

# 🔬 Ablation Study

Experiments were performed using the generalized CNN.

Training:

```
Epochs: 10
Dataset: CIFAR-10
```

| Experiment | BatchNorm | MaxPool | Strides | Padding | Accuracy |
|---|---|---|---|---|---|
| Case 1 | ❌ | ✅ | 1,1 | same | 70.25% |
| Case 2 ⭐ | ✅ | ✅ | 1,1 | same | **72.65%** |
| Case 3 | ✅ | ❌ | 2,2 | same | 64.25% |
| Case 4 | ✅ | ❌ | 2,2 | valid | 64.05% |

---

# 📈 Observations

## Batch Normalization

Adding Batch Normalization improved:

```
70.25%

↓

72.65%
```

because it:

- Stabilizes gradients
- Improves convergence
- Allows better feature learning

---

## Max Pooling vs Strided Convolution

Results:

```
Max Pooling

72.65%


Strided Convolution

64%
```

Max pooling performed better because it:

- Preserves important features
- Provides translation invariance
- Reduces spatial size effectively

---

# 📊 Visualization

File:

```
src/utils.py
```

contains visualization utilities.

---

## Training Curves

Function:

```python
plot_metrics()
```

Generates:

- Accuracy curves
- Loss curves

Used for identifying:

- Overfitting
- Underfitting
- Training behaviour

---

## Feature Maps

Function:

```python
plot_feature_maps()
```

Used to inspect CNN internal representations.

Comparison:

```
Baseline CNN

vs

Optimized CNN
```

The optimized model learned richer features:

Early layers:

```
Edges
Textures
Colours
```

Deep layers:

```
Shapes
Object patterns
Class features
```

---

# 🚀 Future Improvements

## 1. Regularization

Add:

```python
Dropout(0.5)
```

to reduce overfitting.

---

## 2. Improve Model Subclass Version

Further debugging:

- Training flag handling
- BatchNorm behaviour
- Compare Functional API vs Subclassing

---

# 🧠 Key Learnings

Through this project:

✅ Built a custom TensorFlow pipeline  
✅ Designed CNN architectures  
✅ Performed ablation studies  
✅ Compared architectural choices  
✅ Understood Batch Normalization  
✅ Visualized CNN feature extraction  
✅ Practiced ML experimentation workflow  

---

# 📌 Final Results

| Model | Validation Accuracy |
|-|-|
| Baseline CNN | 58.00% |
| Optimized CNN | **72.65%** |

---

# 🛠️ Installation

Clone repository:

```bash
git clone <repository-url>
```

Install dependencies:

```bash
pip install -r reqs.txt
```

---

# ▶️ Running the Project

Run notebook:

```
notebooks/01_model_building.ipynb
```

or execute scripts from:

```
src/
```