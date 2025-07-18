# Galaxy-Classification
## Galaxy Morphology Classification (Galaxy Zoo 2 Dataset – 28,790 Images)

## Table of Contents
- [Data](#data)
- [Directory Structure](#directory-structure)
- [Data Optimization Pipeline](#data-optimization-pipeline)
- [Model Architecture](#model-architecture)
- [Libraries Used](#libraries-used)
- [Final Model Performance](#final-model-performance)
- [Classification Report](#classification-report)
- [Sample Predictions](#sample-predictions)


## Data

This notebook trains a deep learning model to classify galaxy images from the Galaxy Zoo 2 dataset (28,799 images) into five distinct morphological types. The model uses data augmentation, convolutional layers, and regularization to improve accuracy and reduce overfitting.

- **Cigar-shaped smooth**
- **Completely round smooth**
- **Edge-on**
- **In between smooth**
- **Spiral**

---

## Directory Structure
```
├── galaxyClassification.ipynb
├── images/
├── Training images/
├── Test images/
└── Samples/
```
---

## Libraries Used

- `TensorFlow 2.x`
- `Keras`
- `Matplotlib`, `NumPy`
- `sklearn` (for metrics)

---

## Data Optimization Pipeline

To ensure efficient and balanced training, the following preprocessing steps were applied:

- **Data Cleaning**: Removed corrupted or mislabeled images.
- **Resizing**: All images resized to 250x250 for uniformity.
- **Augmentation**:
    - Horizontal and vertical flipping
    - Random rotations and zoom
    - Brightness adjustments
- **Balancing**:
    - Undersampled overrepresented classes
    - Oversampled minority classes (via augmentation)
- **Normalization**: Pixel values scaled to [0, 1]
- **Shuffling & Batching**:
    - Dataset shuffled before each epoch
    - Batched using `batch_size=32` for faster training


---

## Model Architecture

- **Input shape**: 250x250 RGB images
- **Preprocessing**:
  - Data augmentation (flip, rotation, zoom, etc.)
  - Normalization (Rescaling by 1./255)
- **Conv layers**: 4 blocks with increasing filters (16→64)
- **Pooling**: MaxPooling after each Conv2D
- **Regularization**:
  - `Dropout(0.2)`
  - Tried `BatchNormalization`, but didn’t yield significant gain
- **Output**: Softmax layer with 5 classes
- **Pooling method**: Used `Flatten()` instead of GlobalAveragePooling2D due to better results


---


## Final Model Performance

After training, the model achieved **94.18%** accuracy on the test set, with strong precision, recall, and F1-scores across all classes.

### Classification Report:

| Galaxy Type               | Precision | Recall | F1-score | Support |
|---------------------------|-----------|--------|----------|---------|
| Cigar-shaped smooth       | 0.98      | 0.96   | 0.97     | 1044    |
| In between smooth         | 0.94      | 0.89   | 0.91     | 1028    |
| Completely round smooth   | 0.92      | 0.97   | 0.94     | 1075    |
| Edge-on                   | 0.90      | 0.93   | 0.92     | 497     |
| Spiral                    | 0.95      | 0.95   | 0.95     | 995     |

**Test Accuracy:** `0.9418`

---
## Sample Predictions

You can use the `Samples/` folder to test new galaxy images.  
The model predicts the galaxy type with high confidence in most cases.




