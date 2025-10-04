# Cat vs Dog Image Classifier 🐱🐶

This project implements an image classifier for cats and dogs using deep learning and transfer learning (VGG19) in PyTorch. The solution covers loading data, training, model evaluation, and result visualization.

---

## 🚀 Features

- PyTorch-based deep learning pipeline
- Uses pre-trained VGG19 (transfer learning)
- Efficient data loading and augmentation
- Easy training & validation code flow
- Visualizations: sample predictions & confusion matrix

---

## 📁 Dataset Structure

Organize your images in this structure:

data/
Train/
Cat/
Dog/
Test/
Cat/
Dog/

Use `torchvision.datasets.ImageFolder`. In Colab, mount Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## 🛠️ Installation

Clone this repo, then install packages:
git clone https://github.com/yourusername/cat-dog-classifier.git
cd cat-dog-classifier
pip install torch torchvision numpy matplotlib seaborn pandas scikit-learn


---

## 📓 Usage

1. Upload your data in the folder structure above.
2. Open `cat_dog.ipynb` in Colab or Jupyter.
3. Mount your Google Drive if using Colab.
4. Run the cells as instructed for preprocessing, training, and evaluating the model.

---

## 🏗️ Model Details

- Backbone: VGG19 (ImageNet weights)
- Custom head:
  - Linear(25088 → 1024) → ReLU → Dropout(0.4) → Linear(1024 → 2) → LogSoftmax
- Optimizer: Adam
- Loss: CrossEntropyLoss

---

## 📊 Results

- High accuracy on demo/test data
- Includes confusion matrix and prediction visualization

---

## ✨ Example Output

*(Add your output images/graphs here)*

---

## 📚 References

- [VGG19 Original Paper](https://arxiv.org/abs/1409.1556)
- [PyTorch Documentation](https://pytorch.org/)

---



Happy Coding!




