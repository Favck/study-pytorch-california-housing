# California Housing Price Prediction (Regression)

This project focuses on predicting median house values in California using a custom deep learning architecture built with **PyTorch**. 

---

## 🧠 Model Architecture

The neural network is a Multi-Layer Perceptron (MLP) implemented using the `nn.Sequential` container:
* **Input Layer:** Dynamically scales to match the 8 dataset features (`D_in = 8`)
* **Hidden Layers:** 2 fully connected layers with 64 units each and `ReLU` activation functions
* **Output Layer:** 1 linear neuron for continuous value regression

---

## 📊 Results & Performance

* **Loss Function:** Mean Squared Error (`nn.MSELoss`)
* **Test Dataset MSE:** **0.6379**

The model successfully handles non-linear relationships between socioeconomic and geographic factors to evaluate real estate trends across California.

---

## 🛠️ Installation & Setup

1. Clone this repository:
   ```bash
   git clone https://github.com
   cd study-pytorch-california
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
