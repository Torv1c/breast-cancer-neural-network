# Deep Learning to FPGA: Hardware-Aware Neural Network Synthesis

## Project Overview
This repository demonstrates a complete end-to-end pipeline for training a Deep Learning model in Python and preparing it for **hardware synthesis on an FPGA** (Field-Programmable Gate Array). 

While the dataset focuses on Breast Cancer prediction (Malignant/Benign) to prove the clinical accuracy of the model, the core engineering achievement of this project is bridging the gap between high-level AI frameworks (TensorFlow/Keras) and low-level hardware constraints (Fixed-Point Arithmetic and VHDL).

This project highlights skills in **Edge AI, Model Quantization, Matrix Mathematics, and System Integration.**

## Tech Stack
* **AI & Data Science:** `Python`, `TensorFlow/Keras`, `Scikit-Learn`, `NumPy`, `Pandas`
* **Hardware Targeting:** Fixed-Point Quantization (Q2.14 format), VHDL Testbench Generation, Custom Adapter Patterns (`.pkl` serialization).
* **Validation & Visualization:** `Matplotlib`, `Seaborn`

## Methodology & Engineering Pipeline
Unlike standard Data Science workflows, this project extends into hardware integration through 4 distinct phases:

### 1. Data Processing & Model Training (The "Golden Model")
* Handled missing values and normalized 30 medical features using `MinMaxScaler` to prevent hardware overflow.
* Built and trained a Multilayer Perceptron (MLP) with a `[30-5-1]` topology using Keras. 
* Achieved high clinical accuracy, validating the model with a Confusion Matrix.

### 2. Weight Extraction & Mathematical Parity (The Stress Test)
* **The Challenge:** FPGAs do not run Keras. They compute raw matrix multiplications.
* **The Solution:** Extracted the raw weights ($W$) and biases ($b$) from the hidden and output layers.
* Recreated the entire forward propagation mathematically using pure `numpy.dot()` and manual activation functions (ReLU, Sigmoid).
* **Validation:** Ran a cross-validation script across 106 test patients, proving the manual math matched the Keras backend with an absolute error margin of $< 10^{-7}$ (solely due to floating-point rounding noise).

### 3. Hardware Adapter Pattern (Keras $\to$ Scikit-Learn)
* Developed a custom Python Adapter class to encapsulate the Keras weights into a structure native to `Scikit-Learn` (`coefs_`, `intercepts_`).
* Applied the "Bias Trick" (Augmented Matrix with $x_0 = -1$) to format the topology exactly as required by legacy VHDL translators.

### 4. Fixed-Point Quantization for VHDL (Q2.14)
* Converted the normalized floating-point test data into **16-bit binary words** using Q2.14 Fixed-Point arithmetic (2 bits for integer, 14 for fraction).
* Generated automated `.txt` files containing the binary strings properly formatted for direct injection into a **VHDL Testbench** (`after 10ns,` syntax).

## Visual Validation & Results

This section presents the visual proof of both the model's clinical accuracy and the mathematical parity required for successful hardware synthesis.

### Model Accuracy (Clinical Validation)
The trained Neural Network achieved high precision in distinguishing between malignant and benign tumors.


<img width="505" height="470" alt="confusion_matrix" src="https://github.com/user-attachments/assets/b202aba7-cac1-4d94-9f07-b49d9ec99d04" />


* **Figure 1: Confusion Matrix.** This matrix (with Portuguese labels) validates the model's performance on the 106 test patients. Legend: `Maligno` = Malignant, `Benigno` = Benign, `Predito` = Predicted, `Real` = True.

### Mathematical Parity (Hardware Validation)
**This is the critical engineering milestone of the project.** The graph below plots the absolute error between the Keras backend output and our custom-built matrix algebra implementation.

<img width="1089" height="490" alt="error_graph" src="https://github.com/user-attachments/assets/a7b9920e-66d8-4f14-9df2-a3ad4bb37e9f" />


* **Figure 2: Parity Error Analysis.** This plot shows the absolute difference in predictions for all 106 test patients. The error is consistently below $10^{-7}$, proving that our mathematical implementation (destined for FPGA) is computationally identical to the high-level framework with negligible noise due to floating-point rounding.

---
**Author:** Victor Cesar de Mecê Prando
**LinkedIn:** [victor-prando1](https://www.linkedin.com/in/victor-prando1/)
