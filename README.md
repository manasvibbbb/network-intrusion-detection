# 🛡️ Network Intrusion Detection System

An AI-powered network intrusion detection system using machine learning to identify and classify cyber threats in real-time. Built with TensorFlow, Scikit-learn, and Streamlit.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## 🌟 Features

- **Real-time** detection of multiple network attack types (DoS, Probe, R2L, U2R).
- **Multiple ML models** support (e.g., Random Forest, XGBoost, Neural Networks).
- **Interactive Streamlit dashboard** for training, evaluation and inference.
- **Model explainability** using SHAP and LIME.
- **Sample data generator** to quickly create realistic network traffic samples.
- Optional **authentication** for controlled access to the dashboard.

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/manasvibbbb/network-intrusion-detection.git
cd network-intrusion-detection

### 2. (Recommended) Create and Activate a Virtual Environment
Windows:

python -m venv venv
venv\Scripts\activate
macOS / Linux:

python3 -m venv venv
source venv/bin/activate

3. Install Dependencies
bash
pip install -r requirements.txt

4. Run the Streamlit App
From the project root:


cd src
streamlit run dashboard/app.py
Then open your browser at:
http://localhost:8501 (if it does not open automatically)
