# Fake-News-Detection
# 🧠 Fake News Detection using PassiveAggressiveClassifier

This project is a machine learning-based fake news detection system that uses Natural Language Processing (NLP) and the PassiveAggressiveClassifier to distinguish between real and fake news articles.

## 📂 Dataset
The dataset is a CSV file (`news.csv`) containing:
- `title`: Title of the news article
- `text`: Full text of the news article
- `label`: Either `FAKE` or `REAL`

## 🚀 Features
- TF-IDF vectorization of text
- Train/test split (80/20)
- Model training using PassiveAggressiveClassifier
- Evaluation using accuracy and confusion matrix
- Visualization using seaborn and matplotlib

## 🛠️ Tech Stack
- Python 3.x
- Pandas
- scikit-learn
- Matplotlib
- Seaborn

## 📊 Model Overview
The **PassiveAggressiveClassifier** is used for online learning and is particularly suited for large-scale and real-time classification tasks.

## 📈 Results
The model achieves high accuracy and shows a clear distinction between real and fake news, as seen in the confusion matrix visualization.

## ▶️ How to Run

1. Clone this repo:
   ```bash
   git clone https://github.com/yourusername/FakeNewsDetection.git
   cd FakeNewsDetection
   ```

2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook:
   ```bash
   jupyter notebook FakeNewsDetection.ipynb
   ```

## 📁 Files
- `FakeNewsDetection.ipynb`: Main project notebook
- `news.csv`: Dataset (not included – you must add it manually)
- `FakeNewsDetection_Presentation.pptx`: Presentation
- `README.md`: Project documentation

## 📜 License
This project is open-source and available under the MIT License.

## 🙌 Acknowledgements
Dataset source: [Kaggle or other publicly available dataset]
