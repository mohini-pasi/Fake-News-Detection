# 📰 Project: Fake vs Real News Detection
# 🎯 Goal

The aim of this project is to build a machine learning model that can identify whether a given news article is fake or real, based on its textual content.

# ⚙️ Step-by-Step Explanation
# 1. Dataset Loading

The dataset usually contains:

Title → The headline of the article

Text → The full article content

Label → “FAKE” or “REAL”

Example:

Title	Text	Label
Trump signs new law	The U.S. president...	REAL
Celebrity endorses miracle cure	Social media claims...	FAKE

# 2. Text Cleaning and Preprocessing

To make text ready for machine learning:

Convert all words to lowercase.

Remove punctuation, numbers, and special symbols.

Remove stopwords (common words like the, is, at).

Perform stemming or lemmatization (reduce words to base form, e.g., “running” → “run”).

This helps reduce noise and focus only on meaningful words.

# 3. Feature Extraction (Vectorization)

Since machine learning models can’t read text directly, we convert text to numbers using:

CountVectorizer → counts word frequencies.

TF-IDF Vectorizer (Term Frequency – Inverse Document Frequency) → gives weight to important words that are not too common.

👉 In this notebook, TF-IDF Vectorizer is used (which performs better).

# 4. Splitting Data

The dataset is divided into:

Training set (80%)

Testing set (20%)

So we can train on one part and test on unseen data.

# 5. Model Training

Several machine learning algorithms are usually tested:

Algorithm	Description

Logistic Regression	Works very well for text classification

Naive Bayes (MultinomialNB)	Simple and fast for text data

Random Forest	Ensemble model with many decision trees

Support Vector Machine (SVM)	Works well with high-dimensional data like text

K-Nearest Neighbors (KNN)	Slower and less effective for large text data



# 📊 Typical Results

In the original notebook and similar Kaggle datasets, the accuracy of models is approximately:

Algorithm	Average Accuracy

Logistic Regression	⭐ ~92–95%

Multinomial Naive Bayes	~88–90%

Random Forest	~85–88%

SVM (Linear)	~91–93%

KNN	~70–75%

# ✅ Best performing algorithm:

Logistic Regression with TF-IDF Vectorizer usually gives the highest and most stable accuracy (around 94–95%) for fake news detection.

# 🧠 Why Logistic Regression Works Best

Text data after TF-IDF transformation is high-dimensional but sparse.

Logistic Regression handles this efficiently.

It’s simple, interpretable, and avoids overfitting better than Random Forest or KNN for text.

Naive Bayes is fast but slightly less accurate.

# 🔍 Example Prediction
```bash
text = ["Aliens landed in New York last night!"]

pred = model.predict(vectorizer.transform(text))

print(pred)  # Output: ['FAKE']
```

# Output → FAKE 🚫

🛠 Libraries Used

pandas → Data handling

numpy → Numerical operations

sklearn → Machine Learning (LogisticRegression, TF-IDF, train_test_split)

nltk → Text preprocessing (stopwords, tokenization)

matplotlib / seaborn → Visualization

# 🧾 Summary

Step	Description

Preprocess	Clean and prepare text

Vectorize	Convert text → numeric TF-IDF

Train	Logistic Regression model

Evaluate	Accuracy ≈ 95%

Predict	Classify new news as Fake/Real

#  🚀 How to Run the Project

Install dependencies

```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn
```


Run the Jupyter Notebook
```bash
jupyter notebook fake-realnews-detection.ipynb
```



Execute all cells — the notebook will train the model and show results.



# 💡 Future Improvements


Use deep learning models like LSTM or BERT for better accuracy.


Add a Flask or Django web app for live predictions.


Include more datasets for stronger generalization.


Integrate with Power BI or Streamlit dashboard for visual results.




#  👩‍💻 Author
Developed by: Mohini Pasi

