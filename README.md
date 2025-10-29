## 📰 Fake News Detection using Machine Learning
# 📌 Project Overview
This project aims to detect whether a news article is Fake or Real using Machine Learning and Natural Language Processing (NLP) techniques.
Fake news spreads quickly on the internet, especially through social media. To stop misinformation, we train a model that learns the patterns and words used in fake news and predicts if a new article is genuine or not.

# 🧠 Objective
To build a machine learning model that classifies news as “Fake” or “Real” based on its text content.

# 🧩 Steps Involved
#  1. Data Collection


The dataset contains thousands of news articles labeled as Fake or Real.


Each record has the title, text, and label of the news.


#  2. Data Cleaning & Preprocessing


Remove punctuation, special symbols, and stopwords.


Convert all words to lowercase.


Tokenize and clean text to make it ready for analysis.


#  3. Text Vectorization


Convert text into numerical form using TF-IDF Vectorizer (Term Frequency–Inverse Document Frequency).


This helps the machine learning model understand the importance of words.


#  4. Model Building


Used algorithms like Logistic Regression or Naive Bayes for classification.


The model learns from training data to detect patterns that differentiate fake and real news.


#  5. Model Evaluation


Tested the model on unseen (test) data.


Evaluated using Accuracy, Precision, Recall, F1-score, and Confusion Matrix.


#  6. Prediction


Input a new article → model predicts whether it’s Fake or Real.



#  📊 Example Workflow
StepTaskDescription1Load DatasetImport fake/real news data2Clean TextRemove unwanted words and symbols3VectorizeConvert text into numeric form4Train ModelFit ML algorithm on training data5EvaluateTest accuracy and confusion matrix6PredictInput a new news article to check its truth

#  🧾 Results


Achieved high accuracy on test data (around 90% or above depending on dataset).


The model successfully detects fake and real news with good reliability.



#  🛠️ Technologies Used
Tool / LibraryPurposePythonProgramming languagePandas, NumPyData handling & analysisScikit-LearnMachine learning & model buildingNLTKText preprocessing (stopwords, tokenization)Matplotlib / SeabornData visualizationTF-IDF VectorizerText to numerical conversion

#  🚀 How to Run the Project

Install dependencies
pip install pandas numpy scikit-learn nltk matplotlib seaborn



Run the Jupyter Notebook
jupyter notebook fake-realnews-detection.ipynb



Execute all cells — the notebook will train the model and show results.



# 💡 Future Improvements


Use deep learning models like LSTM or BERT for better accuracy.


Add a Flask or Django web app for live predictions.


Include more datasets for stronger generalization.


Integrate with Power BI or Streamlit dashboard for visual results.



#  📘 Output Example
After training, the model can output results like:
Input: "Government announces new education policy..."
Prediction: REAL NEWS ✅

Input: "Aliens landed in New York last night..."
Prediction: FAKE NEWS ❌


#  👩‍💻 Author
Developed by: Mohini Pasi

