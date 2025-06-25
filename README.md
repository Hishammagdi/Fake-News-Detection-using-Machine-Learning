# Fake-News-Detection-using-Machine-Learning

📌 Project Overview
This project focuses on classifying news articles as real or fake using machine learning and natural language processing (NLP) techniques. The aim is to develop a robust pipeline that can detect misinformation effectively.

📂 Dataset
- Source: Kaggle Fake News Dataset (https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
- Features: title, text, subject, date, label
- Target: label (1 = Real, 0 = Fake)
🔧 Technologies Used
- Programming Language: Python
- Libraries:
  - NLP: NLTK, re
  - ML: scikit-learn
  - Deep Learning: TensorFlow, Keras (for extensions)
  - Visualization: matplotlib, seaborn

🧹 Data Preprocessing
- Dropped irrelevant columns (subject, date)
- Removed nulls and duplicates
- Converted text to lowercase
- Removed punctuation, stopwords
- Applied tokenization and lemmatization

  ✨ Feature Engineering
- TF-IDF Vectorization (Unigrams and Bigrams)
- Alternative: CountVectorizer
- Dimensionality reduction (optional): TruncatedSVD / PCA
🤖 Model Building
Classical ML Models:
- Logistic Regression
- Naive Bayes (MultinomialNB)
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest

Deep Learning (Optional Extension):
- LSTM using Keras (for sequential input)
- RNN and GRU comparisons
📊 Evaluation Metrics
- Accuracy
- Precision, Recall, F1-Score
- Confusion Matrix
- ROC Curve (optional)
🧪 Results
Model               | Accuracy
------------------- | --------
Logistic Regression | 94.5%
Naive Bayes         | 95.0%
SVM                 | 96.3%
Decision Tree       | 93.0%
LSTM (Keras)        | ~97.1%

📈 Visualizations
- Word cloud for most frequent terms
- n-gram analysis (bigrams, trigrams)
- Class distribution histogram
🧠 Future Enhancements
- Use pre-trained transformer models (e.g., BERT)
- Add real-time news scraping + classification
- SHAP or LIME for explainable AI
- Deploy as a browser extension
  
✅ Conclusion
Successfully developed a machine learning pipeline to detect fake news with high accuracy. The project demonstrates data preprocessing, NLP feature extraction, model training and evaluation, and provides a base for further deep learning and deployment work.
