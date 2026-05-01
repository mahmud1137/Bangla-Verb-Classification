# Bangla-Verb-Classification
This project focuses on classifying Bangla verbs into transitive and intransitive categories using Machine Learning and Large Language Models (LLMs). It analyzes linguistic patterns and contextual features to improve classification accuracy, supporting Bangla NLP research and applications.


## Project Overview
This project processes Bengali text data and applies machine learning pipelines to understand and classify syntactic sentence structures. The notebook is set up for Google Colab and covers everything from raw data cleaning and custom Bengali tokenization to model evaluation. It includes setups for both traditional Machine Learning (via Scikit-Learn) and advanced Deep Learning models (via Hugging Face Transformers).

## Features & Text Preprocessing
Handling Bengali text requires specific preprocessing steps to ensure the models train on high-quality data. The custom data cleaning pipeline includes:
* **Noise Reduction:** Removing URLs, numbers, English characters, and standard punctuation.
* **Emoji Stripping:** Utilizing Unicode-based regular expressions to filter out emojis and symbols.
* **Custom Bengali Tokenization:** Using RegEx (`[\u0980-\u09FF]+`) to extract purely Bengali word tokens from the cleaned sentences.
* **Data Formatting:** Dropping empty/unnamed columns from the imported Excel dataset and dropping missing values.

## Modeling Approaches
The notebook is equipped to handle multiple modeling strategies:

**1. Traditional Machine Learning:**
* **Feature Extraction:** Converting Bengali text into numerical representations using Scikit-Learn's `TfidfVectorizer`.
* **Algorithms:** Includes setups for `LinearSVC` (Support Vector Classifier) and `LogisticRegression`.
* **Label Encoding:** Transforming 'Transitive' and 'Intransitive' string labels into numerical formats.

**2. Deep Learning (Transformers):**
* Integrates the Hugging Face `transformers` and `datasets` libraries.
* Utilizes `AutoTokenizer` and `DataCollatorWithPadding` for dynamic sequence padding.
* Prepares the dataset for fine-tuning state-of-the-art models via `AutoModelForSequenceClassification` and `Trainer` API (ideal for models like mBERT or BanglaBERT).

## Evaluation Metrics
Model performance is evaluated using standard classification metrics:
* **Accuracy Score**
* **Classification Report:** Detailed Precision, Recall, and F1-Score for both Transitive and Intransitive classes.
* **Confusion Matrix:** To carefully analyze false positives and false negatives in the classification.

## Technologies & Libraries Used
* **Data Manipulation:** `pandas`, `numpy`, `re` (Regular Expressions)
* **Traditional ML:** `scikit-learn` (TF-IDF, SVC, Logistic Regression, LabelEncoder)
* **Advanced NLP:** `transformers`, `datasets` (Hugging Face)
* **Basic Tokenization:** `nltk`
* **Data Visualization:** `matplotlib.pyplot`, `seaborn` (for Confusion Matrix heatmaps)
