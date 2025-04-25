# Policy Sentiment Dashboard

The **Policy Sentiment Dashboard** is a web-based application designed to analyze public sentiment toward government policies using the TweetEval dataset. It leverages a BERT-based model to classify tweets as positive, neutral, or negative and provides interactive visualizations, including sentiment trends, word clouds, and topic modeling. Built with Flask, Bootstrap 5, and Python, the dashboard offers a modern, responsive UI for exploring policy-related sentiments.

## Features
- **Policy Filtering**: Analyze tweets for policies like GST, Digital India, Make in India, and Swachh Bharat.
- **Sentiment Analysis**: View sentiment metrics (positive, neutral, negative) and categorized tweet tables.
- **Interactive Filters**: Search by keyword, filter by sentiment, and select date ranges.
- **Visualizations**: Display sentiment trends and word clouds for key concerns.
- **Topic Modeling**: Extract key topics for each policy using LDA.
- **Data Export**: Download filtered tweets as CSV.
- **Modern UI**: Responsive design with gradient headers, hover effects, tooltips, and a loading spinner.

## Tech Stack
- **Backend**: Flask (Python)
- **Frontend**: HTML, Bootstrap 5, Custom CSS, JavaScript
- **Data Processing**: Pandas, NumPy, NLTK, spaCy
- **Machine Learning**: BERT (Hugging Face Transformers) with 0.6657 accuracy
- **Visualization**: Matplotlib (trends), WordCloud
- **Dataset**: TweetEval sentiment dataset (preprocessed)

## Prerequisites
- **Python 3.8+** (recommended: 3.10)
- **pip** (Python package manager)
- **Virtualenv** (recommended)
- **Git** (for cloning the repository)
- A modern web browser (e.g., Chrome, Firefox)
- Internet access (for downloading TweetEval and Hugging Face models)

## Installation

Follow these steps to set up the project locally.

### Step 1: Clone the Repository
Clone the project from GitHub:

git clone https://github.com/tejs21/policy-sentiment-dashboard.git
cd policy-sentiment-dashboard

### Step 2: Create a Virtual Environment
Create and activate a virtual environment to manage dependencies.

# On Windows
python -m venv venv
.\venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

You should see (venv) in your terminal, indicating the virtual environment is active.

### Step 3: Install Dependencies
Install the required Python packages listed in requirements.txt.

pip install -r requirements.txt

See requirements.txt (#requirements) for a list of dependencies.

### Step 4: Download and Preprocess the TweetEval Dataset
The project uses the TweetEval sentiment dataset, which is downloaded and preprocessed to focus on policy-related tweets.
Download TweetEval
Run the script to download the TweetEval sentiment dataset and save it as CSV files.

python data/dummy/download_tweeteval.py

This creates:
data/dummy/tweeteval_train.csv
data/dummy/tweeteval_val.csv
data/dummy/tweeteval_test.csv

Verify Output:
Check the terminal output for sample data and label distribution (0=Negative, 1=Neutral, 2=Positive). Example:
Label distribution (train):
2    20000
1    15000
0    10000
Name: label, dtype: int64

Preprocess Dataset
Preprocess the dataset to filter policy-related tweets and clean the text.

python data/dummy/preprocess_dummy.py

This generates:
data/processed/filtered_tweeteval.csv (filtered policy-related tweets)

data/processed/preprocessed_tweeteval.csv (cleaned text with lemmatization)

Verify Output:
Check the terminal for:

Filtered X policy-related tweets.
Preprocessing complete. Saved to data/processed/preprocessed_tweeteval.csv

Run the following to inspect the preprocessed dataset:

python -c "import pandas as pd; df = pd.read_csv('data/processed/preprocessed_tweeteval.csv'); print('Columns:', df.columns); print('Label counts:', df['label'].value_counts()); print('Policy counts:', df['policy'].value_counts())"

Expected Output:

Columns: Index(['text', 'label', 'policy', 'cleaned_text'], dtype='object')
Label counts:
2    4000
1    3500
0    2500
Name: label, dtype: int64
Policy counts:
GST              4000
Digital India    3500
Make in India    2500
Swachh Bharat    2000
Name: policy, dtype: int64

### Step 5: Train the BERT Model
Train a BERT model for sentiment classification using the preprocessed dataset.

python src/train_bert.py


This:
Loads data/processed/preprocessed_tweeteval.csv (or data/dummy/preprocessed_dummy_tweets.csv if specified).

Trains a BERT model (bert-base-uncased) for 3 epochs.

Saves the model and tokenizer to models/bert_sentiment/.

Prints validation accuracy (e.g., 0.6657).

Verify Output:
Check the terminal for:
Validation Accuracy: 0.6657

Note: Training may take several hours depending on your hardware. Ensure you have a GPU (optional) or sufficient CPU resources.


### Step 6: Extract Themes
Run the theme extraction script to test topic modeling and keyword extraction.

python src/theme_extraction.py

This processes data/dummy/preprocessed_dummy_tweets.csv for the GST policy and outputs keywords and topics.
Verify Output:
Check the terminal for:
GST Themes:
Keywords: [('tax reform', 150), ('digital payment', 120), ...]
Topics: ['Topic 1: tax, reform, policy', 'Topic 2: digital, india, tech', ...]

## Running the Project
### Step 1: Activate the Virtual Environment
Ensure the virtual environment is active.

# On Windows
.\venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate

Step 2: Run the Flask Application
Start the Flask server.

python src/app.py

Expected Output:
 * Serving Flask app 'app'
 * Running on http://127.0.0.1:5000 (Press CTRL+C to quit)

### Step 3: Access the Dashboard
Open a web browser and navigate to:
http://127.0.0.1:5000

The dashboard displays:
A gradient header with the title.
A filter form for policy, search, sentiment, and date range.
Summary metrics (total tweets, sentiment percentages).
Sentiment trends plot.
Categorized tweet tables (negative, neutral, positive).
Word cloud for key concerns.
List of topics for the selected policy.

### Step 4: Interact with the Dashboard
Select Policy: Choose a policy (e.g., GST).
Search Tweets: Enter keywords (e.g., “tax”).
Filter Sentiment: Select “Negative”, “Neutral”, “Positive”, or “All”.
Set Date Range: Choose start and end dates.
Apply Filters: Click “Apply Filters” to update.
Reset Filters: Click “Reset Filters” to clear.
Download Data: Click “Download CSV” to export tweets.










