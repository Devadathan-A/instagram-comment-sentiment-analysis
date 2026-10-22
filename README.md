# Instagram Comment Sentiment Analysis

A local web application that analyses English Instagram-style comments from a CSV file and presents sentiment results in an interactive dashboard. It uses VADER and TextBlob sentiment scores alongside preprocessing and rule-based heuristics.

## Features

- Upload a CSV file through a Flask web interface.
- Identify a likely comment column from common column names.
- Filter comments to English using script, vocabulary, and language-marker heuristics.
- Classify comments as **Positive**, **Neutral**, or **Negative**.
- Combine VADER and TextBlob polarity scores.
- Display a heuristic confidence tier and a sarcasm flag.
- Show aspect-based insights for quality, price, delivery, service, and design.
- Visualise sentiment distribution and aspect results.
- Search/filter results and download analysed results as a CSV.

## Tech stack

- Python
- Flask
- NLTK (VADER, tokenization, stopwords, WordNet, and English word vocabulary)
- TextBlob
- pandas
- Matplotlib
- HTML, CSS, and JavaScript for the dashboard

## Requirements

- Python 3.10 or newer recommended
- Internet access on the first run so NLTK can download the language resources used by the application

## Run locally in VS Code

### 1. Open the project

Download or clone this repository, then open the project folder in VS Code.

### 2. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start the application

```bash
python app.py
```

On first launch, the program downloads the NLTK resources it needs. When the server starts, open **http://127.0.0.1:5000** in your browser if it does not open automatically. Stop the server with `Ctrl+C`.

## Input CSV format

The easiest format is a CSV with a column named `comment`:

```csv
comment
"The build quality feels solid and durable."
"Delivery was late and the package arrived damaged."
```

See [`sample_comments.csv`](sample_comments.csv) for a small synthetic example that you can use to try the app. The application also recognises several alternative column names, such as `text`, `review`, and `feedback`.

## How the analysis works

1. Load comments from the uploaded CSV.
2. Remove comments that match the app's irrelevance rules.
3. Apply an English-language heuristic filter.
4. Convert supported emoji to text, preprocess comments, and calculate VADER and TextBlob polarity scores.
5. Combine the scores using the weights currently defined in the code and classify each comment.
6. Apply rule-based sarcasm patterns and generate aspect-level summaries and charts.

## Important limitations

- This is a **lexicon/rule-based NLP application**, not a supervised model trained on a labelled Instagram dataset.
- Language detection, sarcasm detection, aspect matching, and the displayed confidence tier are heuristics; they can make mistakes. The confidence tier is based on score magnitude and is not a calibrated probability.
- The current version is designed primarily for English comments. Some valid English comments may be filtered out, and some non-English comments may pass the filter.
- For comments above 5,000, the app samples 5,000 comments for analysis.
- The Flask development server is intended for local demonstration, not direct public production deployment.

## Data and privacy

Use synthetic, public, or appropriately permissioned data. Do not commit private Instagram comments, personal information, access tokens, or other sensitive data to GitHub. The included sample CSV contains synthetic example comments.

## Future improvements

- Add automated tests and a labelled evaluation dataset.
- Evaluate precision, recall, and F1-score against human-labelled comments.
- Improve multilingual handling and sarcasm detection.
- Separate the inline HTML/CSS/JavaScript into template and static files as the app grows.

## Author

Created as an academic NLP project.
