import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('vader_lexicon')
import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords
from collections import Counter
from textblob import TextBlob
! pip install PyPDF2
from nltk.stem import PorterStemmer
from nltk.sentiment import SentimentIntensityAnalyzer
import PyPDF2
import matplotlib.pyplot as plt
import seaborn as sns
import string
import spacy

# Load spaCy model for Named Entity Recognition
nlp = spacy.load("en_core_web_sm")

# Initialize the SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

def plot_emotion_graph(sentiment_data, output_path):
    """Plot emotion graph based on SentimentIntensityAnalyzer's polarity."""
    paragraphs = [f'Paragraph {i+1}' for i in range(len(sentiment_data))]
    positive_scores = [score['pos'] for score in sentiment_data]
    neutral_scores = [score['neu'] for score in sentiment_data]
    negative_scores = [score['neg'] for score in sentiment_data]
    overall_score = [score['compound'] for score in sentiment_data]

    plt.figure(figsize=(10, 6))
    sns.lineplot(x=paragraphs, y=positive_scores, label="Positive", color="g")
    sns.lineplot(x=paragraphs, y=neutral_scores, label="Neutral", color="b")
    sns.lineplot(x=paragraphs, y=negative_scores, label="Negative", color="r")
    sns.lineplot(x=paragraphs, y=overall_score, label="Overall", color="orange")

    plt.title("Emotion Scores by Paragraph")
    plt.xlabel("Paragraphs")
    plt.ylabel("Emotion Score")
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_ngram_graph(ngram_freq, output_path):
    """Plot n-gram frequencies."""
    ngram, counts = zip(*ngram_freq.most_common(10))
    ngram_labels = [' '.join(gram) for gram in ngram]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=ngram_labels, y=counts, palette="Blues_d")

    plt.title("Top 10 Most Frequent N-grams")
    plt.xlabel("N-grams")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def ngram_analysis(text, n=2):
    """Generate n-grams and return their frequencies."""
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]

    n_grams = ngrams(filtered_words, n)
    return Counter(n_grams)

def preprocess_text(text):
    """Preprocess text: tokenize, remove stopwords, stem, and correct spelling."""
    text = text.lower()
    words = word_tokenize(text)

    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
    filtered_words = [word for word in filtered_words if word not in string.punctuation]

    # Spelling correction
    corrected_text = TextBlob(" ".join(filtered_words)).correct()
    corrected_words = word_tokenize(str(corrected_text))

    # Stemming
    ps = PorterStemmer()
    stemmed_words = [ps.stem(word) for word in corrected_words]

    # Named Entity Recognition (optional)
    doc = nlp(" ".join(stemmed_words))
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    return " ".join(stemmed_words), entities

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''.join(page.extract_text() for page in reader.pages)
    return text, reader.pages

def analyze_story(pdf_path):
    """Main function to analyze the story from the PDF, including text extraction, sentiment analysis, and Word2Vec visualization."""
    text, pages = extract_text_from_pdf(pdf_path)

    sentiment_data = []
    ngram_sentiment_data = []
    all_ngram_freq = Counter()
    ngram = 2

    for i, page in enumerate(pages):
        page_text = page.extract_text()
        preprocessed_text, entities = preprocess_text(page_text)

        # N-gram Analysis (Bigrams)
        ngram_freq = ngram_analysis(preprocessed_text, n=2)
        all_ngram_freq.update(ngram_freq)

        # Sentiment Analysis by Paragraph using SentimentIntensityAnalyzer
        paragraphs = page_text.split("\n\n")  # Split by paragraphs
        for paragraph in paragraphs:
            sentiment = sia.polarity_scores(paragraph)
            sentiment_data.append(sentiment)

            # Sentiment analysis using n-grams
            ngram_sentiment = analyze_ngram_sentiment(paragraph, n = ngram)
            ngram_sentiment_data.append(ngram_sentiment)


    # Plot sentiment graph
    plot_emotion_graph(sentiment_data, "/content/emotion_graph.png")

    plot_emotion_graph(ngram_sentiment_data, "/content/ngram_emotion_graph.png")

    # Plot n-gram graph
    plot_ngram_graph(all_ngram_freq, "/content/ngram_graph.png")


    # Print some results for the entire document
    print(f"Top 10 Most Common N-grams: {all_ngram_freq.most_common(10)}")


def analyze_ngram_sentiment(paragraph, n = 2):
    """Analyze sentiment using n-grams for each paragraph."""
    ngram_sentiment = {'pos': 0, 'neu': 0, 'neg': 0, 'compound': 0}

    # Tokenize and preprocess the paragraph
    words = word_tokenize(paragraph.lower())
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]

    # Generate n-grams (bigrams)
    n_grams = ngrams(filtered_words, n)

    # Analyze sentiment for each n-gram
    for ngram in n_grams:
        ngram_text = " ".join(ngram)
        sentiment = sia.polarity_scores(ngram_text)
        ngram_sentiment['pos'] += sentiment['pos']
        ngram_sentiment['neu'] += sentiment['neu']
        ngram_sentiment['neg'] += sentiment['neg']
        ngram_sentiment['compound'] += sentiment['compound']

    # Normalize sentiment scores by number of n-grams
    ngram_count = len(filtered_words) - 1  # Number of bigrams in the paragraph
    if ngram_count > 0:
        ngram_sentiment = {k: v / ngram_count for k, v in ngram_sentiment.items()}

    return ngram_sentiment

# Path to the PDF file in Google Colab (you need to upload the file manually)
pdf_path = "/content/02. Jack and the Beanstalk Author Joseph Jacobs.pdf"  # Make sure to upload the file to /content folder
analyze_story(pdf_path)
