import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from collections import Counter
from textblob import TextBlob
import PyPDF2
import matplotlib.pyplot as plt
import seaborn as sns
import string
import spacy
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Load spaCy model for Named Entity Recognition
nlp = spacy.load("en_core_web_sm")

# Ensure required NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

def visualize_word2vec(model):
    """Visualize Word2Vec model using t-SNE for dimensionality reduction (3D)."""
    words = list(model.wv.index_to_key)
    word_vectors = [model.wv[word] for word in words]

    # Convert word_vectors to a NumPy array
    word_vectors = np.array(word_vectors)

    # Determine a suitable perplexity value
    perplexity = min(30, len(word_vectors) - 1)  # Default perplexity is 30, but it must be less than the number of words

    # Reduce dimensions using t-SNE
    tsne = TSNE(n_components=3, perplexity=perplexity, random_state=42)
    reduced_vectors = tsne.fit_transform(word_vectors)

    # Plotting the words in 3D space
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], reduced_vectors[:, 2], c='b', marker='o', alpha=0.6)

    for i, word in enumerate(words):
        ax.text(reduced_vectors[i, 0], reduced_vectors[i, 1], reduced_vectors[i, 2], word,
                color='orange', fontsize=10, alpha=0.7)

    ax.set_title('Word2Vec Word Embeddings Visualization (3D)')
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_zlabel('t-SNE Dimension 3')

    plt.tight_layout()
    plt.savefig('./Graphs/word2vec_3d_visualization.png')
    plt.show()


def train_word2vec_model(text):
    """Train a Word2Vec model using the input text."""
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
    
    # Train Word2Vec model
    model = Word2Vec([filtered_words], min_count=1, vector_size=100, window=5, sg=0)
    return model

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''.join(page.extract_text() for page in reader.pages)
    return text, reader.pages

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

def ngram_analysis(text, n=2):
    """Generate n-grams and return their frequencies."""
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
    
    n_grams = ngrams(filtered_words, n)
    return Counter(n_grams)

def sentiment_analysis(text):
    """Perform sentiment analysis on different sections of the text (beginning, middle, end)."""
    total_length = len(text)
    sections = {
        "Beginning": text[:total_length // 3],
        "Middle": text[total_length // 3: 2 * total_length // 3],
        "End": text[2 * total_length // 3:]
    }

    sentiments = {section: TextBlob(content).sentiment.polarity for section, content in sections.items()}
    return sentiments

def plot_combined_sentiment_and_ngram(sentiment_data, ngram_freq, output_path):
    """Plot sentiment analysis scores and annotate with most frequent n-grams."""
    sections = [section for page in sentiment_data for section in page.keys()]
    sentiment_scores = [score for page in sentiment_data for score in page.values()]
    
    ngram, counts = zip(*ngram_freq.most_common(10))
    ngram_labels = [' '.join(gram) for gram in ngram]
    
    plt.figure(figsize=(10, 6))

    # Line graph for sentiment scores
    sns.lineplot(x=sections, y=sentiment_scores, marker="o", label="Sentiment Scores", color="b")

    for i, ngram_label in enumerate(ngram_labels):
        plt.text(i, sentiment_scores[i] + 0.05, ngram_label, fontsize=10, ha='center', va='bottom', color="orange")
    
    plt.title("Combined Sentiment Scores and N-gram Annotations")
    plt.xlabel("Story Sections")
    plt.ylabel("Sentiment Score")
    plt.ylim(-1, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def analyze_story(pdf_path):
    """Main function to analyze the story from the PDF, including text extraction, sentiment analysis, and Word2Vec visualization."""
    text, pages = extract_text_from_pdf(pdf_path)
    
    sentiment_data = []
    all_ngram_freq = Counter()

    for i, page in enumerate(pages):
        page_text = page.extract_text()
        preprocessed_text, entities = preprocess_text(page_text)
        
        # N-gram Analysis (Bigrams)
        ngram_freq = ngram_analysis(preprocessed_text, n=2)
        all_ngram_freq.update(ngram_freq)
        
        # Sentiment Analysis
        sentiments = sentiment_analysis(page_text)
        sentiment_data.append(sentiments)

        # Train Word2Vec model on the page's text
        model = train_word2vec_model(page_text)
        print(f"Word2Vec Vocab (Page {i + 1}): {list(model.wv.index_to_key)}")
        visualize_word2vec(model)
    
    # Plot sentiment and n-grams
    plot_combined_sentiment_and_ngram(sentiment_data, all_ngram_freq, "./Graphs/combined_sentiment_ngram.png")
    
    # Print some results for the entire document
    print(f"All Sentiment Scores (Beginning, Middle, End): {sentiment_data}")
    print(f"Top 10 Most Common N-grams: {all_ngram_freq.most_common(10)}")

if __name__ == "__main__":
    pdf_path = "text/The Masque of the Red Death author Edgar Allan Poe.pdf"
    analyze_story(pdf_path)
