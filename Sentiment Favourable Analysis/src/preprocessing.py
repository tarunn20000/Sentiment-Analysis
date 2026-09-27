import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Setup NLTK resources automatically
def _setup_nltk():
    resources = {
        "punkt": "tokenizers/punkt",
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet",
        "omw-1.4": "corpora/omw-1.4"
    }
    for res_name, res_path in resources.items():
        try:
            nltk.data.find(res_path)
        except LookupError:
            nltk.download(res_name, quiet=True)

_setup_nltk()

class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        # Fallback if stopwords couldn't download
        try:
            self.stop_words = set(stopwords.words("english"))
        except Exception:
            self.stop_words = {"i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", 
                               "he", "him", "his", "she", "her", "it", "its", "they", "them", "their", "what", 
                               "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", 
                               "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", 
                               "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", 
                               "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", 
                               "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", 
                               "in", "out", "on", "off", "over", "under", "again", "further", "then", "once"}

    def clean_text(self, text: str) -> str:
        """
        Removes HTML tags, URLs, and non-alphabetic characters.
        Converts text to lowercase.
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Remove HTML tags (e.g. <br />, <p>)
        text = re.sub(r"<[^>]*>", " ", text)
        
        # Remove URLs
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Keep only alphabetic characters and spaces
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
        
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()
        
        return text

    def tokenize_and_lemmatize(self, text: str) -> list:
        """
        Tokenizes cleaned text, removes stopwords, and lemmatizes words.
        """
        # Tokenization fallback to split if NLTK tokenizer fails
        try:
            tokens = nltk.word_tokenize(text)
        except Exception:
            tokens = text.split()
            
        processed_tokens = []
        for token in tokens:
            if token not in self.stop_words and len(token) > 1:
                # Lemmatize as noun (default) and verb
                lemma = self.lemmatizer.lemmatize(token, pos="v")
                lemma = self.lemmatizer.lemmatize(lemma, pos="n")
                processed_tokens.append(lemma)
                
        return processed_tokens

    def preprocess_text(self, text: str) -> str:
        """
        Complete preprocessing pipeline: cleaning, tokenization, lemmatization, 
        and joining back into a single string.
        """
        cleaned = self.clean_text(text)
        tokens = self.tokenize_and_lemmatize(cleaned)
        return " ".join(tokens)

    def preprocess_batch(self, texts: list) -> list:
        """
        Preprocesses a list of text reviews.
        """
        return [self.preprocess_text(text) for text in texts]
