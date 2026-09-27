import pickle
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel

class TFIDFFeatureExtractor:
    def __init__(self, max_features=5000, ngram_range=(1, 2)):
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        
    def fit(self, texts):
        self.vectorizer.fit(texts)
        return self
        
    def transform(self, texts):
        return self.vectorizer.transform(texts)
        
    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(texts)
        
    def get_feature_names_out(self):
        return self.vectorizer.get_feature_names_out()
        
    def save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self.vectorizer, f)
            
    def load(self, filepath):
        with open(filepath, "rb") as f:
            self.vectorizer = pickle.load(f)
        return self


class DistilBertFeatureExtractor:
    def __init__(self, model_name="distilbert-base-uncased", max_length=128, device=None):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None  # Lazy load the transformer model for embeddings to save memory
        
    @property
    def model(self):
        if self._model is None:
            self._model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self._model.eval()
        return self._model

    def tokenize_texts(self, texts, return_tensors="pt"):
        """
        Tokenizes text inputs for model ingestion.
        Returns a dictionary containing input_ids and attention_mask.
        """
        if isinstance(texts, str):
            texts = [texts]
            
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors=return_tensors
        )
        return encoded

    def extract_embeddings(self, texts, batch_size=32):
        """
        Extracts sentence-level embeddings from DistilBERT using mean pooling.
        """
        all_embeddings = []
        
        # Process in batches to avoid GPU/CPU memory overflow
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            encoded = self.tokenize_texts(batch_texts, return_tensors="pt")
            
            # Move tensors to device
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Perform mean pooling
                token_embeddings = outputs.last_hidden_state # Shape: [batch_size, seq_len, hidden_dim]
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                mean_pooled = sum_embeddings / sum_mask
                
                all_embeddings.append(mean_pooled.cpu().numpy())
                
        return np.vstack(all_embeddings)
