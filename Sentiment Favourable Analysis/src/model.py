import os
import pickle
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

class LogisticRegressionBaseline:
    def __init__(self, C=1.0, max_iter=1000):
        self.model = LogisticRegression(C=C, max_iter=max_iter, random_state=42)
        
    def train(self, X, y):
        self.model.fit(X, y)
        return self
        
    def predict(self, X):
        return self.model.predict(X)
        
    def predict_proba(self, X):
        """
        Returns the probability of the positive class (class 1).
        """
        # predict_proba returns [prob_neg, prob_pos]
        return self.model.predict_proba(X)[:, 1]
        
    def save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self.model, f)
            
    def load(self, filepath):
        with open(filepath, "rb") as f:
            self.model = pickle.load(f)
        return self

    def get_top_features(self, vectorizer, top_n=20):
        """
        Returns the top_n positive and negative feature coefficients.
        Useful for explainability and word clouds.
        """
        if hasattr(vectorizer, "get_feature_names_out"):
            feature_names = np.array(vectorizer.get_feature_names_out())
        else:
            feature_names = np.array(vectorizer.vectorizer.get_feature_names_out())
        coefficients = self.model.coef_[0]
        
        # Sort coefficients
        sorted_indices = np.argsort(coefficients)
        
        negative_features = [
            (feature_names[idx], coefficients[idx]) 
            for idx in sorted_indices[:top_n]
        ]
        positive_features = [
            (feature_names[idx], coefficients[idx]) 
            for idx in sorted_indices[-top_n:][::-1]
        ]
        
        return {
            "positive": positive_features,
            "negative": negative_features
        }


class DistilBertClassifier:
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english", device=None):
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        
        # Load sequence classification model
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=2
        ).to(self.device)
        
    def train(self, texts, labels, epochs=3, batch_size=16, lr=2e-5):
        """
        Fine-tune DistilBERT on local data.
        """
        self.model.train()
        
        # Tokenize inputs
        encoded_inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=128, 
            return_tensors="pt"
        )
        
        input_ids = encoded_inputs["input_ids"]
        attention_mask = encoded_inputs["attention_mask"]
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        dataset = TensorDataset(input_ids, attention_mask, labels_tensor)
        dataloader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=batch_size)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                b_input_ids, b_attn_mask, b_labels = [t.to(self.device) for t in batch]
                
                self.model.zero_grad()
                
                outputs = self.model(
                    input_ids=b_input_ids,
                    attention_mask=b_attn_mask,
                    labels=b_labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss / len(dataloader):.4f}")
            
        return self

    def predict_proba(self, texts, batch_size=32):
        """
        Returns the probability of the positive class (class 1).
        """
        self.model.eval()
        
        if isinstance(texts, str):
            texts = [texts]
            
        all_probs = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                # Apply softmax over class dimension
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                # Extract positive class probability
                all_probs.extend(probs[:, 1])
                
        return np.array(all_probs)

    def save(self, directory):
        """
        Saves the model state and tokenizer.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.model.save_pretrained(directory)
        self.tokenizer.save_pretrained(directory)

    def load(self, directory):
        """
        Loads the model state and tokenizer.
        """
        self.model = DistilBertForSequenceClassification.from_pretrained(directory).to(self.device)
        self.tokenizer = DistilBertTokenizer.from_pretrained(directory)
        return self
