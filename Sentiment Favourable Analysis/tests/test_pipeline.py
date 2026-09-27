import unittest
import numpy as np
from src.preprocessing import TextPreprocessor
from src.features import TFIDFFeatureExtractor
from src.favorability_score import FavorabilityScorer
from src.model import LogisticRegressionBaseline

class TestSentimentPipeline(unittest.TestCase):
    def setUp(self):
        self.preprocessor = TextPreprocessor()
        self.scorer = FavorabilityScorer()

    def test_text_cleaning(self):
        # HTML tags removal
        raw_html = "This movie was <br />great! <p>Loved it.</p>"
        cleaned = self.preprocessor.clean_text(raw_html)
        self.assertNotIn("br", cleaned)
        self.assertNotIn("p", cleaned)
        self.assertIn("great loved it", cleaned)

        # URL removal
        raw_url = "Check out this link http://example.com movie reviews"
        cleaned_url = self.preprocessor.clean_text(raw_url)
        self.assertNotIn("http", cleaned_url)
        self.assertNotIn("example.com", cleaned_url)
        self.assertIn("check out this link movie reviews", cleaned_url)

    def test_tokenization_and_lemmatization(self):
        text = "the actors were running and acting very well"
        processed = self.preprocessor.preprocess_text(text)
        # Check that stopwords like "the", "were", "and", "very" are removed
        self.assertNotIn("the", processed.split())
        self.assertNotIn("were", processed.split())
        # Lemmatization check ("running" -> "run", "acting" -> "act" or similar)
        self.assertTrue("run" in processed or "running" not in processed)

    def test_tfidf_feature_extraction(self):
        texts = ["great movie", "terrible script", "excellent performance"]
        extractor = TFIDFFeatureExtractor(max_features=10)
        X = extractor.fit_transform(texts)
        self.assertEqual(X.shape[0], 3)
        self.assertTrue(X.shape[1] <= 10)

    def test_favorability_scoring(self):
        # Standard scaling (T=1.0)
        scorer_t1 = FavorabilityScorer(temperature=1.0)
        self.assertAlmostEqual(scorer_t1.calculate_score(0.5), 0.0, places=4)
        self.assertAlmostEqual(scorer_t1.calculate_score(1.0), 1.0, places=4)
        self.assertAlmostEqual(scorer_t1.calculate_score(0.0), -1.0, places=4)

        # Temperature calibration (T=2.0 - pulls probabilities to 0.5, i.e., scores to 0)
        scorer_t2 = FavorabilityScorer(temperature=2.0)
        prob = 0.95
        score_t1_val = scorer_t1.calculate_score(prob)
        score_t2_val = scorer_t2.calculate_score(prob)
        # Score under T=2.0 should be closer to 0 than under T=1.0
        self.assertTrue(abs(score_t2_val) < abs(score_t1_val))

        # Labels
        self.assertEqual(FavorabilityScorer.get_label(0.85), "Highly Favorable")
        self.assertEqual(FavorabilityScorer.get_label(-0.75), "Highly Unfavorable")
        self.assertEqual(FavorabilityScorer.get_label(0.0), "Neutral")

    def test_baseline_logistic_regression(self):
        # Toy dataset
        texts = [
            "excellent positive movie", 
            "amazing script loved it", 
            "horrible bad worst film", 
            "waste of time trash"
        ]
        labels = [1, 1, 0, 0]
        
        # Preprocess
        processed = self.preprocessor.preprocess_batch(texts)
        
        # Vectorize
        extractor = TFIDFFeatureExtractor()
        X = extractor.fit_transform(processed)
        
        # Train LR
        lr = LogisticRegressionBaseline()
        lr.train(X, labels)
        
        # Check predictions
        probs = lr.predict_proba(X)
        self.assertEqual(len(probs), 4)
        self.assertTrue(np.all(probs >= 0.0) and np.all(probs <= 1.0))
        
        # Check coefficients
        tops = lr.get_top_features(extractor, top_n=2)
        self.assertTrue("positive" in tops)
        self.assertTrue("negative" in tops)


if __name__ == "__main__":
    unittest.main()
