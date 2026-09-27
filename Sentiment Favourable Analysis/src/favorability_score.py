import numpy as np

class FavorabilityScorer:
    def __init__(self, temperature: float = 1.0):
        """
        Initializes the favorability scorer.
        
        Args:
            temperature (float): Controls calibration. 
                                 T > 1.0 smooths probabilities towards 0.5 (neutral/0 favorability).
                                 T < 1.0 pushes probabilities towards extremes (unfavorable/favorable).
        """
        self.temperature = temperature

    def calibrate_probability(self, prob: np.ndarray) -> np.ndarray:
        """
        Calibrates the probability using temperature scaling on the implied logits.
        """
        # Clamp to avoid divide-by-zero or log-of-zero errors
        prob = np.clip(prob, 1e-7, 1 - 1e-7)
        
        if self.temperature == 1.0:
            return prob
            
        # Back-calculate implied logits: z = log(p / (1-p))
        logits = np.log(prob / (1.0 - prob))
        
        # Scale logits and apply sigmoid
        calibrated_prob = 1.0 / (1.0 + np.exp(-logits / self.temperature))
        return calibrated_prob

    def calculate_score(self, prob: float, temperature: float = None) -> float:
        """
        Converts a single probability to a continuous favorability score in [-1.0, 1.0].
        """
        prob_arr = np.array([prob], dtype=float)
        return float(self.calculate_scores(prob_arr, temperature)[0])

    def calculate_scores(self, probs: np.ndarray, temperature: float = None) -> np.ndarray:
        """
        Converts an array of probabilities into continuous favorability scores in [-1.0, 1.0].
        """
        if not isinstance(probs, np.ndarray):
            probs = np.array(probs, dtype=float)
            
        temp = temperature if temperature is not None else self.temperature
        
        # 1. Calibrate probabilities
        if temp != 1.0:
            probs = self.calibrate_probability(probs)
            
        # 2. Map from [0, 1] to [-1, 1]
        scores = 2.0 * probs - 1.0
        return scores

    @staticmethod
    def get_label(score: float) -> str:
        """
        Returns a human-readable label for a given favorability score in [-1.0, 1.0].
        """
        if score < -0.6:
            return "Highly Unfavorable"
        elif score < -0.15:
            return "Unfavorable"
        elif score <= 0.15:
            return "Neutral"
        elif score < 0.6:
            return "Favorable"
        else:
            return "Highly Favorable"

    @staticmethod
    def get_color(score: float) -> str:
        """
        Returns a hex color string representing the favorability level (for frontend usage).
        Red -> Gray -> Green.
        """
        if score < -0.15:
            # Shift from Red to lighter Red/Orange
            return "#ef4444"
        elif score <= 0.15:
            # Gray/Neutral
            return "#6b7280"
        else:
            # Green
            return "#22c55e"
