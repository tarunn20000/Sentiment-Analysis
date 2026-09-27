import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score

class Evaluator:
    @staticmethod
    def calculate_metrics(y_true, y_pred, y_prob=None):
        """
        Calculates standard classification metrics.
        """
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        
        return metrics

    @staticmethod
    def calculate_continuous_metrics(y_true_scaled, favorability_scores):
        """
        Calculates regression metrics comparing continuous favorability scores (-1 to 1) 
        against ground truth targets mapped to -1 and 1.
        """
        mae = mean_absolute_error(y_true_scaled, favorability_scores)
        mse = mean_squared_error(y_true_scaled, favorability_scores)
        r2 = r2_score(y_true_scaled, favorability_scores)
        
        return {
            "mae": mae,
            "mse": mse,
            "r2_score": r2
        }

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, save_path=None):
        """
        Generates and displays (or saves) a confusion matrix heatmap.
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=["Negative", "Positive"], 
                    yticklabels=["Negative", "Positive"])
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_score_distribution(favorability_scores, save_path=None):
        """
        Plots a histogram/KDE of favorability scores.
        """
        plt.figure(figsize=(8, 5))
        sns.histplot(favorability_scores, kde=True, bins=30, color="purple")
        plt.title("Distribution of Favorability Scores")
        plt.xlabel("Favorability Score (-1 to +1)")
        plt.ylabel("Frequency")
        plt.axvline(x=0, color="red", linestyle="--", alpha=0.7, label="Neutral Line")
        plt.xlim(-1.1, 1.1)
        plt.legend()
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()
