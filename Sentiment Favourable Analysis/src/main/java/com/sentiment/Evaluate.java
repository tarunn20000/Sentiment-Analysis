package com.sentiment;

import java.util.HashMap;
import java.util.Map;

public class Evaluate {

    /**
     * Calculates classification metrics: Accuracy and F1 Score.
     */
    public static Map<String, Double> calculateClassificationMetrics(int[] yTrue, int[] yPred) {
        if (yTrue.length == 0 || yTrue.length != yPred.length) {
            throw new IllegalArgumentException("Inputs must have matching non-zero lengths");
        }

        int count = yTrue.length;
        int correct = 0;
        int tp = 0, fp = 0, fn = 0, tn = 0;

        for (int i = 0; i < count; i++) {
            int gold = yTrue[i];
            int pred = yPred[i];

            if (gold == pred) {
                correct++;
            }

            if (gold == 1 && pred == 1) tp++;
            else if (gold == 0 && pred == 1) fp++;
            else if (gold == 1 && pred == 0) fn++;
            else if (gold == 0 && pred == 0) tn++;
        }

        double accuracy = (double) correct / count;
        double precision = (tp + fp > 0) ? (double) tp / (tp + fp) : 0.0;
        double recall = (tp + fn > 0) ? (double) tp / (tp + fn) : 0.0;
        double f1 = (precision + recall > 0) ? 2.0 * (precision * recall) / (precision + recall) : 0.0;

        Map<String, Double> metrics = new HashMap<>();
        metrics.put("accuracy", accuracy);
        metrics.put("precision", precision);
        metrics.put("recall", recall);
        metrics.put("f1", f1);
        return metrics;
    }

    /**
     * Calculates regression error metrics comparing continuous scores in [-1.0, 1.0]
     * against ground truth labels scaled to -1.0 (negative) and 1.0 (positive).
     */
    public static Map<String, Double> calculateContinuousMetrics(double[] yTrueScaled, double[] favorabilityScores) {
        if (yTrueScaled.length == 0 || yTrueScaled.length != favorabilityScores.length) {
            throw new IllegalArgumentException("Inputs must have matching non-zero lengths");
        }

        int count = yTrueScaled.length;
        double absoluteErrorSum = 0.0;
        double squaredErrorSum = 0.0;

        for (int i = 0; i < count; i++) {
            double diff = favorabilityScores[i] - yTrueScaled[i];
            absoluteErrorSum += Math.abs(diff);
            squaredErrorSum += diff * diff;
        }

        double mae = absoluteErrorSum / count;
        double mse = squaredErrorSum / count;

        Map<String, Double> metrics = new HashMap<>();
        metrics.put("mae", mae);
        metrics.put("mse", mse);
        metrics.put("rmse", Math.sqrt(mse));
        return metrics;
    }
}
