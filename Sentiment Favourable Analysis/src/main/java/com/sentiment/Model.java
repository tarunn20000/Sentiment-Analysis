package com.sentiment;

import java.io.Serializable;
import java.util.*;

public class Model implements Serializable {
    private static final long serialVersionUID = 1L;

    private double[] weights;
    private double bias;
    private final double learningRate;
    private final double l2Regularization;
    private final int epochs;

    public Model(int numFeatures) {
        this(numFeatures, 0.05, 0.0001, 15);
    }

    public Model(int numFeatures, double learningRate, double l2Regularization, int epochs) {
        this.weights = new double[numFeatures];
        this.bias = 0.0;
        this.learningRate = learningRate;
        this.l2Regularization = l2Regularization;
        this.epochs = epochs;
    }

    private double sigmoid(double z) {
        return 1.0 / (1.0 + Math.exp(-Math.max(-20.0, Math.min(20.0, z))));
    }

    public double predictProba(double[] x) {
        double dotProduct = 0.0;
        for (int i = 0; i < weights.length; i++) {
            dotProduct += weights[i] * x[i];
        }
        return sigmoid(dotProduct + bias);
    }

    public int predict(double[] x) {
        return predictProba(x) >= 0.5 ? 1 : 0;
    }

    public void train(double[][] X, int[] y) {
        int numSamples = X.length;
        if (numSamples == 0) return;
        int numFeatures = X[0].length;
        
        // Reset or resize weights if dimensions don't match
        if (weights.length != numFeatures) {
            weights = new double[numFeatures];
            bias = 0.0;
        }

        Random rand = new Random(42);

        for (int epoch = 0; epoch < epochs; epoch++) {
            // Shuffle indices for Stochastic Gradient Descent
            List<Integer> indices = new ArrayList<>();
            for (int i = 0; i < numSamples; i++) {
                indices.add(i);
            }
            Collections.shuffle(indices, rand);

            for (int idx : indices) {
                double[] x = X[idx];
                int target = y[idx];

                // Prediction
                double prob = predictProba(x);
                double error = prob - target;

                // Update weights and bias
                for (int j = 0; j < numFeatures; j++) {
                    double gradient = error * x[j] + l2Regularization * weights[j];
                    weights[j] -= learningRate * gradient;
                }
                bias -= learningRate * error;
            }
        }
    }

    /**
     * Extracts the top positive and negative features based on model weights.
     */
    public Map<String, List<Map.Entry<String, Double>>> getTopFeatures(Features extractor, int topN) {
        List<Map.Entry<String, Double>> termWeights = new ArrayList<>();
        
        for (int i = 0; i < weights.length; i++) {
            String term = extractor.getFeatureName(i);
            if (term != null) {
                termWeights.add(new AbstractMap.SimpleEntry<>(term, weights[i]));
            }
        }

        // Sort by weight value ascending
        termWeights.sort(Map.Entry.comparingByValue());

        List<Map.Entry<String, Double>> negative = new ArrayList<>();
        List<Map.Entry<String, Double>> positive = new ArrayList<>();

        // Add most negative (first elements in sorted list)
        for (int i = 0; i < Math.min(topN, termWeights.size()); i++) {
            negative.add(termWeights.get(i));
        }

        // Add most positive (last elements in sorted list, in descending order)
        for (int i = termWeights.size() - 1; i >= Math.max(0, termWeights.size() - topN); i--) {
            positive.add(termWeights.get(i));
        }

        Map<String, List<Map.Entry<String, Double>>> result = new HashMap<>();
        result.put("negative", negative);
        result.put("positive", positive);
        return result;
    }

    public double[] getWeights() {
        return weights;
    }

    public double getBias() {
        return bias;
    }
}
