package com.sentiment;

import java.io.Serializable;

public class FavorabilityScore implements Serializable {
    private static final long serialVersionUID = 1L;

    private double temperature;

    public FavorabilityScore() {
        this(1.0);
    }

    public FavorabilityScore(double temperature) {
        this.temperature = temperature;
    }

    public double getTemperature() {
        return temperature;
    }

    public void setTemperature(double temperature) {
        this.temperature = temperature;
    }

    /**
     * Calibrates the probability using logit temperature scaling.
     */
    public double calibrateProbability(double prob) {
        // Clamp probability to avoid division by zero or log of zero
        prob = Math.max(1e-7, Math.min(1.0 - 1e-7, prob));

        if (temperature == 1.0) {
            return prob;
        }

        // Back-calculate implied logit: z = log(p / (1-p))
        double logit = Math.log(prob / (1.0 - prob));

        // Scale logit and apply sigmoid
        return 1.0 / (1.0 + Math.exp(-logit / temperature));
    }

    /**
     * Calculates the continuous favorability score in [-1.0, 1.0] from raw probability.
     */
    public double calculateScore(double prob) {
        double calibratedProb = calibrateProbability(prob);
        return 2.0 * calibratedProb - 1.0;
    }

    /**
     * Categorizes a favorability score into a human-readable label.
     */
    public static String getLabel(double score) {
        if (score < -0.6) {
            return "Highly Unfavorable";
        } else if (score < -0.15) {
            return "Unfavorable";
        } else if (score <= 0.15) {
            return "Neutral";
        } else if (score < 0.6) {
            return "Favorable";
        } else {
            return "Highly Favorable";
        }
    }

    /**
     * Maps a favorability score to a CSS color representation.
     */
    public static String getColor(double score) {
        if (score < -0.15) {
            return "#ef4444"; // red
        } else if (score <= 0.15) {
            return "#6b7280"; // gray
        } else {
            return "#22c55e"; // green
        }
    }
}
