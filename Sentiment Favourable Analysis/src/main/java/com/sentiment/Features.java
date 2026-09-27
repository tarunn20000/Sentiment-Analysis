package com.sentiment;

import java.io.Serializable;
import java.util.*;

public class Features implements Serializable {
    private static final long serialVersionUID = 1L;

    private final int maxFeatures;
    private List<String> vocabularyList;
    private Map<String, Integer> vocabularyMap;
    private Map<String, Double> idfMap;
    private int numDocs;

    public Features(int maxFeatures) {
        this.maxFeatures = maxFeatures;
        this.vocabularyList = new ArrayList<>();
        this.vocabularyMap = new HashMap<>();
        this.idfMap = new HashMap<>();
        this.numDocs = 0;
    }

    public void fit(List<String> texts) {
        numDocs = texts.size();
        Map<String, Integer> docFrequency = new HashMap<>();
        Map<String, Integer> termOverallCount = new HashMap<>();

        // 1. Calculate document frequencies and overall counts
        for (String text : texts) {
            Set<String> uniqueTerms = new HashSet<>(Arrays.asList(text.split("\\s+")));
            for (String term : uniqueTerms) {
                if (term.length() > 1) {
                    docFrequency.put(term, docFrequency.getOrDefault(term, 0) + 1);
                }
            }
            for (String term : text.split("\\s+")) {
                if (term.length() > 1) {
                    termOverallCount.put(term, termOverallCount.getOrDefault(term, 0) + 1);
                }
            }
        }

        // 2. Select top features by frequency
        List<Map.Entry<String, Integer>> entries = new ArrayList<>(termOverallCount.entrySet());
        entries.sort((a, b) -> b.getValue().compareTo(a.getValue()));

        vocabularyList.clear();
        vocabularyMap.clear();
        int count = 0;
        for (Map.Entry<String, Integer> entry : entries) {
            if (count >= maxFeatures) break;
            String term = entry.getKey();
            vocabularyList.add(term);
            vocabularyMap.put(term, count);
            count++;
        }

        // 3. Compute IDF for all vocabulary items
        idfMap.clear();
        for (String term : vocabularyList) {
            double df = docFrequency.getOrDefault(term, 0);
            // IDF formula: ln(1 + N/DF) + 1
            double idf = Math.log(1.0 + ((double) numDocs / (df + 1.0))) + 1.0;
            idfMap.put(term, idf);
        }
    }

    public double[] transform(String text) {
        double[] vector = new double[vocabularyList.size()];
        if (text == null || text.trim().isEmpty()) {
            return vector;
        }

        String[] terms = text.split("\\s+");
        Map<String, Integer> termCounts = new HashMap<>();
        for (String term : terms) {
            if (vocabularyMap.containsKey(term)) {
                termCounts.put(term, termCounts.getOrDefault(term, 0) + 1);
            }
        }

        // Term Frequency (TF) * Inverse Document Frequency (IDF)
        double squareSum = 0.0;
        for (Map.Entry<String, Integer> entry : termCounts.entrySet()) {
            String term = entry.getKey();
            int idx = vocabularyMap.get(term);
            double tf = (double) entry.getValue(); // raw term frequency
            double idf = idfMap.get(term);
            double tfidf = tf * idf;
            vector[idx] = tfidf;
            squareSum += tfidf * tfidf;
        }

        // L2 normalization
        if (squareSum > 0.0) {
            double norm = Math.sqrt(squareSum);
            for (int i = 0; i < vector.length; i++) {
                vector[i] /= norm;
            }
        }

        return vector;
    }

    public double[][] fitTransform(List<String> texts) {
        fit(texts);
        double[][] matrix = new double[texts.size()][];
        for (int i = 0; i < texts.size(); i++) {
            matrix[i] = transform(texts.get(i));
        }
        return matrix;
    }

    public int getFeatureIndex(String term) {
        return vocabularyMap.getOrDefault(term, -1);
    }

    public String getFeatureName(int index) {
        if (index >= 0 && index < vocabularyList.size()) {
            return vocabularyList.get(index);
        }
        return null;
    }

    public int getVocabSize() {
        return vocabularyList.size();
    }

    public List<String> getVocabularyList() {
        return vocabularyList;
    }
}
