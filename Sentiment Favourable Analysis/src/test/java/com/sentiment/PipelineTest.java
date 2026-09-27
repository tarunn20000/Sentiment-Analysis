package com.sentiment;

import java.util.*;

public class PipelineTest {

    private Preprocessing preprocessor;
    private FavorabilityScore scorer;

    public void setUp() {
        preprocessor = new Preprocessing();
        scorer = new FavorabilityScore();
    }

    public void testTextCleaning() {
        // HTML stripping
        String htmlText = "The movie was <br />great! <p>Loved it.</p>";
        String cleanedHtml = preprocessor.cleanText(htmlText);
        assertFalse(cleanedHtml.contains("br"), "Should not contain br tag");
        assertFalse(cleanedHtml.contains("p"), "Should not contain p tag");
        assertTrue(cleanedHtml.contains("great loved it"), "Should contain cleaned text");

        // URL stripping
        String urlText = "Check out my review http://example.com online";
        String cleanedUrl = preprocessor.cleanText(urlText);
        assertFalse(cleanedUrl.contains("http"), "Should remove http");
        assertFalse(cleanedUrl.contains("example.com"), "Should remove domain");
        assertTrue(cleanedUrl.contains("check out my review online"), "Should preserve words");
    }

    public void testTokenizationAndStemming() {
        String text = "the actors were running and acting very well";
        String cleaned = preprocessor.cleanText(text);
        List<String> tokens = preprocessor.tokenize(cleaned);
        
        assertFalse(tokens.contains("the"), "Stopword 'the' must be removed");
        assertFalse(tokens.contains("were"), "Stopword 'were' must be removed");

        List<String> stemmed = preprocessor.stemTokens(tokens);
        assertTrue(stemmed.contains("run"), "Stemming should map running -> run");
        assertTrue(stemmed.contains("act"), "Stemming should map acting -> act");
    }

    public void testTFIDFFeatureExtraction() {
        List<String> texts = Arrays.asList(
            "great movie love",
            "terrible script waste",
            "excellent performance great acting"
        );
        Features extractor = new Features(10);
        double[][] X = extractor.fitTransform(texts);

        assertEquals(3, X.length, "Should have 3 document vectors");
        assertTrue(X[0].length <= 10, "Feature dimension should be <= 10");

        for (double[] vec : X) {
            double sqSum = 0.0;
            for (double val : vec) {
                sqSum += val * val;
            }
            if (sqSum > 0.0) {
                assertEquals(1.0, sqSum, 1e-4, "L2 norm square sum should equal 1.0");
            }
        }
    }

    public void testFavorabilityScoring() {
        FavorabilityScore scorerT1 = new FavorabilityScore(1.0);
        assertEquals(0.0, scorerT1.calculateScore(0.5), 1e-4, "0.5 probability maps to 0.0 score");
        assertEquals(1.0, scorerT1.calculateScore(1.0), 1e-4, "1.0 probability maps to +1.0 score");
        assertEquals(-1.0, scorerT1.calculateScore(0.0), 1e-4, "0.0 probability maps to -1.0 score");

        FavorabilityScore scorerT2 = new FavorabilityScore(2.0);
        double prob = 0.95;
        double scoreT1 = scorerT1.calculateScore(prob);
        double scoreT2 = scorerT2.calculateScore(prob);
        assertTrue(Math.abs(scoreT2) < Math.abs(scoreT1), "T=2.0 temperature smooths score closer to 0");

        assertEquals("Highly Favorable", FavorabilityScore.getLabel(0.85), "Label for 0.85");
        assertEquals("Highly Unfavorable", FavorabilityScore.getLabel(-0.75), "Label for -0.75");
        assertEquals("Neutral", FavorabilityScore.getLabel(0.0), "Label for 0.0");
    }

    public void testSGDLogisticRegression() {
        List<String> texts = Arrays.asList(
            "excellent positive movie loved", 
            "amazing script perfect acting", 
            "horrible bad worst film waste", 
            "waste of time trash terrible"
        );
        int[] labels = {1, 1, 0, 0};

        List<String> processed = preprocessor.preprocessBatch(texts);
        Features extractor = new Features(100);
        double[][] X = extractor.fitTransform(processed);

        Model lr = new Model(extractor.getVocabSize(), 0.1, 0.0001, 30);
        lr.train(X, labels);

        for (double[] vec : X) {
            double prob = lr.predictProba(vec);
            assertTrue(prob >= 0.0 && prob <= 1.0, "Probability must be in [0, 1]");
        }

        Map<String, List<Map.Entry<String, Double>>> topFeatures = lr.getTopFeatures(extractor, 2);
        assertTrue(topFeatures.containsKey("positive"), "Should contain positive features");
        assertTrue(topFeatures.containsKey("negative"), "Should contain negative features");
        assertEquals(2, topFeatures.get("positive").size(), "Should return 2 positive features");
        assertEquals(2, topFeatures.get("negative").size(), "Should return 2 negative features");
    }

    public void testEvaluationMetrics() {
        int[] yTrue = {1, 1, 0, 0};
        int[] yPred = {1, 0, 0, 1};

        Map<String, Double> classMetrics = Evaluate.calculateClassificationMetrics(yTrue, yPred);
        assertEquals(0.5, classMetrics.get("accuracy"), 1e-4, "Accuracy should be 0.5");

        double[] yTrueScaled = {1.0, 1.0, -1.0, -1.0};
        double[] yPredScore = {0.8, -0.2, -0.9, 0.4};

        Map<String, Double> contMetrics = Evaluate.calculateContinuousMetrics(yTrueScaled, yPredScore);
        assertEquals(0.725, contMetrics.get("mae"), 1e-4, "MAE calculation check");
    }

    // Helper assertions
    private static void assertTrue(boolean condition, String msg) {
        if (!condition) throw new AssertionError("FAILED: " + msg);
    }

    private static void assertFalse(boolean condition, String msg) {
        if (condition) throw new AssertionError("FAILED: " + msg);
    }

    private static void assertEquals(double expected, double actual, double delta, String msg) {
        if (Math.abs(expected - actual) > delta) {
            throw new AssertionError(String.format("FAILED: %s (Expected %f, got %f)", msg, expected, actual));
        }
    }

    private static void assertEquals(Object expected, Object actual, String msg) {
        if (!Objects.equals(expected, actual)) {
            throw new AssertionError(String.format("FAILED: %s (Expected %s, got %s)", msg, expected, actual));
        }
    }

    public static void main(String[] args) {
        System.out.println("==================================================");
        System.out.println("Running Java Sentiment Pipeline Test Suite...");
        System.out.println("==================================================");

        PipelineTest tester = new PipelineTest();
        int passed = 0;

        try {
            tester.setUp(); tester.testTextCleaning(); System.out.println("✓ testTextCleaning passed"); passed++;
            tester.setUp(); tester.testTokenizationAndStemming(); System.out.println("✓ testTokenizationAndStemming passed"); passed++;
            tester.setUp(); tester.testTFIDFFeatureExtraction(); System.out.println("✓ testTFIDFFeatureExtraction passed"); passed++;
            tester.setUp(); tester.testFavorabilityScoring(); System.out.println("✓ testFavorabilityScoring passed"); passed++;
            tester.setUp(); tester.testSGDLogisticRegression(); System.out.println("✓ testSGDLogisticRegression passed"); passed++;
            tester.setUp(); tester.testEvaluationMetrics(); System.out.println("✓ testEvaluationMetrics passed"); passed++;

            System.out.println("==================================================");
            System.out.println("SUCCESS: All " + passed + " Java tests passed cleanly!");
            System.out.println("==================================================");
        } catch (AssertionError e) {
            System.err.println("❌ Test Failure: " + e.getMessage());
            System.exit(1);
        } catch (Exception e) {
            System.err.println("❌ Test Error: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }
}
