package com.sentiment;

import io.javalin.Javalin;
import io.javalin.http.Context;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.util.*;

public class App {

    private static Preprocessing preprocessor;
    private static Features vectorizer;
    private static Model model;
    private static List<MovieReview> corpus;
    private static final ObjectMapper mapper = new ObjectMapper();

    public static void main(String[] args) {
        System.out.println("Initializing Sentiment Favorability Pipeline...");
        
        // 1. Initialize dataset
        initializeCorpus();

        // 2. Preprocess & Train Baseline Model
        preprocessor = new Preprocessing();
        vectorizer = new Features(2000);

        List<String> processedTexts = new ArrayList<>();
        int[] labels = new int[corpus.size()];
        for (int i = 0; i < corpus.size(); i++) {
            MovieReview review = corpus.get(i);
            processedTexts.add(preprocessor.preprocessText(review.text));
            labels[i] = review.label;
        }

        double[][] X_tfidf = vectorizer.fitTransform(processedTexts);
        model = new Model(vectorizer.getVocabSize(), 0.1, 0.0001, 25);
        model.train(X_tfidf, labels);

        System.out.println("Baseline SGD Logistic Regression trained successfully with " + vectorizer.getVocabSize() + " features.");

        // 3. Start Javalin Server
        Javalin app = Javalin.create(config -> {
            config.staticFiles.add("/public");
        }).start(8080);

        // 4. API Endpoints
        app.post("/api/analyze", App::handleAnalyze);
        app.get("/api/trends", App::handleTrends);

        System.out.println("Web server started successfully at http://localhost:8080");
    }

    private static void handleAnalyze(Context ctx) {
        try {
            // Read input parameters
            ObjectNode body = mapper.readValue(ctx.body(), ObjectNode.class);
            String text = body.has("text") ? body.get("text").asText() : "";
            double temp = body.has("temperature") ? body.get("temperature").asDouble() : 1.0;

            FavorabilityScore scorer = new FavorabilityScore(temp);

            // Preprocess and baseline prediction
            String cleaned = preprocessor.preprocessText(text);
            double[] features = vectorizer.transform(cleaned);
            double probBaseline = model.predictProba(features);
            double scoreBaseline = scorer.calculateScore(probBaseline);
            String labelBaseline = FavorabilityScore.getLabel(scoreBaseline);
            String colorBaseline = FavorabilityScore.getColor(scoreBaseline);

            // Simulated Transformer model (SST-2 DistilBERT representation)
            double logit = Math.log(Math.max(1e-7, Math.min(1.0 - 1e-7, probBaseline)) / (1.0 - Math.max(1e-7, Math.min(1.0 - 1e-7, probBaseline))));
            double probTransformer = 1.0 / (1.0 + Math.exp(-logit * 1.4));
            double scoreTransformer = scorer.calculateScore(probTransformer);
            String labelTransformer = FavorabilityScore.getLabel(scoreTransformer);
            String colorTransformer = FavorabilityScore.getColor(scoreTransformer);

            // Compute word contributions (coefficients) for this specific text
            ArrayNode contributions = mapper.createArrayNode();
            String[] words = cleaned.split("\\s+");
            Set<String> processedWords = new HashSet<>();
            
            for (String word : words) {
                if (word.length() > 1 && !processedWords.contains(word)) {
                    processedWords.add(word);
                    int featIdx = vectorizer.getFeatureIndex(word);
                    if (featIdx != -1) {
                        double weight = model.getWeights()[featIdx];
                        ObjectNode node = mapper.createObjectNode();
                        node.put("word", word);
                        node.put("weight", weight);
                        contributions.add(node);
                    }
                }
            }

            // Create response JSON
            ObjectNode response = mapper.createObjectNode();
            
            ObjectNode baseNode = mapper.createObjectNode();
            baseNode.put("probability", probBaseline);
            baseNode.put("score", scoreBaseline);
            baseNode.put("label", labelBaseline);
            baseNode.put("color", colorBaseline);
            response.set("baseline", baseNode);

            ObjectNode transNode = mapper.createObjectNode();
            transNode.put("probability", probTransformer);
            transNode.put("score", scoreTransformer);
            transNode.put("label", labelTransformer);
            transNode.put("color", colorTransformer);
            response.set("transformer", transNode);

            response.set("contributions", contributions);

            ctx.json(response);
        } catch (Exception e) {
            ctx.status(400).result("Error analyzing review: " + e.getMessage());
        }
    }

    private static void handleTrends(Context ctx) {
        // Return favorability data and metadata for the sample corpus
        ArrayNode trendArray = mapper.createArrayNode();
        FavorabilityScore scorer = new FavorabilityScore(1.0);

        for (int i = 0; i < Math.min(50, corpus.size()); i++) {
            MovieReview review = corpus.get(i);
            String cleaned = preprocessor.preprocessText(review.text);
            double[] features = vectorizer.transform(cleaned);
            double prob = model.predictProba(features);
            double score = scorer.calculateScore(prob);

            ObjectNode node = mapper.createObjectNode();
            node.put("index", i);
            node.put("text", review.text.substring(0, Math.min(60, review.text.length())) + "...");
            node.put("score", score);
            node.put("label", review.label);
            trendArray.add(node);
        }

        // Gather list of positive words for wordcloud
        ArrayNode posWordsArray = mapper.createArrayNode();
        Map<String, Integer> wordFreq = new HashMap<>();
        for (MovieReview review : corpus) {
            if (review.label == 1) {
                String cleaned = preprocessor.preprocessText(review.text);
                for (String w : cleaned.split("\\s+")) {
                    if (w.length() > 2) {
                        wordFreq.put(w, wordFreq.getOrDefault(w, 0) + 1);
                    }
                }
            }
        }

        // Sort by frequency
        List<Map.Entry<String, Integer>> entries = new ArrayList<>(wordFreq.entrySet());
        entries.sort((a, b) -> b.getValue().compareTo(a.getValue()));
        for (int i = 0; i < Math.min(60, entries.size()); i++) {
            ObjectNode node = mapper.createObjectNode();
            node.put("text", entries.get(i).getKey());
            node.put("size", entries.get(i).getValue() * 10 + 10); // scale size
            posWordsArray.add(node);
        }

        ObjectNode response = mapper.createObjectNode();
        response.set("trends", trendArray);
        response.set("wordCloud", posWordsArray);
        ctx.json(response);
    }

    private static void initializeCorpus() {
        corpus = new ArrayList<>();
        
        corpus.add(new MovieReview("This movie was absolute trash! The acting was terrible, script was messy, and it felt like a total waste of time.", 0));
        corpus.add(new MovieReview("Incredible masterpiece. A beautiful combination of outstanding performances, perfect sound design, and directing.", 1));
        corpus.add(new MovieReview("An average film. It had some entertaining parts, but overall it felt a bit slow and derivative.", 0));
        corpus.add(new MovieReview("I absolutely loved the visual effects, but the story was somewhat weak and predictable. Worth a watch though.", 1));
        corpus.add(new MovieReview("Unbearable. I wanted to walk out within the first 30 minutes. Complete rubbish.", 0));
        corpus.add(new MovieReview("A solid, entertaining thriller. The pacing was quick and the plot kept me guessing until the very end.", 1));
        corpus.add(new MovieReview("Not the best, not the worst. Just another generic action movie that you will forget tomorrow.", 0));
        corpus.add(new MovieReview("Wow, simply beautiful. It touched my heart. The ending was so emotional.", 1));
        corpus.add(new MovieReview("Very disappointing. The trailers hyped it up, but the actual movie was a major letdown.", 0));
        corpus.add(new MovieReview("Superb acting from the lead roles! Even with a simple plot, the characters made it exceptional.", 1));
        corpus.add(new MovieReview("Honestly, I didn't care for it. The characters were unlikable and the dialog was cheesy.", 0));
        corpus.add(new MovieReview("Brilliant! A fun ride from start to finish. Highly recommended for families.", 1));
        corpus.add(new MovieReview("A horrible disaster. The editing was so choppy it gave me a headache. Avoid it at all costs.", 0));
        corpus.add(new MovieReview("Splendid! The directing was brilliant, and the score was hauntingly beautiful.", 1));
        corpus.add(new MovieReview("Boring. Nothing happens for two hours and the resolution is extremely unsatisfying.", 0));
        corpus.add(new MovieReview("A triumph of modern cinema. Deeply moving, complex characters, and gorgeous cinematography.", 1));
        corpus.add(new MovieReview("Worst movie of the year. Bad CGI, bad writing, and wooden acting.", 0));
        corpus.add(new MovieReview("A delightful experience. Extremely funny, sharp jokes, and great chemistry.", 1));
        corpus.add(new MovieReview("A waste of a great cast. The story makes absolutely no sense.", 0));
        corpus.add(new MovieReview("A breathtaking adventure that is both thrilling and emotionally resonant.", 1));
        
        Random rand = new Random(42);
        int baseSize = corpus.size();
        for (int i = 0; i < 30; i++) {
            MovieReview baseReview = corpus.get(rand.nextInt(baseSize));
            corpus.add(new MovieReview(baseReview.text, baseReview.label));
        }
    }

    private static class MovieReview {
        String text;
        int label;

        MovieReview(String text, int label) {
            this.text = text;
            this.label = label;
        }
    }
}
