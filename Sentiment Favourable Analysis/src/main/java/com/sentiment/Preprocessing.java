package com.sentiment;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.regex.Pattern;

public class Preprocessing {

    private final Set<String> stopwords;
    private final PorterStemmer stemmer;

    public Preprocessing() {
        this.stemmer = new PorterStemmer();
        this.stopwords = new HashSet<>(Arrays.asList(
            "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", 
            "he", "him", "his", "she", "her", "it", "its", "they", "them", "their", "what", 
            "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", 
            "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", 
            "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", 
            "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", 
            "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", 
            "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", 
            "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", 
            "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", 
            "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
        ));
    }

    public String cleanText(String text) {
        if (text == null) return "";
        
        // Remove HTML tags
        text = text.replaceAll("<[^>]*>", " ");
        
        // Remove URLs
        text = text.replaceAll("https?://\\S+|www\\.\\S+", "");
        
        // Convert to lowercase
        text = text.toLowerCase();
        
        // Keep only alphabetic characters and spaces
        text = text.replaceAll("[^a-z\\s]", " ");
        
        // Remove extra spaces
        text = text.replaceAll("\\s+", " ").trim();
        
        return text;
    }

    public List<String> tokenize(String text) {
        List<String> tokens = new ArrayList<>();
        if (text == null || text.trim().isEmpty()) {
            return tokens;
        }
        
        String[] words = text.split("\\s+");
        for (String word : words) {
            if (word.length() > 1 && !stopwords.contains(word)) {
                tokens.add(word);
            }
        }
        return tokens;
    }

    public List<String> stemTokens(List<String> tokens) {
        List<String> stemmed = new ArrayList<>();
        for (String token : tokens) {
            stemmed.add(stemmer.stem(token));
        }
        return stemmed;
    }

    public String preprocessText(String text) {
        String cleaned = cleanText(text);
        List<String> tokens = tokenize(cleaned);
        List<String> stemmed = stemTokens(tokens);
        return String.join(" ", stemmed);
    }

    public List<String> preprocessBatch(List<String> texts) {
        List<String> processed = new ArrayList<>();
        for (String text : texts) {
            processed.add(preprocessText(text));
        }
        return processed;
    }

    /**
     * Complete Porter Stemmer Implementation in Java.
     * Derived from Martin Porter's original algorithm.
     */
    private static class PorterStemmer {
        private char[] b;
        private int i,     /* offset into b */
                    i_end, /* offset to end of stemmed word */
                    j, k;

        public PorterStemmer() {
            b = new char[50];
            i = 0;
            i_end = 0;
        }

        private void add(char ch) {
            if (i == b.length) {
                char[] new_b = new char[i + 50];
                System.arraycopy(b, 0, new_b, 0, i);
                b = new_b;
            }
            b[i++] = ch;
        }

        private boolean cons(int i) {
            switch (b[i]) {
                case 'a': case 'e': case 'i': case 'o': case 'u': return false;
                case 'y': return (i == 0) || !cons(i - 1);
                default: return true;
            }
        }

        private int m() {
            int n = 0;
            int idx = 0;
            while (true) {
                if (idx > j) return n;
                if (!cons(idx)) break;
                idx++;
            }
            idx++;
            while (true) {
                while (true) {
                    if (idx > j) return n;
                    if (cons(idx)) break;
                    idx++;
                }
                idx++;
                n++;
                while (true) {
                    if (idx > j) return n;
                    if (!cons(idx)) break;
                    idx++;
                }
                idx++;
            }
        }

        private boolean vowelinstem() {
            int idx;
            for (idx = 0; idx <= j; idx++) {
                if (!cons(idx)) return true;
            }
            return false;
        }

        private boolean doublec(int idx) {
            if (idx < 1) return false;
            if (b[idx] != b[idx - 1]) return false;
            return cons(idx);
        }

        private boolean cvc(int idx) {
            if (idx < 2 || !cons(idx) || cons(idx - 1) || !cons(idx - 2)) return false;
            int ch = b[idx];
            return ch != 'w' && ch != 'x' && ch != 'y';
        }

        private boolean ends(String s) {
            int l = s.length();
            int o = k - l + 1;
            if (o < 0) return false;
            for (int idx = 0; idx < l; idx++) {
                if (b[o + idx] != s.charAt(idx)) return false;
            }
            j = k - l;
            return true;
        }

        private void setto(String s) {
            int l = s.length();
            int o = j + 1;
            for (int idx = 0; idx < l; idx++) {
                b[o + idx] = s.charAt(idx);
            }
            k = j + l;
        }

        private void r(String s) {
            if (m() > 0) setto(s);
        }

        private void step1() {
            if (b[k] == 's') {
                if (ends("sses")) k -= 2;
                else if (ends("ies")) setto("i");
                else if (b[k - 1] != 's') k--;
            }
            if (ends("eed")) {
                if (m() > 0) k--;
            } else if ((ends("ed") || ends("ing")) && vowelinstem()) {
                k = j;
                if (ends("at")) setto("ate");
                else if (ends("bl")) setto("ble");
                else if (ends("iz")) setto("ize");
                else if (doublec(k)) {
                    k--;
                    int ch = b[k];
                    if (ch == 'l' || ch == 's' || ch == 'z') k++;
                } else if (m() == 1 && cvc(k)) setto("e");
            }
        }

        private void step2() {
            if (ends("y") && vowelinstem()) b[k] = 'i';
        }

        private void step3() {
            if (k == 0) return;
            switch (b[k - 1]) {
                case 'a':
                    if (ends("ational")) { r("ate"); break; }
                    if (ends("tional")) { r("tion"); break; }
                    break;
                case 'c':
                    if (ends("enci")) { r("ence"); break; }
                    if (ends("anci")) { r("ance"); break; }
                    break;
                case 'e':
                    if (ends("izer")) { r("ize"); break; }
                    break;
                case 'l':
                    if (ends("bli")) { r("ble"); break; }
                    if (ends("alli")) { r("al"); break; }
                    if (ends("entli")) { r("ent"); break; }
                    if (ends("eli")) { r("e"); break; }
                    if (ends("ousli")) { r("ous"); break; }
                    break;
                case 'o':
                    if (ends("ization")) { r("ize"); break; }
                    if (ends("ation")) { r("ate"); break; }
                    if (ends("ator")) { r("ate"); break; }
                    break;
                case 's':
                    if (ends("alism")) { r("al"); break; }
                    if (ends("iveness")) { r("ive"); break; }
                    if (ends("fulness")) { r("ful"); break; }
                    if (ends("ousness")) { r("ous"); break; }
                    break;
                case 't':
                    if (ends("aliti")) { r("al"); break; }
                    if (ends("iviti")) { r("ive"); break; }
                    if (ends("biliti")) { r("ble"); break; }
                    break;
                case 'g':
                    if (ends("logi")) { r("log"); break; }
                    break;
            }
        }

        private void step4() {
            switch (b[k]) {
                case 'e':
                    if (ends("icate")) { r("ic"); break; }
                    if (ends("ative")) { r(""); break; }
                    if (ends("alize")) { r("al"); break; }
                    break;
                case 'i':
                    if (ends("iciti")) { r("ic"); break; }
                    break;
                case 'l':
                    if (ends("ical")) { r("ic"); break; }
                    if (ends("ful")) { r(""); break; }
                    break;
                case 's':
                    if (ends("ness")) { r(""); break; }
                    break;
            }
        }

        private void step5() {
            if (k == 0) return;
            switch (b[k - 1]) {
                case 'a': if (ends("al")) break; return;
                case 'c': if (ends("ance") || ends("ence")) break; return;
                case 'e': if (ends("er")) break; return;
                case 'i': if (ends("ic")) break; return;
                case 'l': if (ends("able") || ends("ible")) break; return;
                case 'n': if (ends("ant") || ends("ement") || ends("ment") || ends("ent")) break; return;
                case 'o': if (ends("ion") && j >= 0 && (b[j] == 's' || b[j] == 't')) break;
                          if (ends("ou")) break; return;
                case 's': if (ends("ism")) break; return;
                case 't': if (ends("ate") || ends("iti")) break; return;
                case 'u': if (ends("ous")) break; return;
                case 'v': if (ends("ive")) break; return;
                case 'z': if (ends("ize")) break; return;
                default: return;
            }
            if (m() > 1) k = j;
        }

        private void step6() {
            j = k;
            if (b[k] == 'e') {
                int a = m();
                if (a > 1 || (a == 1 && !cvc(k - 1))) k--;
            }
            if (b[k] == 'l' && doublec(k) && m() > 1) k--;
        }

        public synchronized String stem(String word) {
            word = word.trim();
            if (word.length() < 3) return word;
            
            i = 0;
            char[] chars = word.toCharArray();
            for (char c : chars) {
                add(c);
            }
            
            k = i - 1;
            step1();
            step2();
            step3();
            step4();
            step5();
            step6();
            i_end = k + 1;
            i = 0;
            return new String(b, 0, i_end);
        }
    }
}
