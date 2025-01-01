use std::collections::HashMap;

type TermFrequency = HashMap<String, usize>;
type InverseDocumentFrequency = HashMap<String, f64>;

pub struct Bm25<F>
where
    F: Fn(&str) -> Vec<String>,
{
    corpus_size: usize,
    raw_corpus: Vec<String>,
    corpus: Vec<Vec<String>>,

    corpus_frequencies: Vec<TermFrequency>,
    corpus_lengths: Vec<usize>,
    total_term_count: usize,
    avg_doc_len: f64,

    docs_per_term: TermFrequency,
    idf: InverseDocumentFrequency,

    tokenizer: F,
    k1: f64,
    b: f64,
    epsilon: f64,
}

#[derive(Debug)]
pub struct SearchResult {
    pub document_index: usize,
    pub score: f64,
}

impl<F> Bm25<F>
where
    F: Fn(&str) -> Vec<String>,
{
    pub fn new(raw_corpus: Vec<String>, tokenizer: F, k1: f64, b: f64, epsilon: f64) -> Self {
        let corpus_size = raw_corpus.len();
        let mut bm25 = Self {
            total_term_count: 0,
            avg_doc_len: 0.0,
            idf: InverseDocumentFrequency::new(),
            raw_corpus,
            tokenizer,
            corpus_size,
            corpus: Vec::with_capacity(corpus_size), // Pre-allocate capacity
            corpus_frequencies: Vec::with_capacity(corpus_size),
            corpus_lengths: Vec::with_capacity(corpus_size),
            docs_per_term: TermFrequency::new(),
            k1,
            b,
            epsilon,
        };

        bm25.prepare_corpus();
        bm25.prepare_frequencies();
        bm25.calculate_idf();
        bm25
    }

    fn prepare_corpus(&mut self) {
        self.corpus = self.raw_corpus
            .iter()
            .map(|doc| (self.tokenizer)(doc))
            .collect();
    }

    fn prepare_frequencies(&mut self) {
        for terms in &self.corpus {
            let length = terms.len();
            self.corpus_lengths.push(length);
            self.total_term_count += length;

            let mut freq = TermFrequency::with_capacity(terms.len());
            for term in terms {
                *freq.entry(term.clone()).or_insert(0) += 1;
                *self.docs_per_term.entry(term.clone()).or_insert(0) += 1;
            }
            self.corpus_frequencies.push(freq);
        }

        self.avg_doc_len = self.total_term_count as f64 / self.corpus_size as f64;
    }

    fn calculate_idf(&mut self) {
        let corpus_size_f64 = self.corpus_size as f64;
        let total_terms_f64 = self.total_term_count as f64;

        let mut total_idf = 0.0;
        let mut terms_with_negative_idf = Vec::new();

        for (term, &freq) in &self.docs_per_term {
            let freq_f64 = freq as f64;
            let term_idf = (corpus_size_f64 - freq_f64 + 0.5).ln() - (freq_f64 + 0.5).ln();

            self.idf.insert(term.clone(), term_idf);
            total_idf += term_idf;

            if term_idf < 0.0 {
                terms_with_negative_idf.push(term.clone());
            }
        }

        let eps = self.epsilon * (total_idf / total_terms_f64);
        for term in terms_with_negative_idf {
            self.idf.insert(term, eps);
        }
    }

    fn calculate_score(&self, query_term: &str, doc_lengths: &[f64]) -> Vec<f64> {
        let idf = self.idf.get(query_term).copied().unwrap_or(0.0);
        let mut scores = vec![0.0; self.corpus_size];

        for (doc_idx, doc_freq) in self.corpus_frequencies.iter().enumerate() {
            let freq = *doc_freq.get(query_term).unwrap_or(&0) as f64;
            let len_norm = 1.0 - self.b + self.b * doc_lengths[doc_idx] / self.avg_doc_len;
            scores[doc_idx] = idf * (freq * (self.k1 + 1.0) / (freq + self.k1 * len_norm));
        }

        scores
    }

    fn sum_scores(&self, query: &str) -> Vec<f64> {
        let q_terms = (self.tokenizer)(query);
        let doc_lengths: Vec<f64> = self.corpus_lengths.iter().map(|&x| x as f64).collect();

        let mut scores = vec![0.0; self.corpus_size];
        for term in q_terms {
            let term_scores = self.calculate_score(&term, &doc_lengths);
            for (idx, score) in term_scores.iter().enumerate() {
                scores[idx] += score;
            }
        }
        scores
    }

    pub fn top_n(&self, query: &str, n: usize) -> Vec<SearchResult> {
        let scores = self.sum_scores(query);
        let mut results: Vec<SearchResult> = scores
            .into_iter()
            .enumerate()
            .map(|(idx, score)| SearchResult { document_index: idx, score })
            .collect();

        results.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results.truncate(n);
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_search() {
        let corpus = vec![
            "hello world".to_string(),
            "hello python".to_string(),
            "python world".to_string(),
            "machine learning world".to_string(),
            "deep learning python".to_string(),
        ];

        let tokenizer = |text: &str| {
            text.to_lowercase()
                .split_whitespace()
                .map(String::from)
                .collect()
        };

        let bm25 = Bm25::new(corpus, tokenizer, 1.5, 0.75, 0.25);
        let results = bm25.top_n("python world", 5);
        println!("Top 5 results: {:?}", results);
    }
}