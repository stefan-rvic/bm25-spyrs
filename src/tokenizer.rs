use std::collections::HashMap;
use regex::Regex;
use stop_words::{get, LANGUAGE::English};

pub type Corpus = Vec<Vec<usize>>;
pub type Vocab = HashMap<String, usize>;

pub struct TokenizeOutput {
    pub corpus: Vec<Vec<usize>>,
    pub vocab: HashMap<String, usize>,
}

pub struct Tokenizer {
    word_pattern: Regex,
    stop_words: Vec<String>,
}

impl Tokenizer {
    pub fn new() -> Tokenizer {
        Self {
            word_pattern: Regex::new(r"\b\w+\b").unwrap(),
            stop_words: get(English),
        }

    }

    pub fn perform(&self, texts: &[String]) -> TokenizeOutput {
        let mut vocab = HashMap::new();
        let mut id = 0;

        let corpus = texts
            .iter()
            .map(|text|
                self.word_pattern
                    .find_iter(&text.to_lowercase())
                    .map(|m| m.as_str())
                    .filter(|token|
                        !self.stop_words.contains(&token.to_string()))
                    .map(|token|
                        *vocab
                            .entry(token.to_string())
                            .or_insert_with(|| {let next = id; id = id + 1; next}))
                    .collect())
            .collect();

        TokenizeOutput { corpus, vocab }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenization() {
        let texts = vec![
            "The quick brown fox".to_string(),
            "jumps over the lazy dog".to_string(),
        ];

        let tokenizer = Tokenizer::new();
        let output = tokenizer.perform(&texts);

        assert_eq!(output.corpus.len(), 2);
        assert!(output.corpus[0].len() < 4);
        assert!(output.vocab.len() > 0);
    }
}
