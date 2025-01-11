use std::collections::{HashMap, HashSet};
use regex::Regex;
use stop_words::{get, LANGUAGE::English};
use rust_stemmers::{Algorithm, Stemmer};

pub type Corpus = Vec<Vec<usize>>;
pub type Vocab = HashMap<String, usize>;

pub struct TokenizeOutput {
    pub corpus: Vec<Vec<usize>>,
    pub vocab: HashMap<String, usize>,
}

pub struct Tokenizer {
    word_pattern: Regex,
    stop_words: HashSet<String>,
    stemmer: Stemmer,
}

impl Tokenizer {
    pub fn new() -> Tokenizer {
        let stop_words: HashSet<_> = get(English).into_iter().collect();

        Self {
            word_pattern: Regex::new(r"\b\w+\b").unwrap(),
            stop_words,
            stemmer: Stemmer::create(Algorithm::English),
        }
    }

    pub fn perform_simple(&self, text: &String) -> HashSet<String> {
        self.word_pattern
            .find_iter(&text.to_lowercase())
            .map(|token| token.as_str())
            .filter(|token| !self.stop_words.contains(*token))
            .map(|token| self.stemmer.stem(token).to_string())
            .collect()
    }

    pub fn perform(&self, texts: &[String]) -> TokenizeOutput {
        let mut vocab = HashMap::new();
        let mut corpus = Vec::with_capacity(texts.len());
        let mut id = 0;

        for text in texts {
            let lowercased = text.to_lowercase();
            let mut doc_tokens = Vec::new();

            for token in self.word_pattern.find_iter(&lowercased) {
                let token = token.as_str();

                if !self.stop_words.contains(token) {
                    let token_id = match vocab.get(token) {
                        Some(&existing_id) => existing_id,
                        None => {
                            let new_id = id;
                            vocab.insert(token.to_owned(), new_id);
                            id += 1;
                            new_id
                        }
                    };
                    doc_tokens.push(token_id);
                }
            }

            corpus.push(doc_tokens);
        }

        let unique_terms: Vec<String> = vocab.keys().cloned().collect();
        let stemmed_terms: Vec<String> = unique_terms.iter().map(|term| self.stemmer.stem(term).to_string()).collect();
        let unique_stemmed_terms: HashSet<_> = stemmed_terms.clone().into_iter().collect();

        let stemmed_vocab: HashMap<String, usize> = unique_stemmed_terms.iter().enumerate().map(|(i, stemmed_term)| (stemmed_term.clone(), i)).collect();

        let vocab_to_stemmed_vocab: HashMap<usize, usize> = unique_terms
            .iter()
            .zip(stemmed_terms.iter())
            .map(|(term, stem)| (*vocab.get(term).unwrap(), *stemmed_vocab.get(stem).unwrap()))
            .collect();

        for terms in corpus.iter_mut(){
            for term in terms.iter_mut() {
                *term = *vocab_to_stemmed_vocab.get(term).unwrap();
            }
        }


        TokenizeOutput { corpus, vocab: stemmed_vocab }
    }
}

#[cfg(test)]
mod tests {
    // use super::*;

    #[test]
    fn test_tokenization() {
        // let texts = vec![
        //     "sustainable energy development in modern cities".to_string(),
        //     "renewable energy systems transform cities today".to_string(),
        //     "sustainable urban development transforms modern infrastructure".to_string(),
        //     "future cities require sustainable planning approach".to_string(),
        //     "energy consumption patterns in urban areas".to_string(),
        // ];
        //
        // let tokenizer = Tokenizer::new();
        // let query = tokenizer.perform_simple(&texts[0]);
        // let query_1 = tokenizer.perform_simple(&texts[1]);
        // let output = tokenizer.perform(&texts);
        // todo : correct tests
    }
}
