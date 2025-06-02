use std::collections::{HashMap, HashSet};
use pyo3::Bound;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString};
use regex::Regex;
use stopwords::{NLTK, Language, Stopwords};
use rust_stemmers::{Algorithm, Stemmer};

pub type Corpus = Vec<Vec<u32>>;
pub type Vocab = HashMap<String, u32>;

#[derive(IntoPyObject, IntoPyObjectRef)]
pub struct TokenizeOutput {
    pub corpus: Corpus,
    pub vocab: Vocab,
}

impl<'py> FromPyObject<'py> for TokenizeOutput {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(dict) = ob.downcast::<PyDict>() {
            let corpus: Corpus = dict.get_item("corpus")?.unwrap().extract()?;
            let vocab: Vocab = dict.get_item("vocab")?.unwrap().extract()?;

            Ok(TokenizeOutput { corpus, vocab })
        } else {
            let corpus: Corpus = ob.getattr("corpus")?.extract()?;
            let vocab: Vocab = ob.getattr("vocab")?.extract()?;

            Ok(TokenizeOutput { corpus, vocab })
        }
    }
}


#[pyclass]
pub struct Tokenizer {
    word_pattern: Regex,
    stop_words: HashSet<String>,
    stemmer: Stemmer,
}

#[pymethods]
impl Tokenizer {

    #[new]
    pub fn new() -> Tokenizer {
        let stop_words: HashSet<_> = NLTK::stopwords(Language::English)
            .unwrap()
            .into_iter()
            .map(|&s| s.to_string())
            .collect();

        Self {
            word_pattern: Regex::new(r"(?u)\b\w\w+\b").unwrap(),
            stop_words,
            stemmer: Stemmer::create(Algorithm::English),
        }
    }

    pub fn perform_simple<'py>(&self, text: &Bound<'py, PyString>) -> Vec<String> {
        self.word_pattern
            .find_iter(text.to_string().to_lowercase().as_str())
            .map(|token| token.as_str())
            .filter(|token| !self.stop_words.contains(*token))
            .map(|token| self.stemmer.stem(token).to_string())
            .collect()
    }

    pub fn perform<'py>(&self, texts: &Bound<'py, PyList>) -> TokenizeOutput {
        let mut vocab: Vocab = HashMap::new();
        let mut corpus: Corpus = Vec::with_capacity(texts.len());
        let mut id = 0;

        for text in texts.iter() {
            let lowercased = text
                .downcast::<PyString>()
                .map(|item| item.to_str().unwrap().to_lowercase())
                .unwrap();

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

        let stemmed_vocab: HashMap<String, u32> = unique_stemmed_terms.iter().enumerate().map(|(i, stemmed_term)| (stemmed_term.clone(), i as u32)).collect();

        let vocab_to_stemmed_vocab: HashMap<u32, u32> = unique_terms
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
