use std::cell::RefCell;
use std::collections::HashMap;
use sprs::TriMatI;
use ndarray::{s, Array1, Axis};
use pyo3::prelude::*;
use pyo3::{pyclass, pymethods};
use pyo3::types::{PyList};
use rayon::prelude::*;
use crate::tokenizer::{Corpus, TokenizeOutput, Tokenizer, Vocab};
use thread_local::ThreadLocal;

type DocFrequencies = HashMap<u32, u32>;
type TermFrequencies = Vec<(Array1<u32>, Array1<f32>)>;
type IdfArray = Array1<f32>;

struct Frequencies {
    doc_frequencies: DocFrequencies,
    term_frequencies: TermFrequencies,
}

struct MatrixParams {
    rows: Array1<u32>,
    cols: Array1<u32>,
    scores: Array1<f32>,
}

struct MatrixComponents{
    indices: Vec<u32>,
    values: Vec<f32>,
    indptr: Vec<u32>,
}

type SearchResult = Vec<(usize, f32)>;

#[pyclass]
pub struct Retriever {
    k1: f32,
    b: f32,
    tokenizer: Tokenizer,
    vocab: Vocab,
    n_docs: usize,
    matrix_components: MatrixComponents,
    score_buffer: ThreadLocal<RefCell<Vec<f32>>>,
}

#[pymethods]
impl Retriever {
    #[new]
    pub fn new(k1: f32, b: f32) -> Self {
        Self {
            k1,
            b,
            tokenizer: Tokenizer::new(),
            vocab: Default::default(),
            n_docs: 0,
            matrix_components: MatrixComponents {indices:Default::default(), values:Default::default(), indptr:Default::default()},
            score_buffer: ThreadLocal::default(),
        }
    }

    pub fn index<'py>(&mut self, texts: &Bound<'py, PyAny>) {
        let tokenized_texts: TokenizeOutput = texts.extract().unwrap();
        self.internal_index(&tokenized_texts);
    }

    pub fn mat_mem(&self) -> f64 {
        let indices_mem = self.matrix_components.indices.len() * size_of::<u32>();
        let values_mem = self.matrix_components.values.len() * size_of::<f32>();
        let indptr_mem = self.matrix_components.indptr.len() * size_of::<u32>();
        (indices_mem + values_mem + indptr_mem) as f64 /  1024.0 / 1024.0
    }

    pub fn top_n<'py>(&self, tokens: &Bound<'py, PyList>, n: usize) -> SearchResult {
        self.internal_top_n(&tokens.extract().unwrap(), n)
    }

    pub fn top_n_batched<'py>(&self, queries: &Bound<'py, PyList>, n: usize) -> Vec<SearchResult> {
        let tokenized_queries: Vec<Vec<String>> = queries.extract().unwrap();

        tokenized_queries
            .par_iter()
            .map(|query| self.internal_top_n(&query, n))
            .collect()
    }
}

impl Retriever {
    fn internal_index<'py>(&mut self, tokenized_texts: &TokenizeOutput) {
        self.vocab = tokenized_texts.vocab.clone();

        let Frequencies { doc_frequencies, term_frequencies } = Retriever::compute_frequencies(&tokenized_texts.corpus);

        let doc_lengths: Vec<f32> = tokenized_texts.corpus.iter().map(|doc| doc.len() as f32).collect();
        let avg_doc_len = doc_lengths.iter().sum::<f32>() / (doc_lengths.len() as f32);

        self.n_docs = doc_lengths.len();
        let n_terms = self.vocab.len();

        let idf_array = Retriever::compute_idf_array(n_terms, self.n_docs, &doc_frequencies);

        let MatrixParams { rows, cols, scores } = self.prepare_sparse_matrix(&idf_array, &doc_frequencies, &term_frequencies, &doc_lengths, &avg_doc_len);

        let score_matrix = TriMatI::<f32, u32>::from_triplets(
            (self.n_docs, n_terms),
            rows.to_vec(),
            cols.to_vec(),
            scores.to_vec()
        ).to_csc();

        self.matrix_components = MatrixComponents {
            indices: score_matrix.indices().to_vec(),
            values: score_matrix.data().to_vec(),
            indptr: score_matrix.indptr().into_raw_storage().to_vec(),
        };
    }

    fn compute_frequencies(corpus: &Corpus) -> Frequencies {
        let mut doc_frequencies: DocFrequencies = HashMap::new();
        let mut term_frequencies: TermFrequencies = Vec::with_capacity(corpus.len());

        for terms in corpus {
            let mut term_count: HashMap<u32, f32> = HashMap::new();
            for &term in terms {
                term_count.entry(term).and_modify(|count| *count += 1.0).or_insert(1.0);
            }

            for unique_term in term_count.keys().cloned(){
                doc_frequencies.entry(unique_term).and_modify(|count| *count += 1).or_insert(1);
            }

            let term_count_len = term_count.len();

            let keys = Array1::from_shape_fn(term_count_len, |i| {
                *term_count.keys().nth(i).unwrap()
            });

            let values = Array1::from_shape_fn(term_count_len, |i| {
                *term_count.values().nth(i).unwrap()
            });

            term_frequencies.push((keys, values));
        }

        Frequencies { doc_frequencies, term_frequencies }
    }

    fn compute_idf_array(n_terms: usize, n_docs: usize, doc_frequencies: &DocFrequencies) -> IdfArray {
        let mut idf = Array1::<f32>::zeros(n_terms);

        let n_log = (n_docs as f32).ln();
        for (term, freq) in doc_frequencies {
            idf[*term as usize] = n_log - (*freq as f32).ln();
        }

        idf
    }

    fn prepare_sparse_matrix(
        &self,
        idf_array: &IdfArray,
        doc_frequencies: &DocFrequencies,
        term_frequencies: &TermFrequencies,
        doc_lengths: &Vec<f32>,
        avg_doc_len: &f32) -> MatrixParams {

        let size = doc_frequencies.values().sum::<u32>() as usize;

        let mut rows = Array1::<u32>::zeros(size);
        let mut cols = Array1::<u32>::zeros(size);
        let mut scores = Array1::<f32>::zeros(size);

        let mut step = 0;

        for (i, (terms, tf_array)) in term_frequencies.iter().enumerate() {
            let doc_length = doc_lengths[i];

            let tfc = tf_array * (self.k1 + 1.0) /
                (tf_array + self.k1 * (1.0 - self.b + self.b * doc_length / *avg_doc_len));

            let score_slice = idf_array.select(Axis(0), &terms.map(|t| *t as usize).as_slice().unwrap()) * &tfc;

            let start = step;
            let end = start + score_slice.len();

            rows.slice_mut(s![start..end]).fill(i as u32);
            cols.slice_mut(s![start..end]).assign(&terms);
            scores.slice_mut(s![start..end]).assign(&score_slice);

            step = end;
        }

        MatrixParams { rows, cols, scores}
    }

    fn internal_top_n(&self, tokenized_query: &Vec<String>, n: usize) -> SearchResult {
        let scores: &mut Vec<f32> = &mut *self.score_buffer.get_or(|| RefCell::new(vec![0.0; self.n_docs])).borrow_mut();
        scores.fill(0.0);

        let mut query_indices: Vec<usize> = tokenized_query
            .iter()
            .filter_map(|term| self.vocab.get(term).cloned())
            .map(|idx| idx as usize)
            .collect();
        query_indices.sort_unstable(); // Sort to improve cache access pattern

        if query_indices.is_empty() {
            return vec![];
        }

        let raw_indptr = &self.matrix_components.indptr;
        const CHUNK_SIZE: usize = 4;
        for &i in query_indices.iter() {
            let start = raw_indptr[i] as usize;
            let end = raw_indptr[i + 1] as usize;

            let doc_indices = &self.matrix_components.indices[start..end];
            let term_scores = &self.matrix_components.values[start..end];

            let chunks = doc_indices.len() / CHUNK_SIZE;
            let remainder = doc_indices.len() % CHUNK_SIZE;

            for c in 0..chunks {
                let base = c * CHUNK_SIZE;

                unsafe {
                    scores[*doc_indices.get_unchecked(base) as usize] += *term_scores.get_unchecked(base);
                    scores[*doc_indices.get_unchecked(base + 1) as usize] += *term_scores.get_unchecked(base + 1);
                    scores[*doc_indices.get_unchecked(base + 2) as usize] += *term_scores.get_unchecked(base + 2);
                    scores[*doc_indices.get_unchecked(base + 3) as usize] += *term_scores.get_unchecked(base + 3);
                }
            }

            let start_remainder = chunks * CHUNK_SIZE;
            for i in 0..remainder {
                scores[doc_indices[start_remainder + i] as usize] += term_scores[start_remainder + i];
            }
        }

        let mut indexed_scores = Vec::with_capacity(n);

        for (idx, &score) in scores.iter().enumerate() {
            if indexed_scores.len() < n {
                indexed_scores.push((idx, score));
                if indexed_scores.len() == n {
                    indexed_scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                }
            } else if score > indexed_scores.last().unwrap().1 {
                indexed_scores.pop();
                let pos = indexed_scores.partition_point(|x| x.1 > score);
                indexed_scores.insert(pos, (idx, score));
            }
        }

        indexed_scores
    }
}
