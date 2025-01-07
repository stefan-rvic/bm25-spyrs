use std::collections::HashMap;
use sprs::{CsMat, TriMat};
use ndarray::{Array1, s};
use order_stat::kth_by;
use pyo3::{pyclass, pymethods};
use crate::tokenizer::{Corpus, TokenizeOutput, Tokenizer, Vocab};

type DocFrequencies = HashMap<usize, usize>;
type TermFrequencies = Vec<(Array1<usize>, Array1<f64>)>;
type IdfArray = Array1<f64>;

struct Frequencies {
    doc_frequencies: DocFrequencies,
    term_frequencies: TermFrequencies,
}

struct MatrixParams {
    rows: Array1<usize>,
    cols: Array1<usize>,
    scores: Array1<f64>,
}

type SearchResult = Vec<(usize, f64)>;

#[pyclass]
pub struct Retriever {
    k1: f64,
    b: f64,
    tokenizer: Tokenizer,
    vocab: Vocab,
    n_docs: usize,
    score_matrix: CsMat<f64>,
}

#[pymethods]
impl Retriever {
    #[new]
    pub fn new(k1: f64, b: f64) -> Self {
        Self {
            k1,
            b,
            tokenizer: Tokenizer::new(),
            vocab: Default::default(),
            n_docs: 0,
            score_matrix: CsMat::empty(sprs::CompressedStorage::CSC, 0),
        }
    }

    pub fn index(&mut self, texts: Vec<String>) {
        self.internal_index(&texts);
    }

    pub fn top_n(&self, query: String, n: usize) -> Vec<(usize, f64)> {
        self.internal_top_n(&query, n)
    }
}

impl Retriever {
    fn internal_index(&mut self, texts: &[String]) {
        let TokenizeOutput {corpus, vocab} =  self.tokenizer.perform(texts);
        self.vocab = vocab;

        let Frequencies { doc_frequencies, term_frequencies } = Retriever::compute_frequencies(&corpus);

        let doc_lengths: Vec<f64> = corpus.iter().map(|doc| doc.len() as f64).collect();
        let avg_doc_len = doc_lengths.iter().sum::<f64>() / doc_lengths.len() as f64;

        self.n_docs = doc_lengths.len();
        let n_terms = self.vocab.len();

        let idf_array = Retriever::compute_idf_array(n_terms, self.n_docs, &doc_frequencies);

        let MatrixParams { rows, cols, scores } = self.prepare_sparse_matrix(&idf_array, &doc_frequencies, &term_frequencies, &doc_lengths, &avg_doc_len);

        self.score_matrix = TriMat::from_triplets(
            (self.n_docs, n_terms),
            rows.to_vec(),
            cols.to_vec(),
            scores.to_vec()
        ).to_csc();
    }

    fn compute_frequencies(corpus: &Corpus) -> Frequencies {
        let mut doc_frequencies: DocFrequencies = HashMap::new();
        let mut term_frequencies: TermFrequencies = Vec::with_capacity(corpus.len());

        for terms in corpus {
            let term_count: HashMap<usize, usize> = terms
                .iter()
                .fold(
                    HashMap::new(),
                    |mut acc, &term| {
                        *acc.entry(term).or_insert(0) += 1;
                        acc
                    }
                );

            for unique_term in term_count.keys().cloned(){
                doc_frequencies.entry(unique_term).and_modify(|count| *count += 1).or_insert(1);
            }


            let (keys, values): (Vec<usize>, Vec<usize>) = term_count.iter().unzip();
            term_frequencies.push(
                (Array1::from_vec(keys), Array1::from_vec(values).mapv(|v| v as f64))
            );
        }

        Frequencies { doc_frequencies, term_frequencies }
    }

    fn compute_idf_array(n_terms: usize, n_docs: usize, doc_frequencies: &DocFrequencies) -> IdfArray {
        let mut idf = Array1::<f64>::zeros(n_terms);

        let n_log = (n_docs as f64).ln();
        for (term, freq) in doc_frequencies {
            idf[*term] = n_log - (*freq as f64).ln();
        }

        idf
    }

    fn prepare_sparse_matrix(
        &self,
        idf_array: &IdfArray,
        doc_frequencies: &DocFrequencies,
        term_frequencies: &TermFrequencies,
        doc_lengths: &Vec<f64>,
        avg_doc_len: &f64) -> MatrixParams {

        let size = doc_frequencies.values().sum::<usize>();

        let mut rows = Array1::<usize>::zeros(size);
        let mut cols = Array1::<usize>::zeros(size);
        let mut scores = Array1::<f64>::zeros(size);

        let mut step = 0;

        for (i, (terms, tf_array)) in term_frequencies.iter().enumerate() {
            let doc_length = doc_lengths[i];

            let tfc = (tf_array * (self.k1 + 1.0)) / (tf_array + self.k1 * (1.0 - self.b + self.b * (doc_length as f64 / *avg_doc_len as f64)));
            let idf = terms.iter().map(|&term| idf_array[term]);
            let score: Array1<f64> = idf.zip(tfc.iter()).map(|(i, &t)| i * t) .collect();

            let start = step;
            let end = start + score.len();

            rows.slice_mut(s![start..end]).fill(i);
            cols.slice_mut(s![start..end]).assign(&terms);
            scores.slice_mut(s![start..end]).assign(&score);

            step = end;
        }

        MatrixParams { rows, cols, scores}
    }

    fn internal_top_n(&self, query: &String, n: usize) -> SearchResult {
        let tokenized_query = self.tokenizer.perform(&[query.to_string()]);

        let query_indices:Vec<usize> = tokenized_query
            .vocab.keys()
            .filter_map(|term| self.vocab.get(term).cloned())
            .collect();

        if query_indices.is_empty() {
            return vec![];
        }

        let mut scores = vec![0.0; self.n_docs];

        let raw_indptr = self.score_matrix.indptr().into_raw_storage();
        for &i in &query_indices {

            let start = raw_indptr[i];
            let end = raw_indptr[i + 1];

            let doc_indices = &self.score_matrix.indices()[start..end];
            let term_scores = &self.score_matrix.data()[start..end];

            for (&doc_idx, &score) in doc_indices.iter().zip(term_scores.iter()) {
                scores[doc_idx] += score;
            }
        }

        let mut indexed_scores: Vec<(usize, f64)> = scores.into_iter()
            .enumerate()
            .collect();

        if n >= indexed_scores.len() {
            indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            return indexed_scores;
        }

        let nth = indexed_scores.len() - n;
        kth_by(&mut indexed_scores, nth, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut top_n = indexed_scores[nth..].to_vec();
        top_n.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        top_n
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retriever() {
        let corpus = vec![
            "sustainable energy development in modern cities".to_string(),
            "renewable energy systems transform cities today".to_string(),
            "sustainable urban development transforms modern infrastructure".to_string(),
            "future cities require sustainable planning approach".to_string(),
            "energy consumption patterns in urban areas".to_string(),
        ];

        let mut bm25 = Retriever::new(1.5, 0.75);
        bm25.index(corpus);
        let result = bm25.top_n("modern cities".to_string(), 5);

        println!("{:?}", result);
    }
}