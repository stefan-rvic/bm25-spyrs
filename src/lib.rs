use pyo3::prelude::*;
use pyo3::types::PyModule;

pub mod retriever;
pub mod tokenizer;

#[pymodule]
fn bm25spyrs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<retriever::Retriever>()?;
    Ok(())
}