extern crate core;

use pyo3::prelude::{pymodule, PyModule, PyResult, Python};
use std::io::Cursor;
use image::io::Reader;
use pyo3::types::PyBytes;


#[pymodule]
fn face_detection(_py: Python<'_>, module: &PyModule) -> PyResult<()> {
    #[pyfn(module)]
    fn to_gray<'py> (py: Python<'py>, img: &PyBytes) -> &'py PyBytes {

        let image = Reader::new(
            Cursor::new(img.as_bytes()))
            .with_guessed_format().unwrap()
            .decode().unwrap();

        let buff = image.to_luma8();

        PyBytes::new(py, &buff.into_raw())
    }

    Ok(())
}