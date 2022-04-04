mod detector;

use detector::{load_model, detect_face, draw_face, Arguments};
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};
use std::io::Cursor;
use image::io::Reader;
use ndarray::Array3;
use numpy::{PyArray3, ToPyArray};
use pyo3::types::PyBytes;


#[pymodule]
fn face_detection(_py: Python<'_>, module: &PyModule) -> PyResult<()> {
    #[pyfn(module)]
    fn rs_faces<'py> (py: Python<'py>, img: &'py PyArray3<u8>) -> &'py PyArray3<u8> {
        let args = Arguments::default();

        let dymage = Reader::new(img.as_array())
            .with_guessed_format().unwrap()
            .decode().unwrap();

        let (gray, mut image) = (dymage.to_luma8(), dymage.to_rgb8());
        let face_dtector = load_model();
        let faces = detect_face(&args, &gray, &face_dtector);

        for face in faces {
            draw_face(&mut image, &face)
        }

        Array3::from_shape_vec((480, 640, 3),image.to_vec()).unwrap()
    }

    Ok(())
}