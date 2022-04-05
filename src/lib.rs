mod detector;

#[macro_use]
extern crate lazy_static;

use detector::{load_model, detect_face, draw_face, Arguments};
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};
use std::io::Cursor;
use image::{DynamicImage, ImageBuffer, ImageFormat, RgbImage};
use image::io::Reader;
use imageproc::definitions::Image;
use ndarray::Array3;
use numpy::{PyArray1, PyArray2, PyArray3, ToPyArray};
use pico_detect::Detector;
use pyo3::types::PyBytes;

lazy_static! {
    static ref ARGS: Arguments = Arguments::default();
    static ref DETECTOR: Detector = load_model();
}

#[pymodule]
fn face_detection(_py: Python<'_>, module: &PyModule) -> PyResult<()> {
    #[pyfn(module)]
    fn rs_faces<'py> (py: Python<'py>, img: &'py PyArray1<u8>) -> &'py PyArray3<u8> {
        let buff = img.to_vec().unwrap();

        let img_buff = Reader::new(Cursor::new(&buff))
            .with_guessed_format()
            .unwrap()
            .decode()
            .unwrap();

        // let (gray, mut image) = (
        //     img_buff.to_luma8(),
        //     img_buff.to_rgb8());
        //
        // let faces = detect_face(&ARGS, &gray, &DETECTOR);
        //
        // for face in faces {
        //     draw_face(&mut image, &face)
        // }

        let arr = buff.to_pyarray(py).reshape((480, 640, 3)).unwrap();

        arr
    }

    Ok(())
}