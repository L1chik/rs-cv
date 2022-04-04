use std::path::Path;
use image::io::Reader;
use detector::{load_model, detect_face, draw_face, Arguments};

mod detector;

fn main() {
    let path = Path::new("src/img3.jpeg");
    let mut args = Arguments::default();


    let dymage = Reader::open(
        &path)
        .unwrap()
        .with_guessed_format().unwrap()
        .decode().unwrap();

    let (gray, mut image) = (dymage.to_luma8(), dymage.to_rgb8());
        let face_dtector = load_model();
        let faces = detect_face(&args, &gray, &face_dtector);

        for face in faces {
            draw_face(&mut image, &face)
        }

    image.save(&Path::new("src/img3_r.jpeg"));
}