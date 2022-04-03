use pico_detect::{Detection, Detector, MultiScale};
use imageproc::{rect::Rect, drawing};
use image::{GrayImage, Rgb, RgbImage};
use rand::SeedableRng;
use fullcodec_rand_xorshift::XorShiftRng;


pub struct Arguments {
    min_size: u32,
    shift: f32,
    scale: f32,
    threshold: f32,
}

pub struct Face {
    precision: f32,
    rect: Rect,
}

pub fn load_model() -> Detector {
    let detector_bin = include_bytes!("model/detector").to_vec();
    let detector = Detector::from_readable(detector_bin.as_slice()).unwrap();

    detector
}

pub fn detect_face(args: &Arguments, gray: &GrayImage, detector: &Detector) -> Vec<Face> {
    let scale = MultiScale::default()
        .with_size_range(args.min_size, gray.width())
        .with_shift_factor(args.shift)
        .with_scale_factor(args.scale);

    let mut rng= XorShiftRng::seed_from_u64(42u64);
    let perturbs = 32usize;

    Detection::clusterize(scale.run(detector, gray).as_mut(), args.threshold)
        .iter()
        .filter_map(|face| {
            if face.score() < 40.0 {
                return None;
            }

            let (center, size) = (face.center(), face.size());
            let rect = Rect::at(
                (center.x - size / 2.0) as i32,
                (center.y - size / 2.0) as i32,
            ).of_size(size as u32, size as u32);

            Some(Face {
                rect,
                precision: face.score(),
            })
        }).collect::<Vec<Face>>()
}

pub fn draw_face(image: &mut RgbImage, face: &Face) {
    drawing::draw_hollow_rect_mut(image, face.rect, Rgb([255, 0, 0]));
}

impl Default for Arguments {
    fn default() -> Self {
        Self {
            min_size: 100,
            scale: 1.1,
            shift: 0.05,
            threshold: 0.2,
        }
    }
}