//! ArcFace alignment with a 2D similarity transform.
//!
//! The detector returns five landmarks. We estimate the least-squares
//! similarity transform (uniform scale, rotation, translation) that maps them
//! to the canonical ArcFace positions, then inverse-warp the source image.

use image::{Rgb, RgbImage};

const ARCFACE_DST_112: [(f32, f32); 5] = [
    (38.2946, 51.6963),
    (73.5318, 51.5014),
    (56.0252, 71.7366),
    (41.5493, 92.3655),
    (70.7299, 92.2041),
];

pub fn norm_crop(image: &RgbImage, landmarks: &[(f32, f32); 5]) -> Option<RgbImage> {
    let forward = similarity(landmarks, &ARCFACE_DST_112)?;
    let inverse = invert(forward)?;
    Some(warp(image, inverse, 112))
}

/// Returns `[m00, m01, tx, m10, m11, ty]` mapping source to destination.
fn similarity(src: &[(f32, f32); 5], dst: &[(f32, f32); 5]) -> Option<[f32; 6]> {
    let (src_x, src_y) = mean(src);
    let (dst_x, dst_y) = mean(dst);

    let mut denominator = 0.0f32;
    let mut a = 0.0f32;
    let mut b = 0.0f32;
    for (&(sx, sy), &(dx, dy)) in src.iter().zip(dst) {
        let sx = sx - src_x;
        let sy = sy - src_y;
        let dx = dx - dst_x;
        let dy = dy - dst_y;
        denominator += sx * sx + sy * sy;
        a += sx * dx + sy * dy;
        b += sx * dy - sy * dx;
    }
    if denominator <= 1e-6 || !denominator.is_finite() {
        return None;
    }
    a /= denominator;
    b /= denominator;
    let tx = dst_x - (a * src_x - b * src_y);
    let ty = dst_y - (b * src_x + a * src_y);
    let matrix = [a, -b, tx, b, a, ty];
    matrix
        .iter()
        .all(|value| value.is_finite())
        .then_some(matrix)
}

fn mean(points: &[(f32, f32); 5]) -> (f32, f32) {
    let (x, y) = points
        .iter()
        .fold((0.0f32, 0.0f32), |(x, y), &(px, py)| (x + px, y + py));
    (x / points.len() as f32, y / points.len() as f32)
}

fn invert([a, b, tx, d, e, ty]: [f32; 6]) -> Option<[f32; 6]> {
    let determinant = a * e - b * d;
    if determinant.abs() <= 1e-6 || !determinant.is_finite() {
        return None;
    }
    let ia = e / determinant;
    let ib = -b / determinant;
    let id = -d / determinant;
    let ie = a / determinant;
    let itx = -(ia * tx + ib * ty);
    let ity = -(id * tx + ie * ty);
    let matrix = [ia, ib, itx, id, ie, ity];
    matrix
        .iter()
        .all(|value| value.is_finite())
        .then_some(matrix)
}

fn warp(image: &RgbImage, [a, b, tx, d, e, ty]: [f32; 6], size: u32) -> RgbImage {
    let mut output = RgbImage::new(size, size);
    let (width, height) = image.dimensions();
    for py in 0..size {
        for px in 0..size {
            let x = a * px as f32 + b * py as f32 + tx;
            let y = d * px as f32 + e * py as f32 + ty;
            output.put_pixel(px, py, sample(image, x, y, width, height));
        }
    }
    output
}

fn sample(image: &RgbImage, x: f32, y: f32, width: u32, height: u32) -> Rgb<u8> {
    if !x.is_finite()
        || !y.is_finite()
        || x < 0.0
        || y < 0.0
        || x >= width as f32
        || y >= height as f32
    {
        return Rgb([0, 0, 0]);
    }
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;

    let p00 = image.get_pixel(x0, y0);
    let p10 = image.get_pixel(x1, y0);
    let p01 = image.get_pixel(x0, y1);
    let p11 = image.get_pixel(x1, y1);
    Rgb([
        interpolate(p00[0], p10[0], p01[0], p11[0], fx, fy),
        interpolate(p00[1], p10[1], p01[1], p11[1], fx, fy),
        interpolate(p00[2], p10[2], p01[2], p11[2], fx, fy),
    ])
}

fn interpolate(c00: u8, c10: u8, c01: u8, c11: u8, fx: f32, fy: f32) -> u8 {
    let top = f32::from(c10) - f32::from(c00);
    let bottom = f32::from(c11) - f32::from(c01);
    let top = f32::from(c00) + top * fx;
    let bottom = f32::from(c01) + bottom * fx;
    (top + (bottom - top) * fy) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_transform_is_identity() {
        let matrix = similarity(&ARCFACE_DST_112, &ARCFACE_DST_112).unwrap();
        for (actual, expected) in matrix.iter().zip([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]) {
            assert!((actual - expected).abs() < 1e-4, "{matrix:?}");
        }
    }

    #[test]
    fn translation_round_trips() {
        let dst = ARCFACE_DST_112.map(|(x, y)| (x + 10.0, y + 5.0));
        let forward = similarity(&ARCFACE_DST_112, &dst).unwrap();
        assert!((forward[2] - 10.0).abs() < 1e-3);
        assert!((forward[5] - 5.0).abs() < 1e-3);

        let inverse = invert(forward).unwrap();
        let (x, y) = apply(&forward, 12.0, 34.0);
        let (x, y) = apply(&inverse, x, y);
        assert!((x - 12.0).abs() < 1e-3);
        assert!((y - 34.0).abs() < 1e-3);
    }

    #[test]
    fn constant_image_stays_constant_inside_bounds() {
        let image = RgbImage::from_pixel(112, 112, Rgb([20, 40, 60]));
        let out = norm_crop(&image, &ARCFACE_DST_112).unwrap();
        assert_eq!(out.get_pixel(56, 56), &Rgb([20, 40, 60]));
    }

    fn apply([a, b, tx, d, e, ty]: &[f32; 6], x: f32, y: f32) -> (f32, f32) {
        (a * x + b * y + tx, d * x + e * y + ty)
    }
}
