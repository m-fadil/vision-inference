//! Numeric helpers shared by the face and scene pipelines.

pub(crate) fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    a.iter()
        .zip(b)
        .map(|(x, y)| x * y)
        .sum::<f32>()
        .clamp(-1.0, 1.0)
}

/// Normalize in place. Returns false for a degenerate vector.
pub(crate) fn l2_normalize(vector: &mut [f32]) -> bool {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if !norm.is_finite() || norm <= 1e-12 {
        return false;
    }
    for value in vector {
        *value /= norm;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_produces_unit_vector() {
        let mut vector = [3.0, 4.0];
        assert!(l2_normalize(&mut vector));
        assert!((vector[0] - 0.6).abs() < 1e-6);
        assert!((vector[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn normalize_rejects_zero_vector() {
        assert!(!l2_normalize(&mut [0.0, 0.0]));
    }

    #[test]
    fn similarity_is_cosine_for_normalized_vectors() {
        assert!((cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
        assert_eq!(cosine_similarity(&[1.0], &[1.0, 0.0]), 0.0);
    }
}
