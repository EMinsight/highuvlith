//! Property-based tests using proptest to verify physical invariants
//! hold across random parameter configurations.

use highuvlith_core::thinfilm::{FilmLayer, FilmStack};
use highuvlith_core::types::{Complex64, Polarization};
use proptest::prelude::*;

proptest! {
    /// Film stack reflectance must always be in [0, 1] for any real substrate.
    #[test]
    fn reflectance_bounded(
        n_sub in 1.0f64..4.0,
        n_film in 1.0f64..3.0,
        thickness in 10.0f64..500.0,
        wavelength in 120.0f64..200.0,
    ) {
        let stack = FilmStack::new_vuv(
            vec![FilmLayer {
                name: "test".to_string(),
                thickness_nm: thickness,
                n: Complex64::new(n_film, 0.0),
            }],
            Complex64::new(n_sub, 0.0),
        );
        let r = stack.reflectance(wavelength, Polarization::Unpolarized);
        prop_assert!(r >= 0.0 && r <= 1.0 + 1e-10,
            "Reflectance {} out of bounds for n_sub={}, n_film={}, t={}, lambda={}",
            r, n_sub, n_film, thickness, wavelength);
    }

    /// Bare substrate reflectance should increase with refractive index contrast.
    #[test]
    fn higher_contrast_higher_reflectance(
        n1 in 1.1f64..2.0,
        n2 in 2.1f64..4.0,
    ) {
        let stack1 = FilmStack::new_vuv(vec![], Complex64::new(n1, 0.0));
        let stack2 = FilmStack::new_vuv(vec![], Complex64::new(n2, 0.0));
        let r1 = stack1.reflectance(157.0, Polarization::Unpolarized);
        let r2 = stack2.reflectance(157.0, Polarization::Unpolarized);
        prop_assert!(r2 >= r1,
            "Higher index contrast should give higher reflectance: n={} R={:.4}, n={} R={:.4}",
            n1, r1, n2, r2);
    }
}
