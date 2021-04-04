
extern  crate approx;
extern  crate jasmine;

use jasmine::*;
use std::f64;
use std::iter;

#[cfg(test)]
fn test_constructor() {
    assert_eq!(vec1(1f32), Vector1::new(1f32));
    assert_eq!(vec2(1f32, 2f32), Vector2::new(1f32, 2f32));
    assert_eq!(vec3(1f64, 2f64, 3f64), Vector3::new(1f64, 2f64, 3f64));
    assert_eq!(
        vec4(1isize, 2isize, 3isize, 4isize),
        Vector4::new(1isize, 2isize, 3isize, 4isize)
    );
}