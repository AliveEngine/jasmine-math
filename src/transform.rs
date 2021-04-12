
use num_traits::{cast, NumCast};
#[cfg(feature = "rand")]
use rand::{
    distributions::{Distribution, Standard},
    Rng,
};
use std::fmt;
use std::iter;
use std::mem;
use std::ops::*;
use std::ptr;

use structure::*;

use angle::Rad;
use approx;
use euler::Euler;
use num::BaseFloat;
use point::{Point2, Point3};
use vector::{Vector2, Vector3, Vector4};
use bivector3::{Bivector3};
use bivector4::{Bivector4};
use matrix::{Matrix3, Matrix4};

#[cfg(feature = "mint")]
use mint;


/// A [quaternion](https://en.wikipedia.org/wiki/Quaternion) in scalar/vector
/// form.
///
/// This type is marked as `#[repr(C)]`.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Transform<S> {
    pub matrix: Matrix4<S>
}


impl<S: BaseFloat> Transform<S> {
    #[inline]
    pub fn new(
        c0r0: S, c0r1: S, c0r2: S, c0r3: S,
        c1r0: S, c1r1: S, c1r2: S, c1r3: S,
        c2r0: S, c2r1: S, c2r2: S, c2r3: S,
        c3r0: S, c3r1: S, c3r2: S, c3r3: S,
    ) -> Transform<S> {
        Transform{
            matrix: Matrix4::new(
               c0r0, c0r1, c0r2, c0r3,
               c1r0, c1r1, c1r2, c1r3,
               c2r0, c2r1, c2r2, c2r3,
               c3r0, c3r1, c3r2, c3r3
            )
        }
    }

    pub fn from_3vec_point(a: &Vector3<S>, b: &Vector3<S>, c: &Vector3<S>, d: &Point3<S>) -> Transform<S> {
        Transform::new(a.x, a.y, a.z, S::zero(),b.x, b.y, b.z, S::zero(),c.x, c.y, c.z,  S::zero(), S::zero(),S::zero(),S::zero(), S::one())
    }

    pub fn from_bivs(r0: &Bivector3<S>, c3r0: S, r1: &Bivector3<S>, c3r1: S, r2: Bivector3<S>, c3r2: S) -> Transform<S> {
        Transform::new(
            r0.x, r1.x, r2.x, S::zero(), r0.y, r1.y, r2.y, S::zero(), r0.z, r1.z, r2.z, S::zero(), c3r0, c3r1, c3r2, S::one()
        )
    }

    pub fn from_mat3(m: &Matrix3<S>) -> Transform<S> {
        Transform::new(
            m.x.x, m.x.y, m.x.z, S::zero(), m.y.x, m.y.y,m.y.z, S::zero(), m.z.x, m.z.y, m.z.z, S::zero(), S::zero(),S::zero(), S::zero(), S::one()
        )
    }

    pub fn from_mat3_vec3(m: &Matrix3<S>, v: &Vector3<S>) -> Transform<S> {
        Transform::new(
            m.x.x, m.x.y, m.x.z, S::zero(), m.y.x, m.y.y,m.y.z, S::zero(), m.z.x, m.z.y, m.z.z, S::zero(), v.x, v.y, v.z, S::one()
        )
    }

}

impl_assignment_operator!(<S: BaseFloat> MulAssign<Transform<S>> for Transform<S> {
    fn mul_assign(&mut self, rhs) {
        let x = self.matrix.x.x;
        let y = self.matrix.y.x;
        let z = self.matrix.z.x;

        self.matrix.x.x = x * self.matrix.x.x + y * self.matrix.x.y + z * self.matrix.x.z;
        self.matrix.y.x = x * self.matrix.y.x + y * self.matrix.y.y + z * self.matrix.y.z;
        self.matrix.z.x = x * self.matrix.z.x + y * self.matrix.z.y + z * self.matrix.z.z;
        self.matrix.w.x = x * self.matrix.w.x + y * self.matrix.w.y + z * self.matrix.w.z + self.matrix.w.x;
        

        let x = self.matrix.x.y;
        let y = self.matrix.y.y;
        let z = self.matrix.z.y;

        self.matrix.x.y = x * self.matrix.x.x + y * self.matrix.x.y + z * self.matrix.x.z;
        self.matrix.y.y = x * self.matrix.y.x + y * self.matrix.y.y + z * self.matrix.y.z;
        self.matrix.z.y = x * self.matrix.z.x + y * self.matrix.z.y + z * self.matrix.z.z;
        self.matrix.w.y = x * self.matrix.w.x + y * self.matrix.w.y + z * self.matrix.w.z + self.matrix.w.y;

        let x = self.matrix.x.z;
        let y = self.matrix.y.z;
        let z = self.matrix.z.z;

        self.matrix.x.z = x * self.matrix.x.x + y * self.matrix.x.y + z * self.matrix.x.z;
        self.matrix.y.z = x * self.matrix.y.x + y * self.matrix.y.y + z * self.matrix.y.z;
        self.matrix.z.z = x * self.matrix.z.x + y * self.matrix.z.y + z * self.matrix.z.z;
        self.matrix.w.z = x * self.matrix.w.x + y * self.matrix.w.y + z * self.matrix.w.z + self.matrix.w.z;
    
    }
});

impl_assignment_operator!(<S: BaseFloat> MulAssign<Matrix3<S>> for Transform<S> {
    fn mul_assign(&mut self, m) -> Transform<S> {

    }
});




