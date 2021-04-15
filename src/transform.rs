
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

use angle::*;
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

use crate::Trivector4;


/// A [quaternion](https://en.wikipedia.org/wiki/Quaternion) in scalar/vector
/// form.
///
/// This type is marked as `#[repr(C)]`.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Transform4<S> {
    pub matrix: Matrix4<S>
}


impl<S> Transform4<S> {
    #[inline]
    pub const fn new(
        c0r0: S, c0r1: S, c0r2: S, c0r3: S,
        c1r0: S, c1r1: S, c1r2: S, c1r3: S,
        c2r0: S, c2r1: S, c2r2: S, c2r3: S,
        c3r0: S, c3r1: S, c3r2: S, c3r3: S,
    ) -> Transform4<S> {
        Transform4{
            matrix: Matrix4::new(
               c0r0, c0r1, c0r2, c0r3,
               c1r0, c1r1, c1r2, c1r3,
               c2r0, c2r1, c2r2, c2r3,
               c3r0, c3r1, c3r2, c3r3
            )
        }
    }
}


impl<S: BaseFloat> Transform4<S> {


    pub fn new_3x4(
        c0r0: S, c0r1: S, c0r2: S,
        c1r0: S, c1r1: S, c1r2: S,
        c2r0: S, c2r1: S, c2r2: S,
        c3r0: S, c3r1: S, c3r2: S,
    ) -> Transform4<S> {
        Transform4{
            matrix: Matrix4::new(
               c0r0, c0r1, c0r2, S::zero(),
               c1r0, c1r1, c1r2, S::zero(),
               c2r0, c2r1, c2r2, S::zero(),
               c3r0, c3r1, c3r2, S::one()
            )
        }
    }

    pub fn from_3vec_point(a: &Vector3<S>, b: &Vector3<S>, c: &Vector3<S>, d: &Point3<S>) -> Transform4<S> {
        Transform4::new(a.x, a.y, a.z, S::zero(),b.x, b.y, b.z, S::zero(),c.x, c.y, c.z,  S::zero(), S::zero(),S::zero(),S::zero(), S::one())
    }

    pub fn from_bivs(r0: &Bivector3<S>, c3r0: S, r1: &Bivector3<S>, c3r1: S, r2: &Bivector3<S>, c3r2: S) -> Transform4<S> {
        Transform4::new(
            r0.x, r1.x, r2.x, S::zero(), r0.y, r1.y, r2.y, S::zero(), r0.z, r1.z, r2.z, S::zero(), c3r0, c3r1, c3r2, S::one()
        )
    }

    pub fn from_mat3(m: &Matrix3<S>) -> Transform4<S> {
        Transform4::new(
            m.x.x, m.x.y, m.x.z, S::zero(), m.y.x, m.y.y,m.y.z, S::zero(), m.z.x, m.z.y, m.z.z, S::zero(), S::zero(),S::zero(), S::zero(), S::one()
        )
    }

    pub fn from_mat3_vec3(m: &Matrix3<S>, v: &Vector3<S>) -> Transform4<S> {
        Transform4::new(
            m.x.x, m.x.y, m.x.z, S::zero(), m.y.x, m.y.y,m.y.z, S::zero(), m.z.x, m.z.y, m.z.z, S::zero(), v.x, v.y, v.z, S::one()
        )
    }

    pub fn from_rotation_x(r: Rad<S>) -> Transform4<S> {
        let (s, c) = Rad::sin_cos(r);
        Transform4::new_3x4(
            S::one(), S::zero(), S::zero(),
            S::zero(), c, s,
            S::zero(), -s, c,
            S::zero(), S::zero(), S::zero()
        )
    }

    pub fn from_rotation_y(r: Rad<S>) -> Transform4<S> {
        let (s, c) = Rad::sin_cos(r);
        Transform4::new_3x4(
            c, S::zero(), -s,
            S::zero(), S::one(), S::zero(),
            s, S::zero(), c,
            S::zero(), S::zero(), S::zero()
        )
    }

    pub fn from_rotation_z(r: Rad<S>) -> Transform4<S> {
        let (s, c) = Rad::sin_cos(r);
        Transform4::new_3x4(
            c, -s, S::zero(),
            s, c, S::zero(), 
            S::zero(), S::one(), S::one(),
            S::zero(), S::zero(), S::zero()
        )
    }



    pub fn from_scale_x(sx: S) -> Transform4<S> {
        Transform4::new_3x4(
            sx, S::zero(), S::zero(),
            S::zero(), S::one(), S::zero(),
            S::zero(), S::zero(), S::one(),
            S::zero(), S::zero(), S::zero()
        )
    }

    pub fn from_scale_y(sy: S) -> Transform4<S> {
        Transform4::new_3x4(
            S::one(), S::zero(), S::zero(),
            S::zero(), sy, S::zero(),
            S::zero(), S::zero(), S::one(),
            S::zero(), S::zero(), S::zero()
        )
    }

    pub fn from_scale_z(sz: S) -> Transform4<S> {
        Transform4::new_3x4(
            S::one(), S::zero(), S::zero(),
            S::zero(), S::one(), S::zero(),
            S::zero(), S::zero(), sz,
            S::zero(), S::zero(), S::zero()
        )
    }

    pub fn from_scale(s: S) -> Transform4<S> {
        Transform4::new_3x4(
            s, S::zero(), S::zero(),
            S::zero(), s, S::zero(),
            S::zero(), S::zero(), s,
            S::zero(), S::zero(), S::zero()
        )
    }

    pub fn from_scale_xyz(sx: S, sy: S, sz: S) -> Transform4<S> {
        Transform4::new_3x4(
            sx, S::zero(), S::zero(),
            S::zero(), sy, S::zero(),
            S::zero(), S::zero(), sz,
            S::zero(), S::zero(), S::zero()
        )
    }




    pub fn from_translation(dv: &Vector3<S>) -> Transform4<S> {
        Transform4::new_3x4(
            S::one(), S::zero(), S::zero(),
            S::zero(), S::one(), S::zero(),
            S::zero(), S::zero(), S::one(),
            dv.x, dv.y, dv.z
        )
    }

    pub fn from_rotation_axis(r: Rad<S>, axis: &Bivector3<S>) -> Transform4<S> {
        let (s, c) = Rad::sin_cos(r);
        let d = S::one() - c;

        let x = axis.x * d;
        let y = axis.y * d;
        let z = axis.z * d;
        let axay = x * axis.y;
        let axaz = x * axis.z;
        let ayaz = y * axis.z;
        
        Transform4::new_3x4(
            c + x * axis.x, axay + s * axis.z, axaz - s * axis.y,
            axay - s * axis.z, c + y * axis.y, ayaz + s * axis.x,
            axaz + s * axis.y, ayaz - s * axis.x, c + z * axis.z,
            S::zero(), S::zero(), S::zero()
        )
    }

    pub fn from_reflection_v3(a: &Vector3<S>) -> Transform4<S> {
        let neg_two = cast(-2).unwrap();
        let x = a.x * neg_two;
        let y = a.y * neg_two;
        let z = a.z * neg_two;
        let axay = x * a.y;
        let axaz = x * a.z;
        let ayaz = y * a.z;

        Transform4::new_3x4(
            x * a.x + S::one(), axay, axaz,
            axay, y * a.y + S::one(), ayaz,
            axaz, ayaz, z * a.z + S::one(),
            S::zero(), S::zero(), S::zero()
        )
    }

    pub fn from_involution(a: &Vector3<S>) -> Transform4<S> {
        let two = cast(2).unwrap();
        let x = a.x * two;
        let y = a.y * two;
        let z = a.z * two;
        let axay = x * a.y;
        let axaz = x * a.z;
        let ayaz = y * a.z;

        Transform4::new_3x4(
            x * a.x - S::one(), axay, axaz,
            axay, y * a.y - S::one(), ayaz,
            axaz, ayaz, z * a.z - S::one(),
            S::zero(), S::zero(), S::zero()
        )
    }

    pub fn from_reflection_plane(plane: &Trivector4<S>) -> Transform4<S> {
        let neg_two = cast(-2).unwrap();
        let x = plane.x * neg_two;
        let y = plane.y * neg_two;
        let z = plane.z * neg_two;
        let nxny = x * plane.y;
        let nxnz = x * plane.z;
        let nynz = y * plane.z;
        Transform4::new_3x4(
            x * plane.x + S::one(), nxny, nxnz,
            nxny, y * plane.y + S::one(), nynz,
            nxnz, nynz, z * plane.z + S::one(),
            x * plane.w, y * plane.w, z * plane.w
        )
    }

    pub fn from_scale_vec3(scale: S, a: &Vector3<S>) -> Transform4<S> {
        let s  = scale - S::one();
        let x = a.x * s;
        let y = a.y * s;
        let z = a.z * s;
        let axay = x * a.y;
        let axaz = x * a.z;
        let ayaz = y * a.z;
        Transform4::new_3x4(
            x * a.x + S::one(), axay, axaz,
            axay, y * a.y + S::one(), ayaz,
            axaz, ayaz, z * a.z + S::one(),
            S::zero(), S::zero(), S::zero()
        )
    }

    pub fn from_skew(r: Rad<S>, a: &Vector3<S>, b: &Vector3<S>) -> Transform4<S> {
        let t = Rad::tan(r);
        let x = a.x * t;
        let y = a.y * t;
        let z = a.z * t;
        Transform4::new_3x4(
            x * b.x + S::one(), y * b.x, z * b.z,
            x * b.y, y * b.y + S::one(), z * b.y,
            x * b.z, y * b.z, z * b.z + S::one(),
            S::zero(), S::zero(), S::zero()
        )
    }

    pub fn orthogonalize(&mut self, column: i32) -> &Transform4<S> {
        self.matrix.x.normalize();
        self.matrix.y = (self.matrix.y - self.matrix.x * self.matrix.x.dot(self.matrix.y)).normalize();
        self.matrix.z = (self.matrix.z - self.matrix.x * self.matrix.x.dot(self.matrix.z)).normalize();
        self
    }

    pub fn determinant(&self) -> S {
        self[0][0] * (self[1][1] * self[2][2] - self[1][2] * self[2][1]) - self[0][1] * (self[1][0] * self[2][2] - self[1][2] * self[2][0]) + self[0][2] * (self[1][0] * self[2][1] - self[1][1] * self[2][0])
    }

    // todo simd
    pub fn inverse(&self) -> Transform4<S> {
        let r0 = self.matrix.row_v3(0);
        let r1 = self.matrix.row_v3(1);
        let r2 = self.matrix.row_v3(2);
        let r3 = self.matrix.row_v3(3);

        let mut s: Bivector3<S> = r0 ^ r1;
        let mut t: Bivector3<S> = r2 ^ r3;
        
        let inv_det: S = S::one() / (s ^ r2);

        s *= inv_det;
        t *= inv_det;

        let v: Vector3<S> = r2 * inv_det;

        let bv1 = r1 ^ v;
        let s1 = -(r1 ^ t);
        let bv2 = v ^ r0;
        let s2 = r0 ^ t;
        let s3 = -(r3 ^ s);

        Transform4::from_bivs(&bv1, s1, &bv2, s2, &s, s3)
    }

    // todo simd
    pub fn adjugate(&self) -> Transform4<S> {
        let r0 = self.matrix.row_v3(0);
        let r1 = self.matrix.row_v3(1);
        let r2 = self.matrix.row_v3(2);
        let r3 = self.matrix.row_v3(3);

        let s: Bivector3<S> = r0 ^ r1;
        let t: Bivector3<S> = r2 ^ r3;

        let bv1 = r1 ^ r2;
        let s1 = -(r1 ^ t);
        let bv2 = r2 ^ r0;
        let s2 = r0 ^ t;
        let s3 = -(r3 ^ s);

        Transform4::from_bivs(&bv1, s1, &bv2, s2, &s, s3)
    }

    pub fn adjugate3d(&self) -> Matrix3<S> {
        let r0 = self.matrix.row_v3(0);
        let r1 = self.matrix.row_v3(1);
        let r2 = self.matrix.row_v3(2);

        let g0: Bivector3<S> = r1 ^ r2;
        let g1: Bivector3<S> = r2 ^ r0;
        let g2: Bivector3<S> = r0 ^ r1;

        Matrix3::new(
            g0.x, g0.y, g0.z, g1.x, g1.y, g1.z, g2.x, g2.y, g2.z
        )
    }


}

impl<S> Index<usize> for Transform4<S> {
    type Output = Vector4<S>;
    #[inline]
    fn index<'a>(&'a self, i: usize) -> &'a Vector4<S> {
        From::from(&self.matrix[i])
    }
}

impl<S> IndexMut<usize> for Transform4<S> {
    #[inline]
    fn index_mut<'a>(&'a mut self, i: usize) -> &'a mut Vector4<S> {
        From::from(&mut self.matrix[i])
    }
}

impl_assignment_operator!(<S: BaseFloat> MulAssign<Transform4<S>> for Transform4<S> {
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

impl_assignment_operator!(<S: BaseFloat> MulAssign<Matrix3<S>> for Transform4<S> {
    fn mul_assign(&mut self, m) {
        let x = self.matrix.x.x;
        let y = self.matrix.y.x;
        let z = self.matrix.z.x;

        self.matrix.x.x = x * self.matrix.x.x + y * self.matrix.x.y + z * self.matrix.x.z;
        self.matrix.y.x = x * self.matrix.y.x + y * self.matrix.y.y + z * self.matrix.y.z;
        self.matrix.z.x = x * self.matrix.z.x + y * self.matrix.z.y + z * self.matrix.z.z;
        
        let x = self.matrix.x.y;
        let y = self.matrix.y.y;
        let z = self.matrix.z.y;

        self.matrix.x.y = x * self.matrix.x.x + y * self.matrix.x.y + z * self.matrix.x.z;
        self.matrix.y.y = x * self.matrix.y.x + y * self.matrix.y.y + z * self.matrix.y.z;
        self.matrix.z.y = x * self.matrix.z.x + y * self.matrix.z.y + z * self.matrix.z.z;

        let x = self.matrix.x.z;
        let y = self.matrix.y.z;
        let z = self.matrix.z.z;

        self.matrix.x.z = x * self.matrix.x.x + y * self.matrix.x.y + z * self.matrix.x.z;
        self.matrix.y.z = x * self.matrix.y.x + y * self.matrix.y.y + z * self.matrix.y.z;
        self.matrix.z.z = x * self.matrix.z.x + y * self.matrix.z.y + z * self.matrix.z.z;
    }
});

//#[cfg(not(feature = "simd"))]
// todo simd
impl_operator!(<S: BaseFloat> Mul<Transform4<S>> for Transform4<S> {
    fn mul(m1, m2) -> Transform4<S> {
        Transform4::new_3x4(
            m1[0][0] * m2[0][0] + m1[0][1] * m2[1][0] + m1[0][2] * m2[2][0],
            m1[1][0] * m2[0][0] + m1[1][1] * m2[1][0] + m1[1][2] * m2[2][0],
            m1[2][0] * m2[0][0] + m1[2][1] * m2[1][0] + m1[2][2] * m2[2][0],
            
            m1[0][0] * m2[0][1] + m1[0][1] * m2[1][1] + m1[0][2] * m2[2][1],
            m1[1][0] * m2[0][1] + m1[1][1] * m2[1][1] + m1[1][2] * m2[2][1],
            m1[2][0] * m2[0][1] + m1[2][1] * m2[1][1] + m1[2][2] * m2[2][1],
            
            
            m1[0][0] * m2[0][2] + m1[0][1] * m2[1][2] + m1[0][2] * m2[2][2],
            m1[1][0] * m2[0][2] + m1[1][1] * m2[1][2] + m1[1][2] * m2[2][2],
            m1[2][0] * m2[0][2] + m1[2][1] * m2[1][2] + m1[2][2] * m2[2][2],

            m1[0][0] * m2[0][3] + m1[0][1] * m2[1][3] + m1[0][2] * m2[2][3] + m1[0][3],
            m1[1][0] * m2[0][3] + m1[1][1] * m2[1][3] + m1[1][2] * m2[2][3] + m1[1][3],
            m1[2][0] * m2[0][3] + m1[2][1] * m2[1][3] + m1[2][2] * m2[2][3] + m1[2][3])
    }
});

// todo simd
impl_operator!(<S: BaseFloat> Mul<Transform4<S>> for Matrix4<S> {
    fn mul(m1, m2) -> Matrix4<S> {
        Matrix4::new(
            m1[0][0] * m2[0][0] + m1[0][1] * m2[1][0] + m1[0][2] * m2[2][0],
            m1[1][0] * m2[0][0] + m1[1][1] * m2[1][0] + m1[1][2] * m2[2][0],
            m1[2][0] * m2[0][0] + m1[2][1] * m2[1][0] + m1[2][2] * m2[2][0],
            m1[3][0] * m2[0][0] + m1[3][1] * m2[1][0] + m1[3][2] * m2[2][0],
            
            m1[0][0] * m2[0][1] + m1[0][1] * m2[1][1] + m1[0][2] * m2[2][1],
            m1[1][0] * m2[0][1] + m1[1][1] * m2[1][1] + m1[1][2] * m2[2][1],
            m1[2][0] * m2[0][1] + m1[2][1] * m2[1][1] + m1[2][2] * m2[2][1],
            m1[3][0] * m2[0][1] + m1[3][1] * m2[1][1] + m1[3][2] * m2[2][1],
            
            m1[0][0] * m2[0][2] + m1[0][1] * m2[1][2] + m1[0][2] * m2[2][2],
            m1[1][0] * m2[0][2] + m1[1][1] * m2[1][2] + m1[1][2] * m2[2][2],
            m1[2][0] * m2[0][2] + m1[2][1] * m2[1][2] + m1[2][2] * m2[2][2],
            m1[3][0] * m2[0][2] + m1[3][1] * m2[1][2] + m1[3][2] * m2[2][2],

            m1[0][0] * m2[0][3] + m1[0][1] * m2[1][3] + m1[0][2] * m2[2][3] + m1[0][3],
            m1[1][0] * m2[0][3] + m1[1][1] * m2[1][3] + m1[1][2] * m2[2][3] + m1[1][3],
            m1[2][0] * m2[0][3] + m1[2][1] * m2[1][3] + m1[2][2] * m2[2][3] + m1[2][3],
            m1[3][0] * m2[0][3] + m1[3][1] * m2[1][3] + m1[3][2] * m2[2][3] + m1[2][3]
        )
    }
});

// todo simd
impl_operator!(<S: BaseFloat> Mul<Matrix4<S>> for Transform4<S> {
    fn mul(m1, m2) -> Matrix4<S> {
        Matrix4::new(
            m1[0][0] * m2[0][0] + m1[0][1] * m2[1][0] + m1[0][2] * m2[2][0] + m1[0][3] * m2[3][0],
            m1[0][0] * m2[0][1] + m1[0][1] * m2[1][1] + m1[0][2] * m2[2][1] + m1[0][3] * m2[3][1],
            m1[0][0] * m2[0][2] + m1[0][1] * m2[1][2] + m1[0][2] * m2[2][2] + m1[0][3] * m2[3][2],
            m1[0][0] * m2[0][3] + m1[0][1] * m2[1][3] + m1[0][2] * m2[2][3] + m1[0][3] * m2[3][3],
            
            m1[1][0] * m2[0][0] + m1[1][1] * m2[1][0] + m1[1][2] * m2[2][0] + m1[1][3] * m2[3][0],
            m1[1][0] * m2[0][1] + m1[1][1] * m2[1][1] + m1[1][2] * m2[2][1] + m1[1][3] * m2[3][1],
            m1[1][0] * m2[0][2] + m1[1][1] * m2[1][2] + m1[1][2] * m2[2][2] + m1[1][3] * m2[3][2],
            m1[1][0] * m2[0][3] + m1[1][1] * m2[1][3] + m1[1][2] * m2[2][3] + m1[1][3] * m2[3][3],
            
            m1[2][0] * m2[0][0] + m1[2][1] * m2[1][0] + m1[2][2] * m2[2][0] + m1[2][3] * m2[3][0],
            m1[2][0] * m2[0][1] + m1[2][1] * m2[1][1] + m1[2][2] * m2[2][1] + m1[2][3] * m2[3][1],
            m1[2][0] * m2[0][2] + m1[2][1] * m2[1][2] + m1[2][2] * m2[2][2] + m1[2][3] * m2[3][2],
            m1[2][0] * m2[0][3] + m1[2][1] * m2[1][3] + m1[2][2] * m2[2][3] + m1[2][3] * m2[3][3],
            
            m2[3][0], m2[3][1], m2[3][2], m2[3][3])
    }
});

// todo simd
impl_operator!(<S:BaseFloat> Mul<Vector3<S>> for Transform4<S>{
    fn mul(m, v) -> Vector3<S> {
        m * v
    }
});

impl_operator!(<S: BaseFloat> Mul<Vector2<S>> for Transform4<S> {
    fn mul(m, v) -> Vector2<S> {
        Vector2::new(
            m[0][0] * v.x + m[1][0] * v.y,
            m[0][1] * v.x + m[1][1] * v.y
        )
    }
});

impl_operator!(<S: BaseFloat> Mul<Point2<S>> for Transform4<S> {
    fn mul(m, v) -> Point2<S> {
        Point2::new(
            m[0][0] * v.x + m[1][0] * v.y + m[3][0],
            m[0][1] * v.x + m[1][1] * v.y + m[3][1]
        )
    }
});
 
pub fn Scale_v2<S>(m: &Transform4<S>, v: &Vector3<S>) -> Transform4<S>
where 
    S: BaseFloat
{
    Transform4::new_3x4(
        m[0][0] * v.x, m[0][1] * v.y, m[0][2] * v.z,
        m[1][0] * v.x, m[1][1] * v.y, m[1][2] * v.z,
        m[2][0] * v.x, m[2][1] * v.y, m[2][2] * v.z,
        m[3][0] * v.x, m[3][1] * v.y, m[3][2] * v.z
    )
}

pub fn Transform_m3<S>(m1: &Transform4<S>, m2: &Matrix3<S>) -> Matrix3<S> 
where 
    S: BaseFloat
{
    Matrix3::new(
        m1[0][0] * m2[0][0] + m1[1][0] * m2[0][1] + m1[2][0] * m2[0][2],
        m1[0][1] * m2[0][0] + m1[1][1] * m2[0][1] + m1[2][1] * m2[0][2],
        m1[0][2] * m2[0][0] + m1[1][2] * m2[0][1] + m1[2][2] * m2[0][2],

        m1[0][0] * m2[1][0] + m1[1][0] * m2[1][1] + m1[2][0] * m2[1][2],
        m1[0][1] * m2[1][0] + m1[1][1] * m2[1][1] + m1[2][1] * m2[1][2],
        m1[0][2] * m2[1][0] + m1[1][2] * m2[1][1] + m1[2][2] * m2[1][2],

        m1[0][0] * m2[2][0] + m1[1][0] * m2[2][1] + m1[2][0] * m2[2][2],
        m1[0][1] * m2[2][0] + m1[1][1] * m2[2][1] + m1[2][1] * m2[2][2],
        m1[0][2] * m2[2][0] + m1[1][2] * m2[2][1] + m1[2][2] * m2[2][2],
    )
}

pub fn InverseTransformVec3<S>(m: &Transform4<S>, v: &Vector3<S>) -> Vector3<S>
where 
    S: BaseFloat
{
    let r0 = m.matrix.row_v3(0);
    let r1 = m.matrix.row_v3(1);
    let r2 = m.matrix.row_v3(2);

    let s: Bivector3<S> = r0 ^ r1;

    let inv_det: S = S::one() / (s ^ r2);

    Vector3::new(
        (r1 ^ r2 ^ v) * inv_det,
        (r2 ^ r1 ^ v) * inv_det,
        (s ^ v) * inv_det
    )
}

pub fn AdjugateTransformVec3<S>(m: &Transform4<S>, v: &Vector3<S>) -> Vector3<S>
where 
    S: BaseFloat
{
    let r0 = m.matrix.row_v3(0);
    let r1 = m.matrix.row_v3(1);
    let r2 = m.matrix.row_v3(2);

    Vector3::new(
        r1 ^ r2 ^ v,
        r2 ^ r0 ^ v,
        r0 ^ r1 ^ v
    )
}

pub fn AdjugateTransformPoint3<S>(m: &Transform4<S>, p: &Point3<S>) -> Point3<S>
where 
    S: BaseFloat
{
    let r0 = m.matrix.row_v3(0);
    let r1 = m.matrix.row_v3(1);
    let r2 = m.matrix.row_v3(2);
    let r3 = m.matrix.row_v3(3);

    let q = p - r3;

    Point3::new(
        r1 ^ r2 ^ q,
        r2 ^ r0 ^ q,
        r0 ^ r1 ^ q
    )
}

impl<A> From<Euler<A>> for Transform4<A::Unitless>
where
    A: Angle + Into<Rad<<A as Angle>::Unitless>>,
{
    fn from(src: Euler<A>) -> Transform4<A::Unitless> {
        // Page A-2: http://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19770024290.pdf
        let (sx, cx) = Rad::sin_cos(src.x.into());
        let (sy, cy) = Rad::sin_cos(src.y.into());
        let (sz, cz) = Rad::sin_cos(src.z.into());

        #[cfg_attr(rustfmt, rustfmt_skip)]
        Transform4 {
            matrix: Matrix4::new(
                cy * cz, cx * sz + sx * sy * cz, sx * sz - cx * sy * cz, A::Unitless::zero(),
                -cy * sz, cx * cz - sx * sy * sz, sx * cz + cx * sy * sz, A::Unitless::zero(),
                sy, -sx * cy, cx * cy, A::Unitless::zero(),
                A::Unitless::zero(), A::Unitless::zero(), A::Unitless::zero(), A::Unitless::one(),
            )
        }
    }
}


