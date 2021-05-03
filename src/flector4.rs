
use num_traits::{cast, NumCast, Float};
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
use trivector4::{Trivector4};
use matrix::{Matrix3, Matrix4};
use transform::Transform4;
use quaternion::Quaternion;
use motor4::Motor4;

/// The $motor$ class encapsulates a 4D reflection operator (flector)
/// form.
///
/// This type is marked as `#[repr(C)]`.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Flector4<S> {

    pub point: Vector4<S>,

    pub plane: Trivector4<S>
}

impl<S> Flector4<S> 
{
    #[inline]
    pub const fn new(rx: S, ry: S, rz: S, rw: S, hx: S, hy: S, hz: S, hw: S) -> Flector4<S> {
        Flector4 {
            point: Vector4::new(rw, rx, ry, rz),
            plane: Trivector4::new(hw, hx, hy, hz)
        }
    }

}

pub fn flector4<S>() -> Flector4<S> where S: NumCast{
    Flector4::new(cast(0).unwrap(), cast(0).unwrap(), cast(0).unwrap(),cast(0).unwrap(),cast(0).unwrap(),cast(0).unwrap(),cast(0).unwrap(),cast(0).unwrap())
}

impl<S: BaseFloat> Flector4<S> {
    #[inline]
    pub fn from_v_triv(v: &Vector4<S>, f: &Trivector4<S>) -> Flector4<S> {
        Flector4 {
            point: Vector4::new(v.x, v.y, v.z, v.w),
            plane: Trivector4::new(f.x, f.y, f.z, f.w)
        }
    }

    #[inline]
    pub fn from_p_triv(p: &Point3<S>, f: &Trivector4<S>) -> Flector4<S> {
        Flector4 {
            point: Vector4::new(p.x, p.y, p.z, S::one()),
            plane: Trivector4::new(f.x, f.y, f.z, f.w)
        }
    }

    #[inline]
    pub fn from_v(v: &Vector4<S>) -> Flector4<S> {
        Flector4 {
            point: Vector4::new(v.x, v.y, v.z, v.w),
            plane: Trivector4::new(S::zero(), S::zero(), S::zero(), S::zero())
        }
    }

    #[inline]
    pub fn from_p(p: &Point3<S>) -> Flector4<S> {
        Flector4 {
            point: Vector4::new(p.x, p.y, p.z, S::one()),
            plane: Trivector4::new(S::zero(), S::zero(), S::zero(), S::zero())
        }
    }

    #[inline]
    pub fn from_triv(f: &Trivector4<S>) -> Flector4<S> {
        Flector4 {
            point: Vector4::zero(),
            plane: Trivector4::new(f.x, f.y, f.z, f.w)
        }
    }


    
}

impl<S: BaseFloat> Flector4<S> {
    pub fn untize<'a>(&'a mut self) -> &'a Flector4<S> {
        let mag = self.point.w * self.point.w + self.plane.x * self.plane.x + self.plane.y * self.plane.y + self.plane.z * self.plane.z;
        let inv_mag = Float::sqrt(mag);
        *self *= inv_mag;
        self
    }


}

impl_assignment_operator!(<S: BaseFloat> MulAssign<S> for Flector4<S> {
    fn mul_assign(&mut self, s) {
        self.point *= s;
        self.plane *= s;
    }
});

impl_assignment_operator!(<S: BaseFloat> DivAssign<S> for Flector4<S> {
    fn div_assign(&mut self, s) {
        self.point /= s;
        self.plane /= s;
    }
});

impl<S: BaseFloat> Flector4<S> {
    #[inline]
    pub fn make_trans_flection(offset: &Vector3<S>, f: & Trivector4<S>) -> Flector4<S> {
        let half: S  = cast(0.5f32).unwrap();
        Flector4::new(
            (offset.y * f.z - offset.z * f.y) * half, 
            (offset.z * f.x - offset.x * f.z) * half, 
            (offset.x * f.y - offset.y * f.x) * half, 
            S::zero(), 
            f.x, 
            f.y, 
            f.z, 
            f.w - (offset.x * f.x + offset.y * f.y + offset.z * f.z) * half
        )
    }

    #[inline]
    pub fn make_rotore_flection(r: Rad<S>, axis: &Bivector3<S>, f: &Trivector4<S>) -> Flector4<S> {
        let half: S  = cast(0.5f32).unwrap();
        let (s, c) = Rad::sin_cos(r * half);
        let rx = axis.x * s;
        let ry = axis.y * s;
        let rz = axis.z * s;
        Flector4::new(
            rx * f.w, ry * f.w, rz * f.w, 
            -rx * f.x - ry * f.y - rz * f.z, c * f.x + ry * f.z - rz * f.y, 
            c * f.y + rz * f.x - rx * f.z, c * f.z + rx * f.y - ry * f.x,
            c * f.w
        )
    }

    #[inline]
    pub fn make_rotore_flection_biv4(r: Rad<S>, axis: &Bivector4<S>, f: &Trivector4<S>) -> Flector4<S> {
        let half: S  = cast(0.5f32).unwrap();
        let (s, c) = Rad::sin_cos(r * half);
        let rx = axis.direction.x * s;
        let ry = axis.direction.y * s;
        let rz = axis.direction.z * s;
        let ux = axis.moment.x * s;
        let uy = axis.moment.y * s;
        let uz = axis.moment.z * s;

        Flector4::new(
            rx * f.w + uy * f.z - uz * f.y, ry * f.w + uz * f.x - ux * f.z, 
            rz * f.w + ux * f.y - uy * f.x, -rx * f.x - ry * f.y - rz * f.z, 
            c * f.x + ry * f.z - rz * f.y, c * f.y + rz * f.x - rx * f.z, 
            c * f.z + rx * f.y - ry * f.x, c * f.w - ux * f.x - uy * f.y - uz * f.z
        )
    }

    pub fn get_transform_matrix(&self) -> Transform4<S> {
        let two: S  = cast(2.0f32).unwrap();
        let sx = self.point.x;
        let sy = self.point.y;
        let sz = self.point.z;
        let sw = self.point.w;
        let hx = self.plane.x;
        let hy = self.plane.y;
        let hz = self.plane.z;
        let hw = self.plane.w;
        let hx2 = hx * hx;
        let hy2 = hy * hy;
        let hz2 = hz * hz;
        
        let A00 = (hy2 + hz2) * two - S::one();
        let A11 = (hz2 + hx2) * two - S::one();
        let A22 = (hx2 + hy2) * two - S::one();
        let A01 = hx * hy * -two;
        let A02 = hz * hx * -two;
        let A12 = hy * hz * -two;
        let A03 = (sx * sw - hx * hw) * two;
        let A13 = (sy * sw - hy * hw) * two;
        let A23 = (sz * sw - hz * hw) * two;
        
        let B01 = hz * sw * two;
        let B20 = hy * sw * two;
        let B12 = hx * sw * two;
        let B03 = (hy * sz - hz * sy) * two;
        let B13 = (hz * sx - hx * sz) * two;
        let B23 = (hx * sy - hy * sx) * two;
        
        Transform4::new_3x4(   
                A00,    A01 + B01, A02 - B20, A03 + B03,
                A01 - B01,    A11,    A12 + B12, A13 + B13,
                A02 + B20, A12 - B12,    A22,    A23 + B23)
    }

    pub fn get_inverse_transform_matrix(&self) -> Transform4<S> {
        let two: S  = cast(2.0f32).unwrap();
        let sx = self.point.x;
        let sy = self.point.y;
        let sz = self.point.z;
        let sw = self.point.w;
        let hx = self.plane.x;
        let hy = self.plane.y;
        let hz = self.plane.z;
        let hw = self.plane.w;
        let hx2 = hx * hx;
        let hy2 = hy * hy;
        let hz2 = hz * hz;

        let A00 = (hy2 + hz2) * two - S::one();
        let A11 = (hz2 + hx2) * two - S::one();
        let A22 = (hx2 + hy2) * two - S::one();
        let A01 = hx * hy * -two;
        let A02 = hz * hx * -two;
        let A12 = hy * hz * -two;
        let A03 = (sx * sw - hx * hw) * two;
        let A13 = (sy * sw - hy * hw) * two;
        let A23 = (sz * sw - hz * hw) * two;

        let B01 = hz * sw * two;
        let B20 = hy * sw * two;
        let B12 = hx * sw * two;
        let B03 = (hy * sz - hz * sy) * two;
        let B13 = (hz * sx - hx * sz) * two;
        let B23 = (hx * sy - hy * sx) * two;

        Transform4::new_3x4(  
            A00,    A01 - B01, A02 + B20, A03 - B03,
            A01 + B01,    A11,    A12 - B12, A13 - B13,
            A02 - B20, A12 + B12,    A22,    A23 - B23)
    }

    pub fn get_transform_matrices(&self) -> (Transform4<S>, Transform4<S>) {
        let two: S  = cast(2.0f32).unwrap();
        let sx = self.point.x;
        let sy = self.point.y;
        let sz = self.point.z;
        let sw = self.point.w;
        let hx = self.plane.x;
        let hy = self.plane.y;
        let hz = self.plane.z;
        let hw = self.plane.w;
        let hx2 = hx * hx;
        let hy2 = hy * hy;
        let hz2 = hz * hz;
        
        let A00 = (hy2 + hz2) * two - S::one();
        let A11 = (hz2 + hx2) * two - S::one();
        let A22 = (hx2 + hy2) * two - S::one();
        let A01 = hx * hy * -two;
        let A02 = hz * hx * -two;
        let A12 = hy * hz * -two;
        let A03 = (sx * sw - hx * hw) * two;
        let A13 = (sy * sw - hy * hw) * two;
        let A23 = (sz * sw - hz * hw) * two;
        
        let B01 = hz * sw * two;
        let B20 = hy * sw * two;
        let B12 = hx * sw * two;
        let B03 = (hy * sz - hz * sy) * two;
        let B13 = (hz * sx - hx * sz) * two;
        let B23 = (hx * sy - hy * sx) * two;
        
        let m = Transform4::new_3x4(
            A00,    A01 + B01, A02 - B20, A03 + B03,
            A01 - B01,    A11,    A12 + B12, A13 + B13,
            A02 + B20, A12 - B12,    A22,    A23 + B23
        );
        let minv = Transform4::new_3x4(
            A00,    A01 - B01, A02 + B20, A03 - B03,
                  A01 + B01,    A11,    A12 - B12, A13 - B13,
                  A02 - B20, A12 + B12,    A22,    A23 - B23
        );
        
        (m, minv)
    }

    pub fn set_transform_matrix<'a>(&'a mut self, m: &Transform4<S>) -> &'a Flector4<S> {
        let neg_quarter: S = cast(-0.25f32).unwrap();
        let half = cast(0.5f32).unwrap();
        let two: S  = cast(2.0f32).unwrap();
        let m00 = m.matrix.row_col(0,0);
        let m11 = m.matrix.row_col(1,1);
        let m22 = m.matrix.row_col(2,2);
        let sum = m00 + m11 + m22;

        if (sum < S::zero())
        {
            self.point.w = Float::sqrt(S::one() - sum) * half;
            let f = neg_quarter / self.point.w;

            self.plane.x = (m.matrix.row_col(2,1) - m.matrix.row_col(1,2)) * f;
            self.plane.y = (m.matrix.row_col(0,2) - m.matrix.row_col(2,0)) * f;
            self.plane.z = (m.matrix.row_col(1,0) - m.matrix.row_col(0,1)) * f;
        }
        else if ((m00 < m11) && (m00 < m22))
        {
            self.plane.x = Float::sqrt(S::one() - m00 + m11 + m22) * half;
            let f = neg_quarter / self.plane.x;

            self.plane.y = (m.matrix.row_col(1,0) + m.matrix.row_col(0,1)) * f;
            self.plane.z = (m.matrix.row_col(0,2) + m.matrix.row_col(2,0)) * f;
            self.point.w = (m.matrix.row_col(2,1) - m.matrix.row_col(1,2)) * f;
        }
        else if (m11 < m22)
        {
            self.plane.y = Float::sqrt(S::one() - m11 + m22 + m00) * half;
            let f = neg_quarter / self.plane.y;

            self.plane.x = (m.matrix.row_col(1,0) + m.matrix.row_col(0,1)) * f;
            self.plane.z = (m.matrix.row_col(2,1) + m.matrix.row_col(1,2)) * f;
            self.point.w = (m.matrix.row_col(0,2) - m.matrix.row_col(2,0)) * f;
        }
        else
        {
            self.plane.z = Float::sqrt(S::one() - m22 + m00 + m11) * half;
            let f = neg_quarter / self.plane.z;

            self.plane.x = (m.matrix.row_col(0,2) + m.matrix.row_col(2,0)) * f;
            self.plane.y = (m.matrix.row_col(2,1) + m.matrix.row_col(1,2)) * f;
            self.point.w = (m.matrix.row_col(1,0) - m.matrix.row_col(0,1)) * f;
        }

        let tx = m.matrix.row_col(0,3) * half;
        let ty = m.matrix.row_col(1,3) * half;
        let tz = m.matrix.row_col(2,3) * half;

        let hx = self.plane.x;
        let hy = self.plane.y;
        let hz = self.plane.z;
        let sw = self.point.w;

        self.point.x =  sw * tx + hz * ty - hy * tz;
        self.point.y =  sw * ty + hx * tz - hz * tx;
        self.point.z =  sw * tz + hy * tx - hx * ty;
        self.plane.w = -hx * tx - hy * ty - hz * tz;

        self
    }



}

impl_operator!(<S: BaseFloat> Not for Flector4<S> {
    fn not(f) -> Flector4<S> {
        Flector4::new(
            -f.point.x, -f.point.y, -f.point.z, -f.point.w,
            -f.plane.x, -f.plane.y, -f.plane.z, f.plane.w
        )
    }
});

impl_operator!(<S: BaseFloat> Neg for Flector4<S> {
    fn neg(f) -> Flector4<S> {
        Flector4::new(
            -f.point.x, -f.point.y, -f.point.z, -f.point.w,
            -f.plane.x, -f.plane.y, -f.plane.z, f.plane.w
        )
    }
});

impl_operator!(<S: BaseFloat> Mul<S> for Flector4<S> {
    fn mul(f, s) -> Flector4<S> {
        Flector4::new(
            f.point.x * s, f.point.y * s, f.point.z * s, f.point.w * s,
            f.plane.x * s, f.plane.y * s, f.plane.z * s, f.plane.w * s
        )
    }
});

impl_operator!(<S: BaseFloat> Div<S> for Flector4<S> {
    fn div(f, s) -> Flector4<S> {
        Flector4::new(
            f.point.x / s, f.point.y / s, f.point.z / s, f.point.w / s,
            f.plane.x / s, f.plane.y / s, f.plane.z / s, f.plane.w / s
        )
    }
});


impl<S: NumCast + Copy> Flector4<S> {
    /// Component-wise casting to another type.
    #[inline]
    pub fn cast<T: NumCast + BaseFloat>(&self) -> Option<Flector4<T>> {
        let point: Vector4<T> = match self.point.cast() {
            Some(point) => point,
            None => return None
        };
        let plane: Trivector4<T>  = match self.plane.cast() {
            Some(plane) => plane,
            None => return None
        };
        Some(Flector4 { point: point, plane: plane})
    }
}

impl<S: BaseFloat> approx::AbsDiffEq for Flector4<S> {
    type Epsilon = S::Epsilon;

    #[inline]
    fn default_epsilon() -> S::Epsilon {
        S::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: S::Epsilon) -> bool {
        self.point.abs_diff_eq(&other.point, epsilon) &&
        self.plane.abs_diff_eq(&other.plane, epsilon)
    }
}

impl<S: BaseFloat> approx::RelativeEq for Flector4<S> {
    #[inline]
    fn default_max_relative() -> S::Epsilon {
        S::default_max_relative()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: S::Epsilon, max_relative: S::Epsilon) -> bool {
        self.point.relative_eq(&other.point, epsilon, max_relative) &&
        self.plane.relative_eq(&other.plane, epsilon, max_relative)
    }
}

impl<S: BaseFloat> approx::UlpsEq for Flector4<S> {
    #[inline]
    fn default_max_ulps() -> u32 {
        S::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: S::Epsilon, max_ulps: u32) -> bool {
        self.point.ulps_eq(&other.point, epsilon, max_ulps) &&
        self.plane.ulps_eq(&other.plane, epsilon, max_ulps)
    }
}

#[cfg(feature = "rand")]
impl<S> Distribution<Flector4<S>> for Standard
    where Standard: Distribution<Quaternion<S>>,
        S: BaseFloat {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Flector4<S> {
        Flector4{
            point: self.sample(rng),
            plane: self.sample(rng)
        }
    }
}

impl_operator!(<S:BaseFloat> Mul<Flector4<S>> for Flector4<S> {
    fn mul(a, b) -> Motor4<S> {
        Motor4::new(
            a.plane.z * b.plane.y - a.plane.y * b.plane.z - a.plane.x * b.point.w - a.point.w * b.plane.x,
            a.plane.x * b.plane.z - a.plane.z * b.plane.x - a.plane.y * b.point.w - a.point.w * b.plane.y,
            a.plane.y * b.plane.x - a.plane.x * b.plane.y - a.plane.z * b.point.w - a.point.w * b.plane.z,
            a.plane.x * b.plane.x + a.plane.y * b.plane.y + a.plane.z * b.plane.z - a.point.w * b.point.w,
            a.point.z * b.plane.y - a.point.y * b.plane.z + a.plane.y * b.point.z - a.plane.z * b.point.y + a.plane.x * b.plane.w - a.plane.w * b.plane.x + a.point.w * b.point.x - a.point.x * b.point.w,
            a.point.x * b.plane.z - a.point.z * b.plane.x + a.plane.z * b.point.x - a.plane.x * b.point.z + a.plane.y * b.plane.w - a.plane.w * b.plane.y + a.point.w * b.point.y - a.point.y * b.point.w,
            a.point.y * b.plane.x - a.point.x * b.plane.y + a.plane.x * b.point.y - a.plane.y * b.point.x + a.plane.z * b.plane.w - a.plane.w * b.plane.z + a.point.w * b.point.z - a.point.z * b.point.w,
            a.point.x * b.plane.x + a.point.y * b.plane.y + a.point.z * b.plane.z + a.point.w * b.plane.w - a.plane.x * b.point.x - a.plane.y * b.point.y - a.plane.z * b.point.z - a.plane.w * b.point.w)
    }
});

impl_operator!(<S: BaseFloat> Mul<Motor4<S>>  for Flector4<S> {
    fn mul(a, b) -> Flector4<S> {
        Flector4::new(
            a.plane.z * b.screw.v.y - a.plane.y * b.screw.v.z + a.plane.w * b.rotor.v.x - a.plane.x * b.screw.s + a.point.y * b.rotor.v.z - a.point.z * b.rotor.v.y + a.point.x * b.rotor.s - a.point.w * b.screw.v.x,
            a.plane.x * b.screw.v.z - a.plane.z * b.screw.v.x + a.plane.w * b.rotor.v.y - a.plane.y * b.screw.s + a.point.z * b.rotor.v.x - a.point.x * b.rotor.v.z + a.point.y * b.rotor.s - a.point.w * b.screw.v.y,
            a.plane.y * b.screw.v.x - a.plane.x * b.screw.v.y + a.plane.w * b.rotor.v.z - a.plane.z * b.screw.s + a.point.x * b.rotor.v.y - a.point.y * b.rotor.v.x + a.point.z * b.rotor.s - a.point.w * b.screw.v.z,
            a.point.w * b.rotor.s - a.plane.x * b.rotor.v.x - a.plane.y * b.rotor.v.y - a.plane.z * b.rotor.v.z,
            a.plane.y * b.rotor.v.z - a.plane.z * b.rotor.v.y + a.point.w * b.rotor.v.x + a.plane.x * b.rotor.s,
            a.plane.z * b.rotor.v.x - a.plane.x * b.rotor.v.z + a.point.w * b.rotor.v.y + a.plane.y * b.rotor.s,
            a.plane.x * b.rotor.v.y - a.plane.y * b.rotor.v.x + a.point.w * b.rotor.v.z + a.plane.z * b.rotor.s,
            a.plane.w * b.rotor.s + a.plane.x * b.screw.v.x + a.plane.y * b.screw.v.y + a.plane.z * b.screw.v.z - a.point.w * b.screw.s - a.point.x * b.rotor.v.x - a.point.y * b.rotor.v.y - a.point.z * b.rotor.v.z)
    }
});

impl_operator!(<S: BaseFloat> Mul<Flector4<S>>  for Motor4<S> {
    fn mul(a, b) -> Flector4<S> {
        Flector4::new(
            b.plane.z * a.screw.v.y - b.plane.y * a.screw.v.z + b.plane.w * a.rotor.v.x + b.plane.x * a.screw.s + b.point.z * a.rotor.v.y - b.point.y * a.rotor.v.z + b.point.x * a.rotor.s + b.point.w * a.screw.v.x,
            b.plane.x * a.screw.v.z - b.plane.z * a.screw.v.x + b.plane.w * a.rotor.v.y + b.plane.y * a.screw.s + b.point.x * a.rotor.v.z - b.point.z * a.rotor.v.x + b.point.y * a.rotor.s + b.point.w * a.screw.v.y,
            b.plane.y * a.screw.v.x - b.plane.x * a.screw.v.y + b.plane.w * a.rotor.v.z + b.plane.z * a.screw.s + b.point.y * a.rotor.v.x - b.point.x * a.rotor.v.y + b.point.z * a.rotor.s + b.point.w * a.screw.v.z,
            b.point.w * a.rotor.s - b.plane.x * a.rotor.v.x - b.plane.y * a.rotor.v.y - b.plane.z * a.rotor.v.z,
            b.plane.z * a.rotor.v.y - b.plane.y * a.rotor.v.z + b.point.w * a.rotor.v.x + b.plane.x * a.rotor.s,
            b.plane.x * a.rotor.v.z - b.plane.z * a.rotor.v.x + b.point.w * a.rotor.v.y + b.plane.y * a.rotor.s,
            b.plane.y * a.rotor.v.x - b.plane.x * a.rotor.v.y + b.point.w * a.rotor.v.z + b.plane.z * a.rotor.s,
            b.plane.w * a.rotor.s - b.plane.x * a.screw.v.x - b.plane.y * a.screw.v.y - b.plane.z * a.screw.v.z + b.point.w * a.screw.s - b.point.x * a.rotor.v.x - b.point.y * a.rotor.v.y - b.point.z * a.rotor.v.z)
    }
});


impl<S: BaseFloat> TransformTrait<Flector4<S>, Vector3<S>> for Vector3<S> {

    fn transform(&self, f: &Flector4<S>) -> Vector3<S> {
        let sw = f.point.w;
        let hx = f.plane.x;
        let hy = f.plane.y;
        let hz = f.plane.z;

        let sw2 = sw * sw;
        let hx2 = hx * hx;
        let hy2 = hy * hy;
        let hz2 = hz * hz;
        let hyhz = hy * hz;
        let hzhx = hz * hx;
        let hxhy = hx * hy;
        let hxsw = hx * sw;
        let hysw = hy * sw;
        let hzsw = hz * sw;

        let two = cast(2.0f32).unwrap();

        Vector3::new(
            self.x + ((hzsw - hxhy) * self.y - (hzhx + hysw) * self.z - (hx2 + sw2) * self.x) * two,
            self.y + ((hxsw - hyhz) * self.z - (hxhy + hzsw) * self.x - (hy2 + sw2) * self.y) * two,
            self.z + ((hysw - hzhx) * self.x - (hyhz + hxsw) * self.y - (hz2 + sw2) * self.z) * two)
    }
}

impl<S: BaseFloat> TransformTrait<Flector4<S>, Bivector3<S>> for Bivector3<S> {

    fn transform(&self, f: &Flector4<S>) -> Bivector3<S> {
        let sw = f.point.w;
        let hx = f.plane.x;
        let hy = f.plane.y;
        let hz = f.plane.z;

        let sw2 = sw * sw;
        let hx2 = hx * hx;
        let hy2 = hy * hy;
        let hz2 = hz * hz;
        let hyhz = hy * hz;
        let hzhx = hz * hx;
        let hxhy = hx * hy;
        let hxsw = hx * sw;
        let hysw = hy * sw;
        let hzsw = hz * sw;
        let two = cast(2.0f32).unwrap();

        Bivector3::new(
            self.x + ((hzsw - hxhy) * self.y - (hzhx + hysw) * self.z - (hx2 + sw2) * self.x) * two,
            self.y + ((hxsw - hyhz) * self.z - (hxhy + hzsw) * self.x - (hy2 + sw2) * self.y) * two,
            self.z + ((hysw - hzhx) * self.x - (hyhz + hxsw) * self.y - (hz2 + sw2) * self.z) * two)
    }
}

impl<S: BaseFloat> TransformTrait<Flector4<S>, Point3<S>> for Point3<S> {

    fn transform(&self, f: &Flector4<S>) -> Point3<S> {
        let sx = f.point.x;
        let sy = f.point.y;
        let sz = f.point.z;
        let sw = f.point.w;
        let hx = f.plane.x;
        let hy = f.plane.y;
        let hz = f.plane.z;
        let hw = f.plane.w;

        let sw2 = sw * sw;
        let hx2 = hx * hx;
        let hy2 = hy * hy;
        let hz2 = hz * hz;
        let hyhz = hy * hz;
        let hzhx = hz * hx;
        let hxhy = hx * hy;
        let hxsw = hx * sw;
        let hysw = hy * sw;
        let hzsw = hz * sw;
        let two = cast(2.0f32).unwrap();

        Point3::new(
            self.x + ((hzsw - hxhy) * self.y - (hzhx + hysw) * self.z - (hx2 + sw2) * self.x + sx * sw - hx * hw + hy * sz - hz * sy) * two,
            self.y + ((hxsw - hyhz) * self.z - (hxhy + hzsw) * self.x - (hy2 + sw2) * self.y + sy * sw - hy * hw + hz * sx - hx * sz) * two,
            self.z + ((hysw - hzhx) * self.x - (hyhz + hxsw) * self.y - (hz2 + sw2) * self.z + sz * sw - hz * hw + hx * sy - hy * sx) * two)
    }
}

impl<S: BaseFloat> TransformTrait<Flector4<S>, Bivector4<S>> for Bivector4<S> {

    fn transform(&self, f: &Flector4<S>) -> Bivector4<S> {
        let vx = self.direction.x;
        let vy = self.direction.y;
        let vz = self.direction.z;
        let mx = self.moment.x;
        let my = self.moment.y;
        let mz = self.moment.z;

        let sx = f.point.x;
        let sy = f.point.y;
        let sz = f.point.z;
        let sw = f.point.w;
        let hx = f.plane.x;
        let hy = f.plane.y;
        let hz = f.plane.z;
        let hw = f.plane.w;

        let sw2 = sw * sw;
        let hx2 = hx * hx;
        let hy2 = hy * hy;
        let hz2 = hz * hz;
        let hyhz = hy * hz;
        let hzhx = hz * hx;
        let hxhy = hx * hy;
        let hxsw = hx * sw;
        let hysw = hy * sw;
        let hzsw = hz * sw;
        let two = cast(2.0f32).unwrap();

        Bivector4::new(vx + ((hxhy - hzsw) * vy + (hzhx + hysw) * vz - (hy2 + hz2) * vx) * two,
        vy + ((hyhz - hxsw) * vz + (hxhy + hzsw) * vx - (hz2 + hx2) * vy) * two,
        vz + ((hzhx - hysw) * vx + (hyhz + hxsw) * vy - (hx2 + hy2) * vz) * two,
        mx + ((hx * sy + hy * sx - sz * sw - hz * hw) * vy + (hx * sz + hz * sx + sy * sw + hy * hw) * vz - (hy * sy + hz * sz) * vx * two + (hzsw - hxhy) * my - (hzhx + hysw) * mz - (hx2 + sw2) * mx) * two,
        my + ((hy * sz + hz * sy - sx * sw - hx * hw) * vz + (hy * sx + hx * sy + sz * sw + hz * hw) * vx - (hz * sz + hx * sx) * vy * two + (hxsw - hyhz) * mz - (hxhy + hzsw) * mx - (hy2 + sw2) * my) * two,
        mz + ((hz * sx + hx * sz - sy * sw - hy * hw) * vx + (hz * sy + hy * sz + sx * sw + hx * hw) * vy - (hx * sx + hy * sy) * vz * two + (hysw - hzhx) * mx - (hyhz + hxsw) * my - (hz2 + sw2) * mz) * two)
    }
}

impl<S: BaseFloat> TransformTrait<Flector4<S>, Trivector4<S>> for Trivector4<S> {
    fn transform(&self, f: &Flector4<S>) -> Trivector4<S> {
        let sx = f.point.x;
        let sy = f.point.y;
        let sz = f.point.z;
        let sw = f.point.w;
        let hx = f.plane.x;
        let hy = f.plane.y;
        let hz = f.plane.z;
        let hw = f.plane.w;

        let sw2 = sw * sw;
        let hx2 = hx * hx;
        let hy2 = hy * hy;
        let hz2 = hz * hz;
        let hyhz = hy * hz;
        let hzhx = hz * hx;
        let hxhy = hx * hy;
        let hxsw = hx * sw;
        let hysw = hy * sw;
        let hzsw = hz * sw;
        let two = cast(2.0f32).unwrap();

        Trivector4::new(self.x + ((hzsw - hxhy) * self.y - (hzhx + hysw) * self.z - (hx2 + sw2) * self.x) * two,
                        self.y + ((hxsw - hyhz) * self.z - (hxhy + hzsw) * self.x - (hy2 + sw2) * self.y) * two,
                        self.z + ((hysw - hzhx) * self.x - (hyhz + hxsw) * self.y - (hz2 + sw2) * self.z) * two,
                        self.w + ((sx * sw - hx * hw - hy * sz + hz * sy) * self.x +
                            (sy * sw - hy * hw - hz * sx + hx * sz) * self.y +
                            (sz * sw - hz * hw - hx * sy + hy * sx) * self.z) * two)
    }
}
