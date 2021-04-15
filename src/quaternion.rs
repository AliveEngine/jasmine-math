use num_traits::{Bounded, Float, NumCast, cast};
#[cfg(feature = "rand")]
use rand::{
    distributions::{Distribution, Standard},
    Rng,
};
use std::fmt;
use std::iter;
use std::mem;
use std::ops::*;

use structure::*;

use angle::{Rad, Deg};
use approx;
use euler::Euler;
use num::{BaseFloat, BaseNum};
use point::{Point2, Point3};
use vector::{Vector2, Vector3, Vector4};
use bivector3::{Bivector3, bivec3};
use trivector4::{Trivector4};
use matrix::{Matrix3,Matrix4};

#[cfg(feature = "mint")]
use mint;

/// A [quaternion](https://en.wikipedia.org/wiki/Quaternion) in scalar/vector
/// form.
///
/// This type is marked as `#[repr(C)]`.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Quaternion<S> {
    /// The vector part of the quaternion.
    pub v: Bivector3<S>,
    /// The scalar part of the quaternion.
    pub s: S,
}

impl<S> Quaternion<S> {
    /// Construct a new quaternion from one scalar component and three
    /// imaginary components.
    #[inline]
    pub const fn new(w: S, xi: S, yj: S, zk: S) -> Quaternion<S> {
        Quaternion::from_sv(w, Bivector3::new(xi, yj, zk))
    }

    /// Construct a new quaternion from a scalar and a vector.
    #[inline]
    pub const fn from_sv(s: S, v: Bivector3<S>) -> Quaternion<S> {
        Quaternion { s: s, v: v }
    }

    #[inline]
    pub fn set(&mut self, w: S, xi: S, yj: S, zk: S){
        self.s = w;
        self.v.x = xi;
        self.v.y = yj;
        self.v.z= zk;
    }

}

impl<S: BaseFloat> Quaternion<S> {

    #[inline]
    pub fn from_rotation_x(angle: Deg<S>) -> Quaternion<S> 
    {
        let half: S = cast(0.5f64).unwrap();
        let (s_x, c_x) = Rad::sin_cos(Rad::from(angle) * half);

        Quaternion::new(
            s_x, S::zero(), S::zero(), c_x
        )
    }

    #[inline]
    pub fn from_rotation_y(angle: Deg<S>) -> Quaternion<S> 
    {
        let half: S = cast(0.5f64).unwrap();
        let (s_x, c_x) = Rad::sin_cos(Rad::from(angle) * half);

        Quaternion::new(
            S::zero(), s_x, S::zero(), c_x
        )
    }

    #[inline]
    pub fn from_rotation_z(angle: Deg<S>) -> Quaternion<S> 
    {
        let half: S = cast(0.5f64).unwrap();
        let (s_x, c_x) = Rad::sin_cos(Rad::from(angle) * half);

        Quaternion::new(
            S::zero(), S::zero(), s_x, c_x
        )
    }

    #[inline]
    pub fn from_rotation_axis(angle: Deg<S>, axis: &Bivector3<S>) -> Quaternion<S> 
    {
        let half: S = cast(0.5f64).unwrap();
        let (s_x, c_x) = Rad::sin_cos(Rad::from(angle) * half);

        Quaternion::from_sv(
            c_x,
            axis * s_x,
        )
    }



    /// The conjugate of the quaternion.
    #[inline]
    pub fn conjugate(self) -> Quaternion<S> {
        Quaternion::from_sv(self.s, -self.v)
    }

    /// Do a normalized linear interpolation with `other`, by `amount`.
    /// 
    /// This takes the shortest path, so if the quaternions have a negative
    /// dot product, the interpolation will be between `self` and `-other`.
    pub fn nlerp(self, mut other: Quaternion<S>, amount: S) -> Quaternion<S> {
        if self.dot(other) < S::zero() {
            other = -other;
        }

        (self * (S::one() - amount) + other * amount).normalize()
    }

    /// Spherical Linear Interpolation
    ///
    /// Return the spherical linear interpolation between the quaternion and
    /// `other`. Both quaternions should be normalized first.
    /// 
    /// This takes the shortest path, so if the quaternions have a negative
    /// dot product, the interpolation will be between `self` and `-other`.
    ///
    /// # Performance notes
    ///
    /// The `acos` operation used in `slerp` is an expensive operation, so
    /// unless your quaternions are far away from each other it's generally
    /// more advisable to use `nlerp` when you know your rotations are going
    /// to be small.
    ///
    /// - [Understanding Slerp, Then Not Using It]
    ///   (http://number-none.com/product/Understanding%20Slerp,%20Then%20Not%20Using%20It/)
    /// - [Arcsynthesis OpenGL tutorial]
    ///   (http://www.arcsynthesis.org/gltut/Positioning/Tut08%20Interpolation.html)
    pub fn slerp(self, mut other: Quaternion<S>, amount: S) -> Quaternion<S> {
        let mut dot = self.dot(other);
        let dot_threshold: S = cast(0.9995f64).unwrap();

        if dot < S::zero() {
            other = -other;
            dot = -dot;
        }

        // if quaternions are close together use `nlerp`
        if dot > dot_threshold {
            self.nlerp(other, amount)
        } else {
            // stay within the domain of acos()
            let robust_dot = dot.min(S::one()).max(-S::one());

            let theta = Rad::acos(robust_dot);

            let scale1 = Rad::sin(theta * (S::one() - amount));
            let scale2 = Rad::sin(theta * amount);

            (self * scale1 + other * scale2).normalize()
        }
    }

    pub fn is_finite(&self) -> bool {
        self.s.is_finite() && self.v.is_finite()
    }

    pub fn inverse(&self) -> Quaternion<S> {
        self.conjugate() / self.magnitude2()
    }

    #[inline]
    pub fn magnitude2(self) -> S {
        Self::dot(self, self)
    }

    #[inline]
    pub fn magnitude(self) -> S
    {
        Float::sqrt(self.magnitude2())
    }

    #[inline]
    pub fn squared_mag(self) -> S {
        Self::dot(self, self)
    }

    #[inline]
    pub fn inverse_mag(self) -> S {
        S::one() / Self::magnitude(self)
    }

    #[inline]
    pub fn normalize(self) -> Quaternion<S>
    {
        self.normalize_to(S::one())
    }

    #[inline]
    fn normalize_to(self, magnitude: S) -> Quaternion<S>
    {
        self * (magnitude / self.magnitude())
    }

    pub fn get_direction_x(&self) -> Vector3<S> {
        let two: S = cast(2.0f32).unwrap();
        Vector3::new(
            S::one() - two * (self.v.y * self.v.y + self.v.z * self.v.z),
            two * (self.v.x * self.v.y + self.s * self.v.z),
            two * (self.v.x * self.v.z - self.s * self.v.y)
        )
    }
    
    pub fn get_direction_y(&self) -> Vector3<S> {
        let two: S = cast(2.0f32).unwrap();
        Vector3::new(
            two * (self.v.x * self.v.y - self.s * self.v.z),
            S::one() - two * (self.v.x * self.v.x + self.v.z * self.v.z),
            two * (self.v.y * self.v.z + self.s * self.v.x)
        )
    }

    pub fn get_direction_z(&self) -> Vector3<S> {
        let two: S = cast(2.0f32).unwrap();
        Vector3::new(
            two * (self.v.x * self.v.z + self.s * self.v.y),
            S::one() - two * (self.v.y * self.v.z - self.s * self.v.x),
            two * (self.v.x * self.v.x + self.v.y * self.v.y)
        )
    }

    pub fn get_rotation_matrix(&self) -> Matrix3<S> {
        let x2 = self.v.x * self.v.x;
        let y2 = self.v.y * self.v.y;
        let z2 = self.v.z * self.v.z;
        let xy = self.v.x * self.v.y;
        let xz = self.v.x * self.v.z;
        let yz = self.v.y * self.v.z;
        let wx = self.s * self.v.x;
        let wy = self.s * self.v.y;
        let wz = self.s * self.v.z;
    
        let two: S = cast(2.0f32).unwrap();
        Matrix3::new(
            S::one() - two * (y2 + z2), two * (xy + wz), two * (xz - wy),
            two * (xy - wz), S::one() - two * (x2 + z2), two * (yz + wx),
            two * (xz + wy), two * (yz - wx), S::one() - two * (x2 + y2)
        )
    }

    pub fn set_rotation_matrix<'a, T>(&'a mut self, m: &T) -> &'a Quaternion<S>
    where
        T: Matrix<Scalar = S> + Index<usize, Output = <T as Matrix>::Column> + IndexMut<usize, Output = <T as Matrix>::Column>,
    {
        let m00 = m.row_col(0, 0);
        let m11 = m.row_col(1,1);
        let m22 = m.row_col(2,2,);
        let sum = m00 + m11 + m22;
        let half: S = cast(0.5f32).unwrap();
        let quarter: S = cast(0.25f32).unwrap();
        let t = Float::sqrt(m22 - m00 - m11 + S::one()) * half;
        if sum > S::zero() {
            self.s = t;
            let f = quarter / x;
            self.v.x = (m.row_col(2, 1) + m.row_col(1, 2)) * f;
            self.v.y = (m.row_col(0, 2) + m.row_col(2, 0)) * f;
            self.v.z = (m.row_col(1, 0) + m.row_col(0, 1)) * f;
        } else if m00 > m11 && m00 > m22 {
            self.v.x = t;
            let f = quarter / x;
            self.v.y = (m.row_col(1, 0) + m.row_col(0, 1)) * f;
            self.v.z = (m.row_col(0, 2) + m.row_col(2, 0)) * f;
            self.s = (m.row_col(2, 1) + m.row_col(1, 2)) * f;
        } else if m11 > m22 {
            self.v.y = t;
            let f = quarter / y;
            self.v.x = (m.row_col(1, 0) + m.row_col(0, 1)) * f;
            self.v.z = (m.row_col(2, 1) + m.row_col(1, 2)) * f;
            self.s = (m.row_col(0, 2) + m.row_col(2, 0)) * f;
        } else {
            self.v.z = t;
            let f = quarter / self.v.z;
            self.v.x = (m.row_col(0, 2) + m.row_col(2, 0)) * f;
            self.v.y = (m.row_col(2, 1) + m.row_col(1, 2)) * f;
            self.s = (m.row_col(1, 0) - m.row_col(0, 1)) * f;
        }
    }

    pub fn set_biv3_s<'a>(&'a mut self, v: &Bivector3<S>, s: S) -> &'a Quaternion<S> {
        self.v.x = v.x;
        self.v.y = v.y;
        self.v.z = v.z;
        self.s = s;
        self
    }

    pub fn set<'a>(&'a mut self, x: S, y: S, z: S, s: S) ->'a Quaternion<S> {
        self.v.x = x;
        self.v.y = y;
        self.v.z = z;
        self.s = s;
        self
    }

}


// impl_operator!(<S: BaseFloat> PartialEq<Quaternion<S> > for Quaternion<S> {
//     fn eq(q1, q2) -> bool {
//         q1.v == q2.v && q1.s == q2.s
//     }
// });

// impl_operator!(<S: BaseFloat> PartialEq<Bivector3<S> > for Quaternion<S> {
//     fn eq(q, v) -> bool {
//         q.v.x == v.x && q.v.y == v.y && q.v.z == v.z && q.s == S::zero()
//     }
// });

// impl_operator!(<S: BaseFloat> PartialEq<Quaternion<S> > for Bivector3<S> {
//     fn eq(v, q) -> bool {
//         q == v
//     }
// });

// impl_operator!(<S: BaseFloat> PartialEq<S> for Quaternion<S> {
//     fn eq(q, s) -> bool {
//         q.s == s && q.v == Vector3::zero()
//     }
// });


impl<S: BaseFloat> Zero for Quaternion<S> {
    #[inline]
    fn zero() -> Quaternion<S> {
        Quaternion::from_sv(S::zero(), Bivector3::zero())
    }

    #[inline]
    fn is_zero(&self) -> bool {
        ulps_eq!(self, &Quaternion::<S>::zero())
    }
}

impl<S: BaseFloat> One for Quaternion<S> {
    #[inline]
    fn one() -> Quaternion<S> {
        Quaternion::from_sv(S::one(), Bivector3::zero())
    }
}

impl<S: BaseFloat> iter::Sum<Quaternion<S>> for Quaternion<S> {
    #[inline]
    fn sum<I: Iterator<Item = Quaternion<S>>>(iter: I) -> Quaternion<S> {
        iter.fold(Quaternion::<S>::zero(), Add::add)
    }
}

impl<'a, S: 'a + BaseFloat> iter::Sum<&'a Quaternion<S>> for Quaternion<S> {
    #[inline]
    fn sum<I: Iterator<Item = &'a Quaternion<S>>>(iter: I) -> Quaternion<S> {
        iter.fold(Quaternion::<S>::zero(), Add::add)
    }
}

impl<S: BaseFloat> iter::Product<Quaternion<S>> for Quaternion<S> {
    #[inline]
    fn product<I: Iterator<Item = Quaternion<S>>>(iter: I) -> Quaternion<S> {
        iter.fold(Quaternion::<S>::one(), Mul::mul)
    }
}

impl<'a, S: 'a + BaseFloat> iter::Product<&'a Quaternion<S>> for Quaternion<S> {
    #[inline]
    fn product<I: Iterator<Item = &'a Quaternion<S>>>(iter: I) -> Quaternion<S> {
        iter.fold(Quaternion::<S>::one(), Mul::mul)
    }
}

impl<S: BaseFloat> VectorSpace for Quaternion<S> {
    type Scalar = S;
}

impl<S: BaseFloat> MetricSpace for Quaternion<S> {
    type Metric = S;

    #[inline]
    fn distance2(self, other: Self) -> S {
        (other - self).magnitude2()
    }
}

impl<S: NumCast + Copy> Quaternion<S> {
    pub fn cast<T: BaseFloat>(&self) -> Option<Quaternion<T>> {
        let s = match NumCast::from(self.s) {
            Some(s) => s,
            None => return None,
        };
        let v = match self.v.cast() {
            Some(v) => v,
            None => return None,
        };
        Some(Quaternion::from_sv(s, v))
    }
}

impl<S: BaseFloat> InnerSpace for Quaternion<S> {
    #[inline]
    default_fn!( dot(self, other: Quaternion<S>) -> S {
        self.s * other.s + self.v.dot(other.v)
    } );
}

impl<A> From<Euler<A>> for Quaternion<A::Unitless>
where
    A: Angle + Into<Rad<<A as Angle>::Unitless>>,
{
    fn from(src: Euler<A>) -> Quaternion<A::Unitless> {
        // Euclidean Space has an Euler to quat equation, but it is for a different order (YXZ):
        // http://www.euclideanspace.com/maths/geometry/rotations/conversions/eulerToQuaternion/index.htm
        // Page A-2 here has the formula for XYZ:
        // http://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19770024290.pdf

        let half = cast(0.5f64).unwrap();
        let (s_x, c_x) = Rad::sin_cos(src.x.into() * half);
        let (s_y, c_y) = Rad::sin_cos(src.y.into() * half);
        let (s_z, c_z) = Rad::sin_cos(src.z.into() * half);

        Quaternion::new(
            -s_x * s_y * s_z + c_x * c_y * c_z,
            s_x * c_y * c_z + s_y * s_z * c_x,
            -s_x * s_z * c_y + s_y * c_x * c_z,
            s_x * s_y * c_z + s_z * c_x * c_y,
        )
    }
}

impl<S: BaseFloat> From<Bivector3<S>> for Quaternion<S> {
    fn from(v: Bivector3<S> ) -> Quaternion<S> {
        Quaternion::new(
            v.x, v.y, v.z, S::zero()
        )
    }
}

// impl<S> From<S> for Quaternion<S> {
//     fn from(scalar: S) -> Quaternion<S> {
//         Quaternion::new(
//             S::zero(), S::zero(), S::zero(), scalar
//         )
//     }
// }


impl_operator!(<S: BaseFloat> Neg for Quaternion<S> {
    fn neg(quat) -> Quaternion<S> {
        Quaternion::from_sv(-quat.s, -quat.v)
    }
});

impl_operator!(<S: BaseFloat> Mul<S> for Quaternion<S> {
    fn mul(lhs, rhs) -> Quaternion<S> {
        Quaternion::from_sv(lhs.s * rhs, lhs.v * rhs)
    }
});

impl_assignment_operator!(<S: BaseFloat> MulAssign<S> for Quaternion<S> {
    fn mul_assign(&mut self, scalar) { self.s *= scalar; self.v *= scalar; }
});

impl_assignment_operator!(<S: BaseFloat> MulAssign<Quaternion<S>> for Quaternion<S> {
    fn mul_assign(&mut self, q) {
        let x = self.v.x;
        let y = self.v.y;
        let z = self.v.z;
        let w = self.s;
    
        let a = w * q.v.x + x * q.s + y * q.v.z - z * q.v.y;
        let b = w * q.v.y - x * q.v.z + y * q.s + z * q.v.x;
        let c = w * q.v.z + x * q.v.y - y * q.v.x + z * q.s;
    
        self.s = w * q.s - x * q.v.x - y * q.v.y - z * q.v.z;
        self.v.x = a;
        self.v.y = b;
        self.v.z = c;      
    }
});

impl_assignment_operator!(<S: BaseFloat> MulAssign<Bivector3<S>> for Quaternion<S> {
    fn mul_assign(&mut self, v) {
        let x = self.v.x;
        let y = self.v.y;
        let z = self.v.z;
        let w = self.s;

        let a = w * v.x + y * v.z - z * v.y;
        let b = w * v.y - x * v.z + z * v.x;
        let c = w * v.z + x * v.y - y * v.x;

        self.s = -x * v.x - y * v.y - z * v.z;
        self.v.x = a;
        self.v.y = b;
        self.v.z = c;
    }
});


// impl_operator!(<S: BaseFloat> Mul<Vector3<S> > for Quaternion<S> {
//     fn mul(lhs, rhs) -> Vector3<S> {{
//         let rhs = rhs.clone();
//         let two: S = cast(2i8).unwrap();
//         let tmp = lhs.v.cross(rhs) + (rhs * lhs.s);
//         (lhs.v.cross(tmp) * two) + rhs
//     }}
// });

impl_operator!(<S: BaseFloat> Mul<Quaternion<S> > for Quaternion<S> {
    fn mul(lhs, rhs) -> Quaternion<S> {
        Quaternion::new(
            lhs.s * rhs.s - lhs.v.x * rhs.v.x - lhs.v.y * rhs.v.y - lhs.v.z * rhs.v.z,
            lhs.s * rhs.v.x + lhs.v.x * rhs.s + lhs.v.y * rhs.v.z - lhs.v.z * rhs.v.y,
            lhs.s * rhs.v.y + lhs.v.y * rhs.s + lhs.v.z * rhs.v.x - lhs.v.x * rhs.v.z,
            lhs.s * rhs.v.z + lhs.v.z * rhs.s + lhs.v.x * rhs.v.y - lhs.v.y * rhs.v.x,
        )
    }
});

impl_operator!(<S: BaseFloat> Div<S> for Quaternion<S> {
    fn div(lhs, rhs) -> Quaternion<S> {
        Quaternion::from_sv(lhs.s / rhs, lhs.v / rhs)
    }
});

impl_operator!(<S: BaseFloat> Div<Quaternion<S>> for Quaternion<S> {
    fn div(q1, q2) -> Quaternion<S> {
        q1 * q2.inverse()
    }
});

// impl_operator!(<S: BaseFloat> Div<Bivector3<S>> for Quaternion<S> {
//     fn div(q, v) -> Quaternion<S> {
//         q * (-v / v.magnitude2())
//     }
// });

impl_assignment_operator!(<S: BaseFloat> DivAssign<S> for Quaternion<S> {
    fn div_assign(&mut self, scalar) { self.s /= scalar; self.v /= scalar; }
});

impl_assignment_operator!(<S: BaseFloat> DivAssign<Quaternion<S>> for Quaternion<S> {
    fn div_assign(&mut self, q) {
        let inv_q = q.inverse();
        *self *= inv_q
    }
});


impl_assignment_operator!(<S: BaseFloat> DivAssign<Bivector3<S>> for Quaternion<S> {
    fn div_assign(&mut self, v) {
        let normal_v = (-v) / v.magnitude2();
        *self *= normal_v
    }
});


impl_operator!(<S: BaseFloat> Rem<S> for Quaternion<S> {
    fn rem(lhs, rhs) -> Quaternion<S> {
        Quaternion::from_sv(lhs.s % rhs, lhs.v % rhs)
    }
});

impl_assignment_operator!(<S: BaseFloat> RemAssign<S> for Quaternion<S> {
    fn rem_assign(&mut self, scalar) { self.s %= scalar; self.v %= scalar; }
});

impl_operator!(<S: BaseFloat> Add<Quaternion<S> > for Quaternion<S> {
    fn add(lhs, rhs) -> Quaternion<S> {
        Quaternion::from_sv(lhs.s + rhs.s, lhs.v + rhs.v)
    }
});

impl_operator!(<S: BaseFloat> Add<Quaternion<S>> for Bivector3<S> {
    fn add(v, q) -> Quaternion<S> {
        Quaternion::new(v.x + q.v.x, v.y + q.v.y, v.z + q.v.z, q.s)
    }
});

impl_operator!(<S: BaseFloat> Add<S> for Quaternion<S> {
    fn add(q, s) -> Quaternion<S> {
        Quaternion::new(q.v.x, q.v.y, q.v.z, q.s + s)
    }
});



impl_assignment_operator!(<S: BaseFloat> AddAssign<Quaternion<S> > for Quaternion<S> {
    fn add_assign(&mut self, other) { self.s += other.s; self.v += other.v; }
});

impl_assignment_operator!(<S: BaseFloat> AddAssign<S> for Quaternion<S> {
    fn add_assign(&mut self, scalar) {
        self.s += scalar
    }
});

impl_operator!(<S: BaseFloat> Sub<Quaternion<S> > for Quaternion<S> {
    fn sub(lhs, rhs) -> Quaternion<S> {
        Quaternion::from_sv(lhs.s - rhs.s, lhs.v - rhs.v)
    }
});

impl_operator!(<S: BaseFloat> Sub<Bivector3<S> > for Quaternion<S> {
    fn sub(q, v) -> Quaternion<S> {
        Quaternion::new(q.v.x - v.x, q.v.y - v.y, q.v.z - v.z, q.s)
    }
});

impl_operator!(<S: BaseFloat> Sub<Quaternion<S> > for Bivector3<S> {
    fn sub(v, q) -> Quaternion<S> {
        Quaternion::new(v.x - q.v.x, v.y - q.v.y, v.z - q.v.z, -q.s)
    }
});

impl_operator!(<S:BaseFloat> Sub<S> for Quaternion<S> {
    fn sub(q, scalar) -> Quaternion<S> {
        Quaternion::new(q.v.x, q.v.y, q.v.z, q.s - scalar)
    }
});

impl_assignment_operator!(<S: BaseFloat> SubAssign<Quaternion<S> > for Quaternion<S> {
    fn sub_assign(&mut self, other) { self.s -= other.s; self.v -= other.v; }
});

impl_assignment_operator!(<S: BaseFloat> SubAssign<S> for Quaternion<S> {
    fn sub_assign(&mut self, scalar) {
        self.s -= scalar
    }
});

macro_rules! impl_scalar_ops {
    ($S:ident) => {
        impl_operator!(Div<Quaternion<$S>> for $S{
            fn div(scalar, quat) -> Quaternion<$S> {
                Quaternion::from_sv(scalar / quat.s, scalar / quat.v)
            }
        });
        impl_operator!(Mul<Quaternion<$S>> for $S {
            fn mul(scalar, quat) -> Quaternion<$S> {
                Quaternion::from_sv(scalar * quat.s, scalar * quat.v)
            }
        });
        impl_operator!(Add<Quaternion<$S>> for $S {
            fn add(scalar, quat) -> Quaternion<$S> {
                Quaternion::new(quat.v.x, quat.v.y, quat.v.z, quat.s + scalar)
            }
        });
        impl_operator!(Sub<Quaternion<$S>> for $S {
            fn sub(scalar, quat) -> Quaternion<$S> {
                Quaternion::new(-quat.v.x, -quat.v.y, -quat.v.z, scalar - quat.s)
            }
        });
        // impl_operator!(PartialEq<Quaternion<$S>> for $S {
        //     fn eq(scalar, quat) -> bool{
        //         quat.v == Vector3::zero() && scalar == quat.s
        //     }
        // });
    }
}

impl_scalar_ops!(f32);
impl_scalar_ops!(f64);

impl<S: BaseFloat> approx::AbsDiffEq for Quaternion<S> {
    type Epsilon = S::Epsilon;

    #[inline]
    fn default_epsilon() -> S::Epsilon {
        S::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: S::Epsilon) -> bool {
        S::abs_diff_eq(&self.s, &other.s, epsilon)
            && Bivector3::abs_diff_eq(&self.v, &other.v, epsilon)
    }
}

impl<S: BaseFloat> approx::RelativeEq for Quaternion<S> {
    #[inline]
    fn default_max_relative() -> S::Epsilon {
        S::default_max_relative()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: S::Epsilon, max_relative: S::Epsilon) -> bool {
        S::relative_eq(&self.s, &other.s, epsilon, max_relative)
            && Bivector3::relative_eq(&self.v, &other.v, epsilon, max_relative)
    }
}

impl<S: BaseFloat> approx::UlpsEq for Quaternion<S> {
    #[inline]
    fn default_max_ulps() -> u32 {
        S::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: S::Epsilon, max_ulps: u32) -> bool {
        S::ulps_eq(&self.s, &other.s, epsilon, max_ulps)
            && Bivector3::ulps_eq(&self.v, &other.v, epsilon, max_ulps)
    }
}



impl<S: BaseFloat> Into<[S; 4]> for Quaternion<S> {
    #[inline]
    fn into(self) -> [S; 4] {
        match self.into() {
            (xi, yj, zk, w) => [xi, yj, zk, w],
        }
    }
}

impl<S: BaseFloat> AsRef<[S; 4]> for Quaternion<S> {
    #[inline]
    fn as_ref(&self) -> &[S; 4] {
        unsafe { mem::transmute(self) }
    }
}

impl<S: BaseFloat> AsMut<[S; 4]> for Quaternion<S> {
    #[inline]
    fn as_mut(&mut self) -> &mut [S; 4] {
        unsafe { mem::transmute(self) }
    }
}

impl<S: BaseFloat> From<[S; 4]> for Quaternion<S> {
    #[inline]
    fn from(v: [S; 4]) -> Quaternion<S> {
        Quaternion::new(v[3], v[0], v[1], v[2])
    }
}

impl<'a, S: BaseFloat> From<&'a [S; 4]> for &'a Quaternion<S> {
    #[inline]
    fn from(v: &'a [S; 4]) -> &'a Quaternion<S> {
        unsafe { mem::transmute(v) }
    }
}

impl<'a, S: BaseFloat> From<&'a mut [S; 4]> for &'a mut Quaternion<S> {
    #[inline]
    fn from(v: &'a mut [S; 4]) -> &'a mut Quaternion<S> {
        unsafe { mem::transmute(v) }
    }
}

impl<S: BaseFloat> Into<(S, S, S, S)> for Quaternion<S> {
    #[inline]
    fn into(self) -> (S, S, S, S) {
        match self {
            Quaternion {
                s,
                v: Bivector3 { x, y, z },
            } => (x, y, z, s),
        }
    }
}

impl<S: BaseFloat> AsRef<(S, S, S, S)> for Quaternion<S> {
    #[inline]
    fn as_ref(&self) -> &(S, S, S, S) {
        unsafe { mem::transmute(self) }
    }
}

impl<S: BaseFloat> AsMut<(S, S, S, S)> for Quaternion<S> {
    #[inline]
    fn as_mut(&mut self) -> &mut (S, S, S, S) {
        unsafe { mem::transmute(self) }
    }
}

impl<S: BaseFloat> From<(S, S, S, S)> for Quaternion<S> {
    #[inline]
    fn from(v: (S, S, S, S)) -> Quaternion<S> {
        match v {
            (xi, yj, zk, w) => Quaternion::new(w, xi, yj, zk),
        }
    }
}

macro_rules! impl_matrix_2_quaternion {
    ($MatrixN:ident) => {
        impl<S: BaseFloat> From<$MatrixN<S>> for Quaternion<S> {
            fn from(m: $MatrixN<S>) -> Quaternion<S> {
                let m00 = m.x.x;
                let m11 = m.y.y;
                let m22 = m.z.z;
                let sum = m00 + m11 + m22;
                let half: S = cast(0.5f32).unwrap();
                let quater: S = cast(0.25f32).unwrap();
                let m01 = m.y.x;
                let m02 = m.z.x;
                let m10 = m.x.y;
                let m12 = m.z.y;
                let m20 = m.x.z;
                let m21 = m.y.z;
                if sum > S::zero() {
                    let w = Float::sqrt(sum + S::one()) * half;
                    let f = quater / w;
            
                    let x = (m21 - m12) * f;
                    let y = (m02 - m20) * f;
                    let z = (m10 - m01) * f;
                    Quaternion::new(x, y, z, w)
                }
                else if (m00 > m11) && (m00 > m22) {
                    let x = Float::sqrt(m00 - m11 - m22 + S::one()) * half;
                    let f = quater / x;
            
                    let y = (m10 + m01) * f;
                    let z = (m02 +m20) * f;
                    let w = (m21 - m12) * f;
                    Quaternion::new(x, y, z, w)
                }
                else if m11 > m22 {
                    let  y = Float::sqrt(m11 - m00 - m22 + S::one()) * half;
                    let  f = quater / y;
            
                    let x = (m10 + m01) * f;
                    let z = (m21 + m12) * f;
                    let w = (m02 - m20) * f;
                    Quaternion::new(x, y, z, w)
                }
                else {
                    let z =  Float::sqrt(m22 - m00 - m11 + S::one()) * half;
                    let  f = quater / z;
            
                    let x = (m02 + m20) * f;
                    let y = (m21 + m12) * f;
                    let w = (m10 - m01) * f;
                    Quaternion::new(x, y, z, w)
                }
            }
        }
    }
}

impl_matrix_2_quaternion!(Matrix3);
impl_matrix_2_quaternion!(Matrix4);

impl<'a, S: BaseFloat> From<&'a (S, S, S, S)> for &'a Quaternion<S> {
    #[inline]
    fn from(v: &'a (S, S, S, S)) -> &'a Quaternion<S> {
        unsafe { mem::transmute(v) }
    }
}

impl<'a, S: BaseFloat> From<&'a mut (S, S, S, S)> for &'a mut Quaternion<S> {
    #[inline]
    fn from(v: &'a mut (S, S, S, S)) -> &'a mut Quaternion<S> {
        unsafe { mem::transmute(v) }
    }
}

macro_rules! index_operators {
    ($S:ident, $Output:ty, $I:ty) => {
        impl<$S: BaseFloat> Index<$I> for Quaternion<$S> {
            type Output = $Output;

            #[inline]
            fn index<'a>(&'a self, i: $I) -> &'a $Output {
                let v: &[$S; 4] = self.as_ref();
                &v[i]
            }
        }

        impl<$S: BaseFloat> IndexMut<$I> for Quaternion<$S> {
            #[inline]
            fn index_mut<'a>(&'a mut self, i: $I) -> &'a mut $Output {
                let v: &mut [$S; 4] = self.as_mut();
                &mut v[i]
            }
        }
    };
}

index_operators!(S, S, usize);
index_operators!(S, [S], Range<usize>);
index_operators!(S, [S], RangeTo<usize>);
index_operators!(S, [S], RangeFrom<usize>);
index_operators!(S, [S], RangeFull);

#[cfg(feature = "rand")]
impl<S> Distribution<Quaternion<S>> for Standard
where
    Standard: Distribution<S>,
    Standard: Distribution<Vector3<S>>,
    S: BaseFloat,
{
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Quaternion<S> {
        Quaternion::from_sv(rng.gen(), rng.gen())
    }
}

#[cfg(feature = "mint")]
impl<S> From<mint::Quaternion<S>> for Quaternion<S> {
    fn from(q: mint::Quaternion<S>) -> Self {
        Quaternion {
            s: q.s,
            v: q.v.into(),
        }
    }
}

#[cfg(feature = "mint")]
impl<S: Clone> Into<mint::Quaternion<S>> for Quaternion<S> {
    fn into(self) -> mint::Quaternion<S> {
        mint::Quaternion {
            s: self.s,
            v: self.v.into(),
        }
    }
}

pub fn transform<S: BaseFloat>(v: &Vector3<S>, q: &Quaternion<S> ) -> Vector3<S> {
    let c = q.v * ((v ^ q.v) * cast(2.0f32).unwrap()) + (!q.v ^ v) * (q.s * cast(2.0f32).unwrap());
    v * (q.s * q.s - q.v.magnitude()) + c.complement()
}

