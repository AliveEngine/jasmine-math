
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
use trivector4::{Trivector4};
use matrix::{Matrix3, Matrix4};
use transform::Transform4;
use quaternion::Quaternion;



/// The $motor$ class encapsulates a 4D motion operator (motor), also known as a dual quaternion.
/// form.
///
/// This type is marked as `#[repr(C)]`.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Motor<S> {
    //## The coordinates of the rotor part consisting of the weight components using basis elements <b>e</b><sub>41</sub>, <b>e</b><sub>42</sub>, <b>e</b><sub>43</sub>, and <b>e</b><sub>1234</sub>.
    pub rotor: Quaternion<S>,

    //## The coordinates of the screw part consisting of the bulk components using basis elements <b>e</b><sub>23</sub>, <b>e</b><sub>31</sub>, <b>e</b><sub>12</sub>, and 1.
    pub screw: Quaternion<S>
}

impl<S> Motor<S> 
{
    #[inline]
    pub const fn new(rx: S, ry: S, rz: S, rw: S, ux: S, uy: S, uz: S, uw: S) -> Motor<S> {
        Motor {
            rotor: Quaternion::new(rw, rx, ry, rz),
            screw: Quaternion::new(uw, ux, uy, uz)
        }
    }

}

pub fn motor<S>() -> Motor<S> where S: NumCast{
    Motor::new(cast(0).unwrap(), cast(0).unwrap(), cast(0).unwrap(),cast(0).unwrap(),cast(0).unwrap(),cast(0).unwrap(),cast(0).unwrap(),cast(0).unwrap())
}

impl<S: BaseFloat> Motor<S> {
    #[inline]
    pub fn from_quaternion(r: &Quaternion<S>) -> Motor<S> {
        Motor {
            rotor: Quaternion::new(r.s, r.v.x, r.v.y, r.v.z),
            screw: Quaternion::new(S::zero(), S::zero(), S::zero(), S::zero())
        }
    }

    #[inline]
    pub fn from_r_u(r: &Quaternion<S>, u: &Quaternion<S>) -> Motor<S> {
        Motor {
            rotor: Quaternion::new(r.s, r.v.x, r.v.y, r.v.z),
            screw: Quaternion::new(u.s, u.v.x, u.v.y, u.v.z)
        }
    }

    #[inline]
    pub fn from_f_g(f: &Trivector4<S>, g: &Trivector4<S>) -> Motor<S> {
        let r: Quaternion<S> = Quaternion::new(
            f.x * g.x + f.y * g.y + f.z * g.z,
            f.y * g.z - f.z * g.y, f.z * g.x - f.x * g.z, f.x * g.y - f.y * g.x
        );
        let u: Quaternion<S> = Quaternion::new(
            S::zero(),
            f.w * g.x - f.x * g.w, f.w * g.y - f.y * g.w, f.w * g.z - f.z * g.w
        );
        Motor::from_r_u(&r, &u)
    }

    #[inline]
    pub fn from_K_L(K: &Bivector4<S>, L: &Bivector4<S>) -> Motor<S> {
        let r_v3 = K.direction ^ L.direction;
        let r_s = -K.direction.dot(L.direction);
        let u_v3 = (L.direction ^ !K.moment) - (K.direction ^ !L.moment);
        let u_s = -(L.direction ^ K.moment) - (K.direction ^ L.moment);
        let r = Quaternion::from_sv(r_s, r_v3);
        let u = Quaternion::from_sv(u_s, u_v3);
        Motor::from_r_u(&r, &u)
    }

    #[inline]
    pub fn from_points(p: &Point3<S>, q: &Point3<S>) -> Motor<S> {
        let r: Quaternion<S> = Quaternion::new(
            S::zero(), S::zero(), S::zero(), S::zero(), 
        );
        let u: Quaternion<S> = Quaternion::new(
            S::zero(),
            p.x - q.x, p.y - q.y, p.z - q.z
        );
        Motor::from_r_u(&r, &u)
    }

    #[inline]
    pub fn from_rotation(r: Rad<S>, axis: &Bivector3<S>) -> Motor<S> {
        let (s, c) = Rad::sin_cos(r);
        Motor::new(
            axis.x * s, axis.y * s, axis.z * s, c,
            S::zero(), S::zero(), S::zero(), S::zero()
        )
    }

    #[inline]
    pub fn from_translation(offset: &Vector3<S>) -> Motor<S> {
        let half = cast(0.5f32).unwrap();
        Motor::new(
            S::zero(), S::zero(), S::zero(), S::one(),
            offset.x * half, offset.y * half, offset.z * half, S::zero()
        )        
    }

    #[inline]
    pub fn from_screw(r: Rad<S>, axis: &Bivector4<S>, disp: S) -> Motor<S> {
        let half = cast(0.5f32).unwrap();
        let (s, c) = Rad::sin_cos(r * half);
        Motor::new(
            axis.direction.x * s, axis.direction.y * s, axis.direction.z * s, 
            c, disp * axis.direction.x * c + axis.moment.x * s, disp * axis.direction.y * c + axis.moment.y * s, disp * axis.direction.z * c + axis.moment.z * s, -disp * s
        )
    }

    #[inline]
    pub fn set(&mut self, rx: S, ry: S, rz: S, rw: S, ux: S, uy: S, uz: S, uw: S) {
        self.rotor.set(rw, rx, ry, rz);
        self.screw.set(uw, ux, uy, uz);
    }

    #[inline]
    pub fn set_quaternion(&mut self, r: &Quaternion<S>){
        self.rotor.set(r.s, r.v.x, r.v.y, r.v.z);
        self.screw.set(S::zero(), S::zero(), S::zero(), S::zero());
    }

    #[inline]
    pub fn set_r_u(&mut self, r: &Quaternion<S>, u: &Quaternion<S>){
        self.rotor.set(r.s, r.v.x, r.v.y, r.v.z);
        self.screw.set(u.s, u.v.x, u.v.y, u.v.z);
    }

    pub fn get_transform_matrix(&self) -> Transform4<S> {
        let r = &self.rotor.v;
        let u = &self.screw.v;
        let t: Vector3<S> = !*u * self.rotor.s - !*r * self.screw.s - (u ^ r);
        let m = self.rotor.get_rotation_matrix();
        let two: S = cast(2.0f32).unwrap();
        let v3 = t * two;
        Transform4::from_mat3_vec3(&m, &v3)
    }

    pub fn set_transform_matrix<'a>(&'a mut self, m: &Transform4<S>) -> &'a Motor<S> {
        self.rotor.set_rotation_matrix(m);
        let r = &self.rotor.v;
        let half: S = cast(0.5f32).unwrap();
        let u = Vector3::new(m.row_col(0,3) * half, m.row_col(1,3) * half, m.row_col(2,3) * half);
        let biv3 = !u * rotor.w - (!r ^ u);
        let s = -(u  ^ r);
        self.screw.set_biv3_s(&biv3, s);
        self
    }

}

/// util functions.
impl<S: BaseFloat> Motor<S> {
    pub fn untize<'a>(&'a mut self) -> &'a Motor<S> {
        *self *= self.rotor.inverse_mag();
        self
    }


}

impl_assignment_operator!(<S: BaseFloat> AddAssign<Motor<S>> for Motor<S> {
    fn add_assign(&mut self, q) {
        self.rotor += q.rotor;
        self.screw += q.screw;
    }
});

impl_assignment_operator!(<S: BaseFloat> SubAssign<Motor<S>> for Motor<S> {
    fn sub_assign(&mut self, q) {
        self.rotor -= q.rotor;
        self.screw -= q.screw;
    }
});

impl_assignment_operator!(<S: BaseFloat> MulAssign<Motor<S>> for Motor<S> {
    fn mul_assign(&mut self, Q) {
        self.rotor *= Q.rotor;
        self.screw *= Q.screw;
    }
});

impl_assignment_operator!(<S: BaseFloat> MulAssign<S> for Motor<S> {
    fn mul_assign(&mut self, s) {
        self.rotor *= s;
        self.screw *= s;
    }
});

impl_assignment_operator!(<S: BaseFloat> DivAssign<S> for Motor<S> {
    fn div_assign(&mut self, s) {
        self.rotor *= s;
        self.screw *= s;
    }
});

/// Returns the antireverse of the motor 
impl_operator!(<S: BaseFloat> Not for Motor<S> {
    fn not(q) -> Motor<S> {
        Motor::new(
            -q.rotor.v.x, -q.rotor.v.y, - q.rotor.v.z, -q.rotor.s,
            -q.screw.v.x, -q.screw.v.y, -q.screw.v.z, q.screw.s
        )
    }
});

/// Returns the negation of the motor 
impl_operator!(<S: BaseFloat> Neg for Motor<S> {
    fn neg(q) -> Moto<S> {
        Motor::new(
            -q.rotor.v.x, -q.rotor.v.y, - q.rotor.v.z, -q.rotor.s,
            -q.screw.v.x, -q.screw.v.y, -q.screw.v.z, -q.screw.s
        )
    }
});

impl_operator!(<S: BaseFloat> Add<Motor<S>> for Motor<S> {
    fn add(a, b) -> Motor<S> {
        Motor::from_r_u(
            a.rotor + b.rotor,
            a.screw + b.screw
        )
    }
});

impl_operator!(<S: BaseFloat> Sub<Motor<S>> for Motor<S> {
    fn sub(a, b) -> Motor<S> {
        Motor::from_r_u(
            a.rotor - b.rotor,
            a.screw - b.screw
        )
    }
});

impl_operator!(<S: BaseFloat> Mul<S> for Motor<S> {
    fn mul(q, s) -> Motor<S> {
        Motor::new(
            q.rotor.v.x * s, q.rotor.v.y * s, q.rotor.v.z * s, q.rotor.s * s,
            q.screw.v.x * s, q.screw.v.y * s, q.screw.v.z * s, q.screw.s * s
        )
    }
});

impl_operator!(<S: BaseFloat> Div<S> for Motor<S> {
    fn div(q, s) -> Motor<S> {
        Motor::new(
            q.rotor.v.x / s, q.rotor.v.y / s, q.rotor.v.z / s, q.rotor.s / s,
            q.screw.v.x / s, q.screw.v.y / s, q.screw.v.z / s, q.screw.s / s
        )
    }
});

impl_operator!(<S: BaseFloat> Mul<Quaternion<S>> for Motor<S> {
    fn mul(q, r) -> Motor<S> {
        Motor::new(
            q.rotor.s * r.v.x + q.rotor.v.x * r.s + q.rotor.v.y * r.v.z - q.rotor.v.z * r.v.y,
            q.rotor.s * r.v.y - q.rotor.v.x * r.v.z + q.rotor.v.y * r.s + q.rotor.v.z * r.v.x,
            q.rotor.s * r.v.z + q.rotor.v.x * r.v.y - q.rotor.v.y * r.v.x + q.rotor.v.z * r.s,
            q.rotor.s * r.s - q.rotor.v.x * r.v.x - q.rotor.v.y * r.v.y - q.rotor.v.z * r.v.z,
            q.screw.w * r.v.z + q.screw.x * r.v.y - q.screw.y * r.v.x + q.screw.z * r.s,
            q.screw.w * r.v.x + q.screw.x * r.s + q.screw.y * r.v.z - q.screw.z * r.v.y,
            q.screw.w * r.v.y - q.screw.x * r.v.z + q.screw.y * r.s + q.screw.z * r.v.x,
            q.screw.w * r.s - q.screw.x * r.v.x - q.screw.y * r.v.y - q.screw.z * r.v.z
        )
    }
});


impl_operator!(<S: BaseFloat> Mul<Motor<S>> for Quaternion<S> {
    fn mul(r, q) -> Motor<S> {
        Motor::new(
            r.s * q.rotor.v.x + r.v.x * q.rotor.s + r.v.y * q.rotor.v.z - r.v.z * q.rotor.v.y,
            r.s * q.rotor.v.y - r.v.x * q.rotor.v.z + r.v.y * q.rotor.s + r.v.z * q.rotor.v.x,
            r.s * q.rotor.v.z + r.v.x * q.rotor.v.y - r.v.y * q.rotor.v.x + r.v.z * q.rotor.s,
            r.s * q.rotor.s - r.v.x * q.rotor.v.x - r.v.y * q.rotor.v.y - r.v.z * q.rotor.v.z,
            r.s * q.screw.z + r.v.x * q.screw.y - r.v.y * q.screw.x + r.v.z * q.screw.w,
            r.s * q.screw.x + r.v.x * q.screw.w + r.v.y * q.screw.z - r.v.z * q.screw.y,
            r.s * q.screw.y - r.v.x * q.screw.z + r.v.y * q.screw.w + r.v.z * q.screw.x,
            r.s * q.screw.w - r.v.x * q.screw.x - r.v.y * q.screw.y - r.v.z * q.screw.z
        )
    }
});

macro_rules! impl_scalar_ops {
    ($S:ident) => {
        impl_operator!(<$S> Mul<Motor<$S>> for $S {
            fn mul(s, q) -> Motor<S> {
                Motor::new(
                    q.rotor.v.x * s, q.rotor.v.y * s, q.rotor.v.z * s, q.rotor.s * s,
                    q.screw.v.x * s, q.screw.v.y * s, q.screw.v.z * s, q.screw.s * s
                )
            }
        });
    };
}

impl_scalar_ops!(f32);
impl_scalar_ops!(f64);


impl<S: NumCast + Copy> Motor<S> {
    /// Component-wise casting to another type.
    #[inline]
    pub fn cast<T: NumCast>(&self) -> Option<Motor<T>> {
        let rotor = match self.rotor.cast() {
            Some(rotor) => rotor,
            None => return None
        };
        let screw = match.self.screw.cast() {
            Some(screw) => screw,
            None => return None
        }
        Some(Motor { rotor: rotor, screw: screw})
    }
}

impl<S: BaseFloat> approx::AbsDiffEq for Motor<S> {
    type Epsilon = S::Epsilon;

    #[inline]
    fn default_epsilon() -> S::Epsilon {
        S::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: S::Epsilon) -> bool {
        self.rotor.abs_diff_eq(&other.rotor, epsilon) &&
        self.screw.abs_diff_eq(&other.screw, epsilon)
    }
}

impl<S: BaseFloat> approx::RelativeEq for Motor<S> {
    #[inline]
    fn default_max_relative() -> S::Epsilon {
        S::default_max_relative()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: S::Epsilon, max_relative: S::Epsilon) -> bool {
        self.rotor.relative_eq(&other.rotor, epsilon, max_relative) &&
        self.screw.relative_eq(&other.screw, epsilon, max_relative)
    }
}

impl<S: BaseFloat> approx::UlpsEq for Motor<S> {
    #[inline]
    fn default_max_ulps() -> u32 {
        S::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: S::Epsilon, max_ulps: u32) -> bool {
        self.rotor.ulps_eq(&other.rotor, epsilon, max_ulps) &&
        self.screw.ulps_eq(&other.screw, epsilon, max_ulps)
    }
}

#[cfg(feature = "rand")]
impl<S> Distribution<Bivector4<S>> for Standard
    where Standard: Distribution<S>,
        S: BaseFloat {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Motor<S> {
        Motor{
            rotor: self.sample(rng),
            screw: self.sample(rng)
        }
    }
}

// todo simd
impl_operator!(<S: BaseFloat> TransformTrait<Motor<S>> for Point3<S> {
    fn transform(p, q) -> Point3<S> {
        let rx = q.rotor.x;
        let ry = q.rotor.y;
        let rz = q.rotor.z;
        let rw = q.rotor.w;
        let ux = q.screw.x;
        let uy = q.screw.y;
        let uz = q.screw.z;
        let uw = q.screw.w;
    
        let rx2 = rx * rx;
        let ry2 = ry * ry;
        let rz2 = rz * rz;
        let rxy = rx * ry;
        let rzx = rz * rx;
        let ryz = ry * rz;
    
        Vector3D v = !q.screw.v + p * rw;
    
        Motor::new(
            p.x + (ux * rw - rx * uw + (ry * v.z - rz * v.y) + p.y * rxy + p.z * rzx - p.x * (ry2 + rz2)) * 2.0F,
            p.y + (uy * rw - ry * uw + (rz * v.x - rx * v.z) + p.x * rxy + p.z * ryz - p.y * (rz2 + rx2)) * 2.0F,
            p.z + (uz * rw - rz * uw + (rx * v.y - ry * v.x) + p.x * rzx + p.y * ryz - p.z * (rx2 + ry2)) * 2.0F
        )
    }
});

impl_operator!(<S: BaseFloat> TransformTrait<Motor<S>> for Bivector4<S> {
    fn transform(l, q) -> Bivector4<S> {
        let rx = Q.rotor.x;
        let ry = Q.rotor.y;
        let rz = Q.rotor.z;
        let rw = Q.rotor.w;
        let ux = Q.screw.x;
        let uy = Q.screw.y;
        let uz = Q.screw.z;
        let uw = Q.screw.w;

        let rx2 = rx * rx;
        let ry2 = ry * ry;
        let rz2 = rz * rz;
        let rxy = rx * ry;
        let rzx = rz * rx;
        let ryz = ry * rz;
        let rwx = rw * rx;
        let rwy = rw * ry;
        let rwz = rw * rz;

        let v: Vector3<S> = L.direction * rw;

        Bivector3::new(L.direction.x + ((ry * v.z - rz * v.y) + L.direction.y * rxy + L.direction.z * rzx - L.direction.x * (ry2 + rz2)) * 2.0F,
        L.direction.y + ((rz * v.x - rx * v.z) + L.direction.x * rxy + L.direction.z * ryz - L.direction.y * (rz2 + rx2)) * 2.0F,
        L.direction.z + ((rx * v.y - ry * v.x) + L.direction.x * rzx + L.direction.y * ryz - L.direction.z * (rx2 + ry2)) * 2.0F,
        L.moment.x + (L.moment.z * (rwy + rzx) - L.moment.y * (rwz + rxy) + L.direction.y * (ux * ry + uy * rx - uz * rw - rz * uw) + L.direction.z * (rz * ux + rw * uy + rx * uz + ry * uw) - L.direction.x * (uy * ry - uz * rz) * 2.0F - L.moment.x * (ry2 + rz2)) * 2.0F,
        L.moment.y + (L.moment.x * (rwz + rxy) - L.moment.z * (rwx + ryz) + L.direction.z * (uy * rz + uz * ry - ux * rw - rx * uw) + L.direction.x * (ry * ux + rx * uy + rw * uz + rz * uw) - L.direction.y * (ux * rx - uz * rz) * 2.0F - L.moment.y * (rz2 + rx2)) * 2.0F,
        L.moment.z + (L.moment.y * (rwx + ryz) - L.moment.x * (rwy + rzx) + L.direction.x * (uz * rx + ux * rz - uy * rw - ry * uw) + L.direction.y * (rw * ux + rz * uy + ry * uz + rx * uw) - L.direction.z * (ux * rx - uy * ry) * 2.0F - L.moment.z * (rx2 + ry2)) * 2.0F
    )
});

impl_operator!(<S: BaseFloat> TransformTrait<Motor<S>> for Trivector4<S> {
    fn transform(f, q) -> Trivector4<S> {
        let rx = Q.rotor.x;
        let ry = Q.rotor.y;
        let rz = Q.rotor.z;
        let rw = Q.rotor.w;
        let ux = Q.screw.x;
        let uy = Q.screw.y;
        let uz = Q.screw.z;
        let uw = Q.screw.w;

        let rx2 = rx * rx;
        let ry2 = ry * ry;
        let rz2 = rz * rz;
        let rxy = rx * ry;
        let rzx = rz * rx;
        let ryz = ry * rz;

        let fr = f.x * rx + f.y * ry + f.z * rz;
        let fu = f.x * ux + f.y * uy + f.z * uz;

        Trivector4::new(f.x + ((ry * f.z - rz * f.y) * rw + f.y * rxy + f.z * rzx - f.x * (ry2 + rz2)) * 2.0F,
            f.y + ((rz * f.x - rx * f.z) * rw + f.x * rxy + f.z * ryz - f.y * (rz2 + rx2)) * 2.0F,
            f.z + ((rx * f.y - ry * f.x) * rw + f.x * rzx + f.y * ryz - f.z * (rx2 + ry2)) * 2.0F,
            f.w + (fr * uw - fu * rw + f.x * (ry * uz - rz * uy) + f.y * (rz * ux - rx * uz) + f.z * (rx * uy - ry * ux)) * 2.0F
        ）
    }
})



/// todo simd
// pub fn Transform<S: BaseFloat>(p: &Point3<S>, q: &Motor<S>) -> Point3<S> {
//     let rx = q.rotor.x;
//     let ry = q.rotor.y;
//     let rz = q.rotor.z;
//     let rw = q.rotor.w;
//     let ux = q.screw.x;
//     let uy = q.screw.y;
//     let uz = q.screw.z;
//     let uw = q.screw.w;

//     let rx2 = rx * rx;
//     let ry2 = ry * ry;
//     let rz2 = rz * rz;
//     let rxy = rx * ry;
//     let rzx = rz * rx;
//     let ryz = ry * rz;

//     Vector3D v = !q.screw.v + p * rw;

//     Motor::new(
//         p.x + (ux * rw - rx * uw + (ry * v.z - rz * v.y) + p.y * rxy + p.z * rzx - p.x * (ry2 + rz2)) * 2.0F,
//         p.y + (uy * rw - ry * uw + (rz * v.x - rx * v.z) + p.x * rxy + p.z * ryz - p.y * (rz2 + rx2)) * 2.0F,
//         p.z + (uz * rw - rz * uw + (rx * v.y - ry * v.x) + p.x * rzx + p.y * ryz - p.z * (rx2 + ry2)) * 2.0F
//     )
// }

pub fn 


