
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

