use num_traits::{Bounded, Float, NumCast};
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

use angle::Rad;
use approx;
use num::{BaseFloat, BaseNum};
use point::{Point2, Point3};
use vector::{Vector2, Vector3};
use bivector3::{Bivector3, bivec3};

#[cfg(feature = "mint")]
use mint;

use crate::Trivector4;


/// Bivector4D in four dimensional bivector having size float components.
/// 
#[repr(C)]
#[derive(PartialEq, Eq, Copy, Clone, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Bivector4<S> {
    pub direction: Vector3<S>,
    pub moment: Bivector3<S>,
}

impl<S> Bivector4<S> 
{
            
    #[inline]
    pub const fn new(dx: S, dy: S, dz: S, mx: S, my: S, mz: S) -> Bivector4<S> {
        Bivector4{
            direction: Vector3{x:dx, y:dy, z:dz},
            moment: Bivector3{x:mx, y:my, z:mz}
        }
    }

    pub const fn new_dir_m(dir: Vector3<S>, m: Bivector3<S>) -> Bivector4<S> {
        Bivector4{
            direction: dir,
            moment: m
        }
    }

    /// Perform the given operation on each field in the vector, returning a new point
    /// constructed from the operations.
    #[inline]
    pub fn map<U, F>(self, mut f: F) -> Bivector4<U>
        where F: FnMut(S) -> U
    {
        Bivector4 {
            direction: Vector3{x: f(self.direction.x), y: f(self.direction.y), z: f(self.direction.z) },
            moment: Bivector3{x: f(self.moment.x), y: f(self.moment.y), z: f(self.moment.z) } 
        }
    }

    /// Construct a new vector where each component is the result of
    /// applying the given operation to each pair of components of the
    /// given vectors.
    #[inline]
    pub fn zip<S2, S3, F>(self, v2: Bivector4<S2>, mut f: F) -> Bivector4<S3>
        where F: FnMut(S, S2) -> S3
    {
        Bivector4{ 
            direction: Vector3{ x: f(self.direction.x, v2.direction.x), y: f(self.direction.y, v2.direction.y), z: f(self.direction.z, v2.direction.z), },
            moment: Bivector3{ x: f(self.moment.x, v2.moment.x), y: f(self.moment.y, v2.moment.y), z: f(self.moment.z, v2.moment.z), },
        }
    }
}

#[inline]
pub const fn bivec4<S>(dir: Vector3<S>, m:Bivector3<S>) -> Bivector4<S> {
    Bivector4{direction: dir, moment: m}
}

impl<S: BaseNum> Bivector4<S> {
    pub fn new_point3_point3(p: Point3<S>, q: Point3<S>) -> Bivector4<S> {
        let dir: Vector3<S> = q - p;
        let m: Bivector3<S> = p.cross(&q);
        Bivector4::new_dir_m(dir, m)
    }

    pub fn new_point3_vec3(p:Point3<S>, v:Vector3<S>) -> Bivector4<S> {
        let m: Bivector3<S> = Bivector3::new(p.y * v.z - p.z * v.y, p.z * v.x - p.x * v.z, p.x * v.y - p.y * v.x);
        Bivector4::new_dir_m(v, m)
    }

    pub fn new_trivec4_trivec4(f: &Trivector4<S>, g: &Trivector4<S>) -> Bivector4<S> {
        let dir: Vector3<S> = Vector3::new(f.y * g.z - f.z * g.y, f.z * g.x - f.x * g.z, f.x * g.y - f.y * g.x);
        let m: Bivector3<S> = Bivector3::new(f.w * g.x - f.x * g.w, f.w * g.y - f.y * g.w, f.w * g.z - f.z * g.w);
        Bivector4{direction: dir,moment: m}
    }

    pub fn set_v_bv(&mut self, dir: Vector3<S>, m:Bivector3<S>) {
        self.direction = dir;
        self.moment = m;
    }

    pub fn set_p_p(&mut self, p: Point3<S>, q: Point3<S>) 
    where
        S: Neg<Output = S> + BaseFloat
    {
        self.direction = q - p;
        self.moment.set(p.y * q.z - q.y * p.z, p.z * q.x - q.z * p.x, p.x * q.y - q.x * p.y);
    }

    pub fn set_p_v(&mut self, p: Point3<S>, v: Vector3<S>) {
        self.direction = v;
        self.moment.set(p.y * v.z - v.y * p.z, p.z * v.x - v.z * p.x, p.x * v.y - v.x * p.y);
    }

    pub fn set_trvec4_trvec4(&mut self, f: Trivector4<S>, g: Trivector4<S>) {
        self.direction.set(f.y * g.z - f.z * g.y, f.z * g.x - f.x * g.z, f.x * g.y - f.y * g.x);
        self.moment.set(f.w * g.x - f.x * g.w, f.w * g.y - f.y * g.w, f.w * g.z - f.z * g.w);
    }

    pub fn get_support(self) -> Vector3<S> {
        !self.direction ^ self.moment
    }
}

impl<S: BaseNum> MulAssign<S> for Bivector4<S> {
    fn mul_assign(&mut self, scalar: S) {
        self.direction *= scalar;
        self.moment *= scalar;
    }
}

impl<S: BaseNum> DivAssign<S> for Bivector4<S> {
    fn div_assign(&mut self, scalar: S) {
        
    }
}