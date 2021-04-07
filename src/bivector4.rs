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
use point::{Point2};
use vector::{Vector2, Vector3};
use bivector3::{Bivector3, bivec3};

#[cfg(feature = "mint")]
use mint;



/// Bivector4D in four dimensional bivector having size float components.
/// 
#[repr(C)]
#[derive(PartialEq, Eq, Copy, Clone, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Bivector4<S> {
    pub direction: Vector3<S>,
    pub moment: Bivector3<S>,
}

impl<S> Bivector4<S> {
            
    #[inline]
    pub const fn new(dx: S, dy: S, dz: S, mx: S, my: S, mz: S) -> Bivector4<S> {
        Bivector4{
            direction: Vector3{x:dx, y:dy, z:dz},
            moment: Bivector3{x:mx, y:my, z:mz}
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
