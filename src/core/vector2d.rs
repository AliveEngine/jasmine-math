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

#[cfg(feature = "mint")]
use mint;

/// A 1-dimensional vector.
///
/// This type is marked as `#[repr(C)]`.
##[repr(C)]
##[derive(PartialEq, Eq, Copy, Clone, Hash)]
##[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Vector2D<S> {
    /// x component of the vector
    pub x: S,
}

impl<S> Vector2D {

    #[inline]
    

}