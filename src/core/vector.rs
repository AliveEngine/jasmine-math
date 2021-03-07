


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
pub struct Vector1<S> {
    /// x component of the vector
    pub x: S,
}

/// A 2-dimensional vector.
///
/// This type is marked as `#[repr(C)]`.
##[repr(C)]
##[derive(PartialEq, Eq, Copy, Clone, Hash)]
##[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Vector2<S> {
    /// x component of the vector
    pub x: S,
    pub y: S,
}

/// A 3-dimensional vector.
///
/// This type is marked as `#[repr(C)]`.
#[repr(C)]
#[derive(PartialEq, Eq, Copy, Clone, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Vector3<S> {
    /// The x component of the vector.
    pub x: S,
    /// The y component of the vector.
    pub y: S,
    /// The z component of the vector.
    pub z: S,
}

/// A 4-dimensional vector.
///
/// This type is marked as `#[repr(C)]`.
#[repr(C)]
#[derive(PartialEq, Eq, Copy, Clone, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Vector4<S> {
    /// The x component of the vector.
    pub x: S,
    /// The y component of the vector.
    pub y: S,
    /// The z component of the vector.
    pub z: S,
    /// The w component of the vector.
    pub w: S,
}

