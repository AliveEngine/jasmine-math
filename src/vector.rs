


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

// Utility macro for generating associated functions for the vectors
macro_rules! impl_vector {
    ($VectorN:ident {$($field:ident),+ }, $n:expr, $constructor:ident) => {
        impl<S> $VectorN<S> {
            /// Construct a new vector, using the provided values.
            #[inline]
            pub const fn new($($field: S),+ ) -> $VectorN<S> {
                $VectorN{$($field: $field),+ }
            }

            /// Perform the given operation on each field in the vector, returning a new point constructed from the operations.
            #[inline]
            pub fn map<U, F>(self, mut f: F) -> $VectorN<U>
                where F: FnMut(S) -> U
            {
                $VectorN {$($field: f(self.$field)),+ }
            }

            /// Construct a new vector where each component is the result of
            /// applying the given operatin to each pair of components of the
            /// given vecctors.
            #[inline]
            pub fn zip<S2, S3, F>(self, v2: $VectorN<S2>, mut f: F) -> $VectorN<S3>
                where F: FnMut<S,S2> -> S3 
            {
                $VectorN{$($field: f(self.$field, v2.$field)),+ }
            }
        }




    };
}

impl_vector!(Vector1 { x }, 1, vec1);
impl_vector!(Vector2 { x, y }, 2, vec2);
impl_vector!(Vector3 { x, y, z }, 3, vec3);
impl_vector!(Vector4 { x, y, z, w }, 4, vec4);

impl_fixed_array_conversions!(Vector1<S> {x: 0 }, 1);
impl_fixed_array_conversions!(Vector2<S> {x: 0, y: 0 }, 2);
impl_fixed_array_conversions!(Vector3<S> {x: 0, y: 0, z: 0 }, 3);
impl_fixed_array_conversions!(Vector4<S> {x: 0, y: 0, z: 0, w: 0 }, 4);

impl_tuple_conversions!(Vector1<S> { x }, (S,));
impl_tuple_conversions!(Vector2<S> { x, y }, (S, S));
impl_tuple_conversions!(Vector3<S> { x, y, z }, (S, S, S));
impl_tuple_conversions!(Vector4<S> { x, y, z, w }, (S, S, S, S));




