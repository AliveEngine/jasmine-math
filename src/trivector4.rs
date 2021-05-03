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
use vector::{Vector2, Vector3, Vector4};
use bivector3::{Bivector3, bivec3};

#[cfg(feature = "mint")]
use mint;

#[repr(C)]
#[derive(PartialEq, Eq, Copy, Debug, Clone, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Trivector4<S> {
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
    ($VectorN:ident { $($field:ident),+ }, $n:expr, $constructor:ident) => {
        impl<S> $VectorN<S> {
            /// Construct a new vector, using the provided values.
            #[inline]
            pub const fn new($($field: S),+) -> $VectorN<S> {
                $VectorN { $($field: $field),+ }
            }

            /// Perform the given operation on each field in the vector, returning a new point
            /// constructed from the operations.
            #[inline]
            pub fn map<U, F>(self, mut f: F) -> $VectorN<U>
                where F: FnMut(S) -> U
            {
                $VectorN { $($field: f(self.$field)),+ }
            }

            /// Construct a new vector where each component is the result of
            /// applying the given operation to each pair of components of the
            /// given vectors.
            #[inline]
            pub fn zip<S2, S3, F>(self, v2: $VectorN<S2>, mut f: F) -> $VectorN<S3>
                where F: FnMut(S, S2) -> S3
            {
                $VectorN { $($field: f(self.$field, v2.$field)),+ }
            }
        }

       /// The short constructor.
       #[inline]
       pub const fn $constructor<S>($($field: S),+) -> $VectorN<S> {
           $VectorN::new($($field),+)
       }

       impl<S: NumCast + Copy> $VectorN<S> {
           /// Component-wise casting to another type.
           #[inline]
           pub fn cast<T: NumCast>(&self) -> Option<$VectorN<T>> {
               $(
                   let $field = match NumCast::from(self.$field) {
                       Some(field) => field,
                       None => return None
                   };
               )+
               Some($VectorN { $($field),+ })
           }
       }


        impl<S: Copy> Array for $VectorN<S> {
            type Element = S;

            #[inline]
            fn len() -> usize {
                $n
            }

            #[inline]
            fn from_value(scalar: S) -> $VectorN<S> {
                $VectorN { $($field: scalar),+ }
            }

            #[inline]
            fn sum(self) -> S where S: Add<Output = S> {
                fold_array!(add, { $(self.$field),+ })
            }

            #[inline]
            fn product(self) -> S where S: Mul<Output = S> {
                fold_array!(mul, { $(self.$field),+ })
            }

            fn is_finite(&self) -> bool where S: Float {
                $(self.$field.is_finite())&&+
            }
        }


        impl<S: Neg<Output = S>> Neg for $VectorN<S> {
            type Output = $VectorN<S>;

            #[inline]
            default_fn!( neg(self) -> $VectorN<S> { $VectorN::new($(-self.$field),+) } );
        }

        impl<S: BaseFloat> approx::AbsDiffEq for $VectorN<S> {
            type Epsilon = S::Epsilon;

            #[inline]
            fn default_epsilon() -> S::Epsilon {
                S::default_epsilon()
            }

            #[inline]
            fn abs_diff_eq(&self, other: &Self, epsilon: S::Epsilon) -> bool {
                $(S::abs_diff_eq(&self.$field, &other.$field, epsilon))&&+
            }
        }

        impl<S: BaseFloat> approx::RelativeEq for $VectorN<S> {
            #[inline]
            fn default_max_relative() -> S::Epsilon {
                S::default_max_relative()
            }

            #[inline]
            fn relative_eq(&self, other: &Self, epsilon: S::Epsilon, max_relative: S::Epsilon) -> bool {
                $(S::relative_eq(&self.$field, &other.$field, epsilon, max_relative))&&+
            }
        }

        impl<S: BaseFloat> approx::UlpsEq for $VectorN<S> {
            #[inline]
            fn default_max_ulps() -> u32 {
                S::default_max_ulps()
            }

            #[inline]
            fn ulps_eq(&self, other: &Self, epsilon: S::Epsilon, max_ulps: u32) -> bool {
                $(S::ulps_eq(&self.$field, &other.$field, epsilon, max_ulps))&&+
            }
        }

        #[cfg(feature = "rand")]
        impl<S> Distribution<$VectorN<S>> for Standard
            where Standard: Distribution<S>,
                S: BaseFloat {
            #[inline]
            fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> $VectorN<S> {
                $VectorN { $($field: rng.gen()),+ }
            }
        }

        impl<S: Bounded> Bounded for $VectorN<S> {
            #[inline]
            fn min_value() -> $VectorN<S> {
                $VectorN { $($field: S::min_value()),+ }
            }

            #[inline]
            fn max_value() -> $VectorN<S> {
                $VectorN { $($field: S::max_value()),+ }
            }
        }

        impl_operator!(<S: BaseNum> Mul<S> for $VectorN<S> {
            fn mul(vector, scalar) -> $VectorN<S> { $VectorN::new($(vector.$field * scalar),+) }
        });    

        impl_assignment_operator!(<S: BaseNum> MulAssign<S> for $VectorN<S> {
            fn mul_assign(&mut self, scalar) { $(self.$field *= scalar);+ }
        });

        impl_operator!(<S: BaseNum> Div<S> for $VectorN<S> {
            fn div(vector, scalar) -> $VectorN<S> { $VectorN::new($(vector.$field / scalar),+) }
        });           
        impl_assignment_operator!(<S: BaseNum> DivAssign<S> for $VectorN<S> {
            fn div_assign(&mut self, scalar) { $(self.$field /= scalar);+ }
        });
        impl_scalar_ops!($VectorN<usize> { $($field),+ });
        impl_scalar_ops!($VectorN<u8> { $($field),+ });
        impl_scalar_ops!($VectorN<u16> { $($field),+ });
        impl_scalar_ops!($VectorN<u32> { $($field),+ });
        impl_scalar_ops!($VectorN<u64> { $($field),+ });
        impl_scalar_ops!($VectorN<isize> { $($field),+ });
        impl_scalar_ops!($VectorN<i8> { $($field),+ });
        impl_scalar_ops!($VectorN<i16> { $($field),+ });
        impl_scalar_ops!($VectorN<i32> { $($field),+ });
        impl_scalar_ops!($VectorN<i64> { $($field),+ });
        impl_scalar_ops!($VectorN<f32> { $($field),+ });
        impl_scalar_ops!($VectorN<f64> { $($field),+ });

        impl_index_operators!($VectorN<S>, $n, S, usize);
        impl_index_operators!($VectorN<S>, $n, [S], Range<usize>);
        impl_index_operators!($VectorN<S>, $n, [S], RangeTo<usize>);
        impl_index_operators!($VectorN<S>, $n, [S], RangeFrom<usize>);
        impl_index_operators!($VectorN<S>, $n, [S], RangeFull);
       
    };
}

macro_rules! impl_scalar_ops {
    ($VectorN:ident<$S:ident> { $($field:ident),+ }) => {
        impl_operator!(Mul<$VectorN<$S>> for $S {
            fn mul(scalar, vector) -> $VectorN<$S> { $VectorN::new($(scalar * vector.$field),+) }
        });
    };
}

impl_vector!(Trivector4 { x, y, z, w }, 4, trivec4);
impl_fixed_array_conversions!(Trivector4<S> {x: 0, y: 0, z: 0, w: 0 }, 4);
impl_tuple_conversions!(Trivector4<S> { x, y, z, w }, (S, S, S, S));

impl_operator!(<S:BaseNum> BitXor<Trivector4<S> > for Vector4<S> {
    fn bitxor(v, f) -> S {
        v.x * f.x + v.y + f.y + v.z * f.z + v.w * f.w
    }
});

impl_operator!(<S:BaseNum> BitXor<Trivector4<S> > for Vector3<S> {
    fn bitxor(v, f) -> S {
        v.x * f.x + v.y + f.y + v.z * f.z
    }
});

impl_operator!(<S:BaseNum> BitXor<Trivector4<S> > for Point3<S> {
    fn bitxor(p, f) -> S {
        p.x * f.x + p.y + f.y + p.z * f.z + f.w
    }
});


impl_operator!(<S:BaseNum> BitXor<Trivector4<S> > for Point2<S> {
    fn bitxor(p, f) -> S {
        p.x * f.x + p.y + f.y + f.w
    }
});

impl_operator!(<S:BaseNum> BitXor<Vector4<S> > for Trivector4<S> {
    fn bitxor(f, v) -> S {
        v.x * f.x + v.y + f.y + v.z * f.z + v.w * f.w
    }
});

impl_operator!(<S:BaseNum> BitXor<Vector3<S> > for Trivector4<S> {
    fn bitxor(f, v) -> S {
        v.x * f.x + v.y + f.y + v.z * f.z
    }
});

impl_grassmann_antiwedge! (<S:BaseNum>, Vector4<S>, Trivector4<S>, S);
impl_grassmann_antiwedge! (<S:BaseNum>, Vector3<S>, Trivector4<S>, S);
impl_grassmann_antiwedge! (<S:BaseNum>, Point3<S>, Trivector4<S>, S);
impl_grassmann_antiwedge! (<S:BaseNum>, Point2<S>, Trivector4<S>, S);
impl_grassmann_antiwedge! (<S:BaseNum>, Trivector4<S>, Vector4<S>, S);
impl_grassmann_antiwedge! (<S:BaseNum>, Trivector4<S>, Vector3<S>, S);

impl<S: BaseNum> Trivector4<S> {
    pub fn new_bivec3_s(bivec3: &Bivector3<S>, d: S) -> Trivector4<S> 
    {
        Trivector4::new(bivec3.x, bivec3.y, bivec3.z, d)
    }

    pub fn xyz(self) -> Bivector3<S> {
        Bivector3::new(self.x, self.y, self.z)
    }
}

impl<S:BaseNum> Trivector4<S> {

    pub fn new_bivec3_point3(bivec3: &Bivector3<S>, p: &Point3<S>) -> Trivector4<S> 
    where 
        S:Neg<Output = S>
    {
        let w = -(bivec3 ^ p);
        Trivector4::new(bivec3.x, bivec3.y, bivec3.z, w)
    }

    pub fn new_three_points(p1: Point3<S>, p2: Point3<S>, p3: Point3<S>) -> Trivector4<S> 
    {
        let v12: Vector3<S> = p2 - p1;
        let v13: Vector3<S> = p3 - p1;
        let xyz = v12 ^ v13;
        let w = xyz ^ p1;
        Trivector4::new(xyz.x ,xyz.y, xyz.z, w)
    }
}

impl<S:BaseNum> ProjectTrait<Trivector4<S>> for Point3<S> {
    fn project(&self, f: &Trivector4<S>) -> Self {
        self - (!f.xyz()) * (self ^ f)
    }
}

impl<S: BaseNum> AntiprojectTrait<Point3<S>> for Trivector4<S> 
where 
    S:Neg<Output = S>
{
    fn anti_project(&self, p: &Point3<S>) -> Trivector4<S>  
    {
        Trivector4::new_bivec3_point3(&self.xyz(), &p)
    }
}


