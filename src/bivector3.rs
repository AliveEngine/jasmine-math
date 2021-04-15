//use mint::Vector3;
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

#[cfg(feature = "mint")]
use mint;

/// A 3-dimensional bivector having three float components.
/// x,y,z is the value of $e_23,e_31,e_12$ coordinate.
/// This type is marked as `#[repr(C)]`.
#[repr(C)]
#[derive(PartialEq, Eq, Copy, Clone, Hash, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Bivector3<S> {
    pub x: S,
    pub y: S,
    pub z: S,
}

macro_rules! impl_scalar_ops {
    ($VectorN:ident<$S:ident> { $($field:ident),+ }) => {
        impl_operator!(Mul<$VectorN<$S>> for $S {
            fn mul(scalar, vector) -> $VectorN<$S> { $VectorN::new($(scalar * vector.$field),+) }
        });
        impl_operator!(Div<$VectorN<$S>> for $S {
            fn div(scalar, vector) -> $VectorN<$S> { $VectorN::new($(scalar / vector.$field),+) }
        });
        impl_operator!(Rem<$VectorN<$S>> for $S {
            fn rem(scalar, vector) -> $VectorN<$S> { $VectorN::new($(scalar % vector.$field),+) }
        });
    };
}

macro_rules! impl_bivector3 {
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


        impl<S: BaseNum> iter::Sum<$VectorN<S>> for $VectorN<S> {
            #[inline]
            fn sum<I: Iterator<Item=$VectorN<S>>>(iter: I) -> $VectorN<S> {
                iter.fold($VectorN::zero(), Add::add)
            }
        }

        impl<'a, S: 'a + BaseNum> iter::Sum<&'a $VectorN<S>> for $VectorN<S> {
            #[inline]
            fn sum<I: Iterator<Item=&'a $VectorN<S>>>(iter: I) -> $VectorN<S> {
                iter.fold($VectorN::zero(), Add::add)
            }
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


        impl<S: BaseNum> ElementWise for $VectorN<S> {
            #[inline] default_fn!( add_element_wise(self, rhs: $VectorN<S>) -> $VectorN<S> { $VectorN::new($(self.$field + rhs.$field),+) } );
            #[inline] default_fn!( sub_element_wise(self, rhs: $VectorN<S>) -> $VectorN<S> { $VectorN::new($(self.$field - rhs.$field),+) } );
            #[inline] default_fn!( mul_element_wise(self, rhs: $VectorN<S>) -> $VectorN<S> { $VectorN::new($(self.$field * rhs.$field),+) } );
            #[inline] default_fn!( div_element_wise(self, rhs: $VectorN<S>) -> $VectorN<S> { $VectorN::new($(self.$field / rhs.$field),+) } );
            #[inline] fn rem_element_wise(self, rhs: $VectorN<S>) -> $VectorN<S> { $VectorN::new($(self.$field % rhs.$field),+) }

            #[inline] default_fn!( add_assign_element_wise(&mut self, rhs: $VectorN<S>) { $(self.$field += rhs.$field);+ } );
            #[inline] default_fn!( sub_assign_element_wise(&mut self, rhs: $VectorN<S>) { $(self.$field -= rhs.$field);+ } );
            #[inline] default_fn!( mul_assign_element_wise(&mut self, rhs: $VectorN<S>) { $(self.$field *= rhs.$field);+ } );
            #[inline] default_fn!( div_assign_element_wise(&mut self, rhs: $VectorN<S>) { $(self.$field /= rhs.$field);+ } );
            #[inline] fn rem_assign_element_wise(&mut self, rhs: $VectorN<S>) { $(self.$field %= rhs.$field);+ }
        }


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


impl_bivector3!(Bivector3 { x, y, z }, 3, bivec3);
impl_fixed_array_conversions!(Bivector3<S> {x: 0, y: 0, z: 0 }, 3);
impl_tuple_conversions!(Bivector3<S> { x, y, z }, (S, S, S));

impl<S: BaseNum> Bivector3<S> 
{

    #[inline]
    pub fn from_points(p: Point2<S>, q: Point2<S>) -> Bivector3<S> {
        Bivector3{ x: p.y - q.y, y: q.x - p.x, z: p.x * q.y - p.y * q.x}
    }

    #[inline]
    pub fn point_vector(p: Point2<S>, v: Vector2<S>) -> Bivector3<S> 
    where
        S: Neg<Output = S> + BaseFloat + Zero +  One 
    {
        Bivector3{x: -v.y,y: v.x,z: p.x * v.y - p.y * v.x}
    }

    #[inline]
    pub fn yz_unit() -> Bivector3<S> { Bivector3::new(S::one(), S::zero(), S::zero()) }

    #[inline]
    pub fn zx_unit() -> Bivector3<S> { Bivector3::new(S::zero(), S::one(), S::zero()) }

    #[inline]
    pub fn xy_unit() -> Bivector3<S> { Bivector3::new(S::zero(), S::zero(), S::one()) }
    
    #[inline]
    pub fn minus_yz_unit() -> Bivector3<S>
    where
        S: Neg<Output = S> + BaseFloat + Zero +  One 
    { 
        Bivector3::new(-S::one(), S::zero(), S::zero())
    }

    #[inline]
    pub fn minus_zx_unit() -> Bivector3<S>
    where
        S: Neg<Output = S> + BaseFloat + Zero +  One
    { 
        Bivector3::new(S::zero(), -S::one(), S::zero())
     }

    #[inline]
    pub fn minus_xy_unit() -> Bivector3<S> 
    where
        S: Neg<Output = S> + BaseFloat + Zero +  One 
    {
        Bivector3::new(S::zero(), S::zero(), -S::one()) 
    }

    #[inline]
    pub fn set(&mut self, a: S, b: S, c: S) {
        self.x = a; self.y = b; self.z = c;
    }

    #[inline]
    pub fn dot(self, other: Bivector3<S>) -> S {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    #[inline]
    pub fn magnitude(self) -> S
    where
        S: Float,
    {
        S::sqrt(self.magnitude2())
    }

    #[inline]
    pub fn magnitude2(self) -> S {
        Self::dot(self, self)
    }

    #[inline]
    pub fn inverse_mag(self) -> S 
    where
        S: Neg<Output = S> + BaseFloat + Zero +  One 
    {
        S::one() / Self::magnitude(self)
    }

    #[inline]
    pub fn normalize(self) -> Bivector3<S> 
    where
        S: Neg<Output = S> + BaseFloat + Zero +  One 
    {
        self * Self::inverse_mag(self)
    }

}


impl_operator!(<S: BaseFloat> Rem<S> for Bivector3<S> {
    fn rem(lhs, rhs) -> Bivector3<S> {
        Bivector3::new(lhs.x % rhs, lhs.y % rhs, lhs.z % rhs)
    }
});

impl_assignment_operator!(<S: BaseFloat> RemAssign<S> for Bivector3<S> {
    fn rem_assign(&mut self, scalar) { self.x %= scalar; self.y %= scalar; self.z %= scalar; }
});

impl<S: BaseNum> Zero for Bivector3<S> {
    #[inline]
    fn zero() -> Bivector3<S> {
        Bivector3::from_value(S::zero())
    }

    #[inline]
    fn is_zero(&self) -> bool {
        *self == Bivector3::zero()
    }
}

impl<S:BaseNum> Not for Bivector3<S> {
    type Output = Vector3<S>;

    fn not(self) -> Vector3<S> {
        Vector3{x: self.x, y : self.y, z: self.z} 
    }
}


impl<S:BaseNum> Not for Vector3<S> {
    type Output = Bivector3<S>;

    fn not(self) -> Bivector3<S> {
        Bivector3{x: self.x, y : self.y, z: self.z} 
    }
}


impl<S:BaseNum> ComplementTrait<Vector3<S>> for Bivector3<S> {
    fn complement(self) -> Vector3<S> {
        Vector3{x: self.x, y: self.y, z: self.z}
    }
}

impl<S:BaseNum> ComplementTrait<Bivector3<S>> for Vector3<S> {
    fn complement(self) -> Bivector3<S> {
        Bivector3{x: self.x, y: self.y, z: self.z}
    }
}

impl_operator!(<S: BaseNum> Add<Bivector3<S> > for Bivector3<S> {
    fn add(lhs, rhs) -> Bivector3<S> { Bivector3{x:lhs.x + rhs.x, y: lhs.y + rhs.y, z: lhs.z + rhs.z}}
});

impl_assignment_operator!(<S: BaseNum> AddAssign<Bivector3<S> > for Bivector3<S> {
    fn add_assign(&mut self, other) { self.x += other.x;self.y += other.y; self.z += other.z;}
});

impl_operator!(<S: BaseNum> Sub<Bivector3<S> > for Bivector3<S> {
    fn sub(lhs, rhs) -> Bivector3<S> { 
        Bivector3{x:lhs.x- rhs.x, y:lhs.y-rhs.y,z:lhs.z-rhs.z}
    }
});
impl_assignment_operator!(<S: BaseNum> SubAssign<Bivector3<S> > for Bivector3<S> {
    fn sub_assign(&mut self, other) { self.x -= other.x; self.y -= other.y; self.z -= other.z; }
});

impl<S: Neg<Output = S>> Neg for Bivector3<S> {
    type Output = Bivector3<S>;

    #[inline]
    default_fn!( neg(self) -> Bivector3<S> { Bivector3{x:-self.x,y:-self.y,z:-self.z}} );
}

impl_operator!(<S: BaseNum> Mul<S> for Bivector3<S> {
    fn mul(vector, scalar) -> Bivector3<S> { Bivector3{x:vector.x * scalar, y: vector.y * scalar, z:vector.z * scalar}}
});    

impl_operator!(<S: BaseNum> Mul<Bivector3<S>> for Bivector3<S> {
    fn mul(a, b) -> Bivector3<S> { 
        Bivector3 {
        x: a.x * b.x,
        y: a.y * b.y,
        z: a.z * b.z
    }}
});    

impl_assignment_operator!(<S: BaseNum> MulAssign<S> for Bivector3<S> {
    fn mul_assign(&mut self, scalar) { self.x *= scalar; self.y *= scalar; self.z *= scalar;}
});

impl_operator!(<S: BaseNum> Div<S> for Bivector3<S> {
    fn div(vector, scalar) -> Bivector3<S> { Bivector3{x: vector.x / scalar, y: vector.y / scalar, z: vector.z /scalar} }
});
impl_assignment_operator!(<S: BaseNum> DivAssign<S> for Bivector3<S> {
    fn div_assign(&mut self, scalar) { self.x/= scalar; self.y /= scalar; self.z /= scalar;}
});

impl_operator!(<S: BaseNum> BitXor<Vector3<S>> for Vector3<S> {
    fn bitxor(vec3, b) -> Bivector3<S> { 
        Bivector3{
            x: vec3.y * b.z - vec3.z * b.y, 
            y: vec3.z * b.x - vec3.x * b.z, 
            z: vec3.x * b.y - vec3.y * b.x
        }
    }
});

// impl<S:BaseNum> BitXor<Vector3<S>> for Vector3<S> {
//     type Output = Bivector3<S>;
//     fn bitxor(self, b: Vector3<S>) -> Bivector3<S> { 
//         Bivector3{
//             x: self.y * b.z - self.z * b.y, 
//             y: self.z * b.x - self.x * b.z, 
//             z: self.x * b.y - self.y * b.x
//         }
//     }
// }
// impl_operator!(<S: BaseNum> BitXor<Vector3<S> > for Vector3<S> {
//     fn bitxor(a, b) -> Bivector3<S> { 
//         Bivector3{
//             x: a.y * b.z - a.z * b.y, 
//             y: a.z * b.x - a.x * b.z, 
//             z: a.x * b.y - a.y * b.x
//         }
//     }
// });

impl_operator!(<S: BaseNum> BitXor<Point2<S> > for Point2<S> {
    fn bitxor(p, q) -> Bivector3<S> {
        Bivector3{
            x: p.y - q.y,
            y: q.x - p.x,
            z: p.x * q.y - p.y * q.x
        }
    }
});

impl_operator!(<S:BaseFloat> BitXor<Vector2<S> > for Point2<S> {
    fn bitxor(p, v) -> Bivector3<S> {
        Bivector3{
            x: -v.y,
            y: v.x,
            z: p.x * v.y - p.y * v.x
        }
    }
});

impl_operator!(<S:BaseNum> BitXor<Bivector3<S> > for Bivector3<S> {
    fn bitxor(a, b) -> Vector3<S> {
        Vector3{
            x: a.y * b.z - a.z * b.y,
            y: a.z * b.x - a.x * b.z,
            z: a.x * b.y - a.y * b.x
        }
    }
});

impl_operator!(<S:BaseNum> BitXor<Vector3<S> > for Bivector3<S> {
    fn bitxor(a, b) -> S {
        a.x * b.x + a.y + b.y + a.z * b.z
    }
});

impl_operator!(<S:BaseNum> BitXor<Point3<S> > for Bivector3<S> {
    fn bitxor(a, b) -> S {
        a.x * b.x + a.y + b.y + a.z * b.z
    }
});

impl_operator!(<S:BaseNum> BitXor<Bivector3<S> > for Vector3<S> {
    fn bitxor(a, b) -> S {
        a.x * b.x + a.y + b.y + a.z * b.z
    }
});

impl_grassmann_wedge! (<S:BaseNum>, Vector3<S>, Vector3<S>, Bivector3<S>);
impl_grassmann_wedge! (<S:BaseNum>, Point2<S>, Point2<S>, Bivector3<S>);
impl_grassmann_wedge! (<S:BaseFloat>, Point2<S>, Vector2<S>, Bivector3<S>);


impl_grassmann_antiwedge! (<S:BaseNum>, Bivector3<S>, Bivector3<S>, Vector3<S>);
impl_grassmann_antiwedge! (<S:BaseNum>, Bivector3<S>, Vector3<S>, S);
impl_grassmann_antiwedge! (<S:BaseNum>, Vector3<S>, Bivector3<S>, S);

impl<S: Mul<Output = S> + Sub<Output = S> + Copy> Cross<Self, Bivector3<S>> for Point3<S> {
    fn cross(self, q: &Self) -> Bivector3<S> {
        Bivector3::new(self.y * q.z - self.z * q.y, self.z * q.x - self.x * q.z, self.x * q.y - self.y * q.x)
    }
}


impl<S:BaseNum> ProjectTrait<Bivector3<S>> for Vector3<S> {
    fn project(&self, b: &Bivector3<S>) -> Vector3<S> {
        let v = !(*b);

        (v ^ self) ^ b
    }
}
