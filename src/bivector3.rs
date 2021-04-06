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
use point::{Point2};
use vector::{Vector2, Vector3};

#[cfg(feature = "mint")]
use mint;

/// A 3-dimensional bivector having three float components.
/// x,y,z is the value of $e_23,e_31,e_12$ coordinate.
/// This type is marked as `#[repr(C)]`.
#[repr(C)]
#[derive(PartialEq, Eq, Copy, Clone, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Bivector3<S> {
    pub x: S,
    pub y: S,
    pub z: S,
}

macro_rules! impl_scalar_ops {
    ($VectorN:ident<$S:ident>) => {
        impl_operator!(Mul<$VectorN<$S>> for $S {
            fn mul(scalar, vector) -> $VectorN<$S> { $VectorN{x:scalar * vector.x, y:scalar * vector.y, z:scalar * vector.z} }
        });
        impl_operator!(Div<$VectorN<$S>> for $S {
            fn div(scalar, vector) -> $VectorN<$S> { $VectorN{x:scalar / vector.x, y:scalar / vector.y, z:scalar / vector.z} }
        });
        impl_operator!(Rem<$VectorN<$S>> for $S {
            fn rem(scalar, vector) -> $VectorN<$S> { $VectorN{x:scalar % vector.x, y:scalar % vector.y, z:scalar % vector.z} }
        });

    };
}

impl_scalar_ops!(Bivector3<usize>);
impl_scalar_ops!(Bivector3<u8>);
impl_scalar_ops!(Bivector3<u16>);
impl_scalar_ops!(Bivector3<u32>);
impl_scalar_ops!(Bivector3<u64>);
impl_scalar_ops!(Bivector3<isize>);
impl_scalar_ops!(Bivector3<i8>);
impl_scalar_ops!(Bivector3<i16>);
impl_scalar_ops!(Bivector3<i32>);
impl_scalar_ops!(Bivector3<i64>);
impl_scalar_ops!(Bivector3<f32>);
impl_scalar_ops!(Bivector3<f64>);

impl<S: BaseFloat> Bivector3<S> 
{
    /// Construct a new vector, using the provided values.
    #[inline]
    pub fn new(a: S, b: S, d: S) -> Bivector3<S> {
        Bivector3 { x: a, y: b, z: d }
    }

    #[inline]
    pub fn from_points(p: Point2<S>, q: Point2<S>) -> Bivector3<S> {
        Bivector3{ x: p.y - q.y, y: q.x - p.x, z: p.x * q.y - p.y * q.x}
    }

    #[inline]
    pub fn point_vector(p: Point2<S>, v: Vector2<S>) -> Bivector3<S> {
        Bivector3{x: -v.y,y: v.x,z: p.x * v.y - p.y * v.x}
    }

    #[inline]
    pub fn set(&mut self, a: S, b: S, c: S) {
        self.x = a; self.y = b; self.z = c;
    }

    /// Perform the given operation on each field in the vector, returning a new point
    /// constructed from the operations.
    #[inline]
    pub fn map<U, F>(self, mut f: F) -> Bivector3<U>
        where F: FnMut(S) -> U
    {
        Bivector3{ x: f(self.x), y: f(self.y), z: f(self.z) }
    }

    /// Construct a new vector where each component is the result of
    /// applying the given operation to each pair of components of the
    /// given vectors.
    #[inline]
    pub fn zip<S2, S3, F>(self, v2: Bivector3<S2>, mut f: F) -> Bivector3<S3>
        where F: FnMut(S, S2) -> S3
    {
        Bivector3{ x: f(self.x, v2.x), y: f(self.y, v2.y), z: f(self.z, v2.z) }
    }

    #[inline]
    fn dot(self, other: Bivector3<S>) -> S {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    #[inline]
    fn magnitude(self) -> S
    where
        S: Float,
    {
        Float::sqrt(self.magnitude2())
    }

    #[inline]
    fn magnitude2(self) -> S {
        Self::dot(self, self)
    }

    #[inline]
    fn inverse_mag(self) -> S 
    where
        S: Float+One
    {
        S::one() / Self::magnitude(self)
    }

    #[inline]
    fn normalize(self) -> Bivector3<S> {
        self * self.inverse_mag()
    }

}

impl<S:BaseNum> ProjectTrait for Vector3<S> {
    type Other = Bivector3<S>;
    fn project(self, other: Self::Other) -> Self {
        ((!other) ^ self) ^ other
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

impl_assignment_operator!(<S: BaseNum> MulAssign<S> for Bivector3<S> {
    fn mul_assign(&mut self, scalar) { self.x *= scalar; self.y *= scalar; self.z *= scalar;}
});

impl_operator!(<S: BaseNum> Div<S> for Bivector3<S> {
    fn div(vector, scalar) -> Bivector3<S> { Bivector3{x: vector.x / scalar, y: vector.y / scalar, z: vector.z /scalar} }
});
impl_assignment_operator!(<S: BaseNum> DivAssign<S> for Bivector3<S> {
    fn div_assign(&mut self, scalar) { self.x/= scalar; self.y /= scalar; self.z /= scalar;}
});

impl<S:BaseNum> BitXor<Vector3<S>> for Vector3<S> {
    type Output = Bivector3<S>;
    fn bitxor(self, b: Vector3<S>) -> Bivector3<S> { 
        Bivector3{
            x: self.y * b.z - self.z * b.y, 
            y: self.z * b.x - self.x * b.z, 
            z: self.x * b.y - self.y * b.x
        }
    }
}
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

