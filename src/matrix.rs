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
use std::ptr;

use structure::*;

use angle::Rad;
use approx;
use num::{BaseFloat, BaseNum};
use point::{Point2, Point3};
use vector::{Vector2, Vector3, Vector4};
use bivector3::{Bivector3, bivec3};
use trivector4::{Trivector4};


#[cfg(feature = "mint")]
use mint;



macro_rules! mul_v_unrolled {
    ($m: ident, $v: ident, Vector2, Vector2) => {
        Vector2::new(
            $m[0].x * $v.x + $m[1].x * $v.y,
            $m[0].y * $v.x + $m[1].y * $v.y
        )
    };
    ($m: ident, $v: ident, Vector2, Vector3) => {
        Vector2::new(
            $m[0].x * $v.x + $m[1].x * $v.y + $m[2].x * $v.z,
            $m[0].y * $v.x + $m[1].y * $v.y + $m[2].y * $v.z
        )
    };
    ($m: ident, $v: ident, Vector2, Vector4) => {
        Vector2::new(
            $m[0].x * $v.x + $m[1].x * $v.y + $m[2].x * $v.z + $m[3].x * $v.w,
            $m[0].y * $v.x + $m[1].y * $v.y + $m[2].y * $v.z + $m[3].y * $v.w
        )
    };
    ($m: ident, $v: ident, Vector3, Vector2) => {
        Vector3::new(
            $m[0].x * $v.x + $m[1].x * $v.y,
            $m[0].y * $v.x + $m[1].y * $v.y,
            $m[0].z * $v.x + $m[1].z * $v.y
        )
    };
    ($m: ident, $v: ident, Vector3, Vector3) => {
        Vector3::new(
            $m[0].x * $v.x + $m[1].x * $v.y + $m[2].x * $v.z,
            $m[0].y * $v.x + $m[1].y * $v.y + $m[2].y * $v.z,
            $m[0].z * $v.x + $m[1].z * $v.y + $m[2].z * $v.z
        )
    };
    ($m: ident, $v: ident, Vector3, Vector4) => {
        Vector3::new(
            $m[0].x * $v.x + $m[1].x * $v.y + $m[2].x * $v.z + $m[3].x * $v.w,
            $m[0].y * $v.x + $m[1].y * $v.y + $m[2].y * $v.z + $m[3].y * $v.w,
            $m[0].z * $v.x + $m[1].z * $v.y + $m[2].z * $v.z + $m[3].z * $v.w
        )
    };
    ($m: ident, $v: ident, Vector4, Vector2) => {
        Vector4::new(
            $m[0].x * $v.x + $m[1].x * $v.y,
            $m[0].y * $v.x + $m[1].y * $v.y,
            $m[0].z * $v.x + $m[1].z * $v.y,
            $m[0].w * $v.x + $m[1].w * $v.y
        )
    };
    ($m: ident, $v: ident, Vector4, Vector3) => {
        Vector4::new(
            $m[0].x * $v.x + $m[1].x * $v.y + $m[2].x * $v.z,
            $m[0].y * $v.x + $m[1].y * $v.y + $m[2].y * $v.z,
            $m[0].z * $v.x + $m[1].z * $v.y + $m[2].z * $v.z,
            $m[0].w * $v.x + $m[1].w * $v.y + $m[2].w * $v.z
        )
    };
    ($m: ident, $v: ident, Vector4, Vector4) => {
        Vector4::new(
            $m[0].x * $v.x + $m[1].x * $v.y + $m[2].x * $v.z + $m[3].x * $v.w,
            $m[0].y * $v.x + $m[1].y * $v.y + $m[2].y * $v.z + $m[3].y * $v.w,
            $m[0].z * $v.x + $m[1].z * $v.y + $m[2].z * $v.z + $m[3].z * $v.w,
            $m[0].w * $v.x + $m[1].w * $v.y + $m[2].w * $v.z + $m[3].w * $v.w
        )
    };
}

macro_rules! mul_m_unrolled {
    ($lm: ident, $rm: ident, Matrix2) => {
        Matrix2::new(
            $lm.mul_v(&$rm.c0),
            $lm.mul_v(&$rm.c1)
        )
    };
    ($lm: ident, $rm: ident, Matrix3) => {
        Matrix3::new(
            $lm.mul_v(&$rm.c0),
            $lm.mul_v(&$rm.c1),
            $lm.mul_v(&$rm.c2)
        )
    };
    ($lm: ident, $rm: ident, Matrix4) => {
        Matrix4::new(
            $lm.mul_v(&$rm.c0),
            $lm.mul_v(&$rm.c1),
            $lm.mul_v(&$rm.c2),
            $lm.mul_v(&$rm.c3)
        )
    };
}

macro_rules! transpose_unrolled {
    ($m: ident, Vector2, Vector2) => {
        Matrix2::new(
            Vector2::new($m[0][0], $m[1][0]),
            Vector2::new($m[0][1], $m[1][1])
        )
    };
    ($m: ident, Vector2, Vector3) => {
        Matrix2x3::new(
            Vector3::new($m[0][0], $m[1][0], $m[2][0]),
            Vector3::new($m[0][1], $m[1][1], $m[2][1])
        )
    };
    ($m: ident, Vector2, Vector4) => {
        Matrix2x4::new(
            Vector4::new($m[0][0], $m[1][0], $m[2][0], $m[3][0]),
            Vector4::new($m[0][1], $m[1][1], $m[2][1], $m[3][1])
        )
    };
    ($m: ident, Vector3, Vector2) => {
        Matrix3x2::new(
            Vector2::new($m[0][0], $m[1][0]),
            Vector2::new($m[0][1], $m[1][1]),
            Vector2::new($m[0][2], $m[1][2])
        )
    };
    ($m: ident, Vector3, Vector3) => {
        Matrix3::new(
            Vector3::new($m[0][0], $m[1][0], $m[2][0]),
            Vector3::new($m[0][1], $m[1][1], $m[2][1]),
            Vector3::new($m[0][2], $m[1][2], $m[2][2])
        )
    };
    ($m: ident, Vector3, Vector4) => {
        Matrix3x4::new(
            Vector4::new($m[0][0], $m[1][0], $m[2][0], $m[3][0]),
            Vector4::new($m[0][1], $m[1][1], $m[2][1], $m[3][1]),
            Vector4::new($m[0][2], $m[1][2], $m[2][2], $m[3][2])
        )
    };
    ($m: ident, Vector4, Vector2) => {
        Matrix4x2::new(
            Vector2::new($m[0][0], $m[1][0]),
            Vector2::new($m[0][1], $m[1][1]),
            Vector2::new($m[0][2], $m[1][2]),
            Vector2::new($m[0][3], $m[1][3])
        )
    };
    ($m: ident, Vector4, Vector3) => {
        Matrix4x3::new(
            Vector3::new($m[0][0], $m[1][0], $m[2][0]),
            Vector3::new($m[0][1], $m[1][1], $m[2][1]),
            Vector3::new($m[0][2], $m[1][2], $m[2][2]),
            Vector3::new($m[0][3], $m[1][3], $m[2][3])
        )
    };
    ($m: ident, Vector4, Vector4) => {
        Matrix4::new(
            Vector4::new($m[0][0], $m[1][0], $m[2][0], $m[3][0]),
            Vector4::new($m[0][1], $m[1][1], $m[2][1], $m[3][1]),
            Vector4::new($m[0][2], $m[1][2], $m[2][2], $m[3][2]),
            Vector4::new($m[0][3], $m[1][3], $m[2][3], $m[3][3])
        )
    };
}

macro_rules! def_matrix {
    ($({
        $t: ident,          // type to be defined,
        $ct: ident,         // type of column vector,
        $($field: ident), + // name of columen vectors.
    }), +) => {
        $(
            #[repr(C)]
            #[derive(Copy, Clone, PartialEq, Debug)]
            pub struct $t<S: BaseFloat> {
                $(pub $field: $ct<S>), +
            }
        )+
    }
}

def_matrix! {
    { Matrix2,   Vector2, c0, c1 },

    { Matrix3,   Vector3, c0, c1, c2 },

    { Matrix4,   Vector4, c0, c1, c2, c3 }
}



macro_rules! impl_matrix {
    ($({
        $t: ident,          // type to impl (e.g., Matrix3),
        $ct: ident,         // type of column vector (e.g., Vec2),
        $rt: ident,         // type of row vector,
        $tr: ident,         // type of transpose matrix,
        $om: ident,         // the product of multiplying transpose matrix,
        $cn: expr,          // number of columns, i.e., the dimension of $rt,
        $($field: ident), + // fields for repeating reference columns,
    }), +) => {
        $(
            impl<S: BaseFloat> $t<S> {
                #[inline(always)]
                pub const fn new($($field: $ct<S>), +) -> $t<S> {
                    $t { $($field: $field), + }
                }
                #[inline(always)]
                pub fn from_array(ary: &[$ct<S>; $cn]) -> &$t<S> {
                    let m: &Self = unsafe { mem::transmute(ary) };
                    m
                }
                #[inline(always)]
                pub fn from_array_mut(ary: &mut [$ct<S>; $cn]) -> &mut $t<S> {
                    let m: &mut Self = unsafe { mem::transmute(ary) };
                    m
                }
                #[inline(always)]
                pub fn as_array(&self) -> &[$ct<S>; $cn] {
                    let ary: &[$ct<S>; $cn] = unsafe { mem::transmute(self) };
                    ary
                }
                #[inline(always)]
                pub fn as_array_mut(&mut self) -> &mut [$ct<S>; $cn] {
                    let ary: &mut[$ct<S>; $cn] = unsafe { mem::transmute(self) };
                    ary
                }
                #[inline(always)]
                pub fn add_s(&self, rhs: S) -> $t<S> {
                    $t::new($(self.$field + rhs), +)
                }
                #[inline(always)]
                pub fn add_m(&self, rhs: &$t<S>) -> $t<S> {
                    $t::new($(self.$field + rhs.$field), +)
                }
                #[inline(always)]
                pub fn sub_s(&self, rhs: S) -> $t<S> {
                    $t::new($(self.$field - rhs), +)
                }
                #[inline(always)]
                pub fn sub_m(&self, rhs: &$t<S>) -> $t<S> {
                    $t::new($(self.$field - rhs.$field), +)
                }
                #[inline(always)]
                pub fn div_m(&self, rhs: &$t<S>) -> $t<S> {
                    $t::new($(self.$field / rhs.$field), +)
                }
                #[inline(always)]
                pub fn div_s(&self, rhs: S) -> $t<S> {
                    $t::new($(self.$field / rhs), +)
                }
                #[inline(always)]
                pub fn rem_m(&self, rhs: &$t<S>) -> $t<S> {
                    $t::new($(self.$field % rhs.$field), +)
                }
                #[inline(always)]
                pub fn rem_s(&self, rhs: S) -> $t<S> {
                    $t::new($(self.$field % rhs), +)
                }
                #[inline(always)]
                pub fn mul_s(&self, rhs: S) -> $t<S> {
                    $t::new($(self.$field * rhs), +)
                }
                #[inline(always)]
                pub fn mul_v(&self, rhs: &$rt<S>) -> $ct<S> {
                    mul_v_unrolled! { self, rhs, $ct, $rt }
                }
                #[inline(always)]
                pub fn mul_m(&self, rhs: &$tr<S>) -> $om<S> {
                    mul_m_unrolled! { self, rhs, $om }
                }
                #[inline(always)]
                pub fn neg_m(&self) -> $t<S> {
                    $t::new($(self.$field.neg()), +)
                }
            }
            impl<S: BaseFloat> Index<usize> for $t<S> {
                type Output = $ct<S>;
                #[inline(always)]
                fn index<'a>(&'a self, i: usize) -> &'a $ct<S> {
                    self.as_array().index(i)
                }
            }
            impl<S: BaseFloat> IndexMut<usize> for $t<S> {
                #[inline(always)]
                fn index_mut<'a>(&'a mut self, i: usize) -> &'a mut $ct<S> {
                    self.as_array_mut().index_mut(i)
                }
            }
 
            impl<S: BaseFloat> Add<S> for $t<S> {
                type Output = $t<S>;
                #[inline(always)]
                fn add(self, rhs: S) -> $t<S> {
                    self.add_s(rhs)
                }
            }

            impl<S: BaseFloat> approx::AbsDiffEq for $t<S> {
                type Epsilon = S::Epsilon;
    
                #[inline]
                fn default_epsilon() -> S::Epsilon {
                    S::default_epsilon()
                }
    
                #[inline]
                fn abs_diff_eq(&self, other: &Self, epsilon: S::Epsilon) -> bool {
                    $(self.$field.abs_diff_eq(&other.$field, epsilon))&&+
                }
            }
    
            impl<S: BaseFloat> approx::RelativeEq for $t<S> {
                #[inline]
                fn default_max_relative() -> S::Epsilon {
                    S::default_max_relative()
                }
    
                #[inline]
                fn relative_eq(&self, other: &Self, epsilon: S::Epsilon, max_relative: S::Epsilon) -> bool {
                    $(self.$field.relative_eq(&other.$field, epsilon, max_relative))&&+
                }
            }
    
            impl<S: BaseFloat> approx::UlpsEq for $t<S> {
                #[inline]
                fn default_max_ulps() -> u32 {
                    S::default_max_ulps()
                }
    
                #[inline]
                fn ulps_eq(&self, other: &Self, epsilon: S::Epsilon, max_ulps: u32) -> bool {
                    $(self.$field.ulps_eq(&other.$field, epsilon, max_ulps))&&+
                }
            }

            #[cfg(feature = "rand")]
            impl<S:BaseFloat> Distribution<$t<S>> for Standard
            where 
                Standard: Distribution<$ct<S>>,
            {
                #[inline]
                fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> $t<S> {
                    $t { $($field: self.sample(rng)),+ }
                    //$t { c0: self.sample(rng), c1: self.sample(rng) }
                }
            }

            impl<S: BaseFloat> Add<$t<S>> for $t<S> {
                type Output = $t<S>;
                #[inline(always)]
                fn add(self, rhs: $t<S>) -> $t<S> {
                    self.add_m(&rhs)
                }
            }
            impl<S: BaseFloat> Sub<S> for $t<S> {
                type Output = $t<S>;
                #[inline(always)]
                fn sub(self, rhs: S) -> $t<S> {
                    self.sub_s(rhs)
                }
            }
            impl<S: BaseFloat> Sub<$t<S>> for $t<S> {
                type Output = $t<S>;
                #[inline(always)]
                fn sub(self, rhs: $t<S>) -> $t<S> {
                    self.sub_m(&rhs)
                }
            }
            impl<S: BaseFloat> Div<S> for $t<S> {
                type Output = $t<S>;
                #[inline(always)]
                fn div(self, rhs: S) -> $t<S> {
                    self.div_s(rhs)
                }
            }
            impl<S: BaseFloat> Div<$t<S>> for $t<S> {
                type Output = $t<S>;
                #[inline(always)]
                fn div(self, rhs: $t<S>) -> $t<S> {
                    self.div_m(&rhs)
                }
            }
            impl<S: BaseFloat> Rem<S> for $t<S> {
                type Output = $t<S>;
                #[inline(always)]
                fn rem(self, rhs: S) -> $t<S> {
                    self.rem_s(rhs)
                }
            }
            impl<S: BaseFloat> Rem<$t<S>> for $t<S> {
                type Output = $t<S>;
                #[inline(always)]
                fn rem(self, rhs: $t<S>) -> $t<S> {
                    self.rem_m(&rhs)
                }
            }
            impl<S: BaseFloat> Neg for $t<S> {
                type Output = $t<S>;
                #[inline(always)]
                fn neg(self) -> $t<S> {
                    self.neg_m()
                }
            }
            impl<S: BaseFloat> Mul<S> for $t<S> {
                type Output = $t<S>;
                #[inline(always)]
                fn mul(self, rhs: S) -> $t<S> {
                    self.mul_s(rhs)
                }
            }
            impl<S: BaseFloat> Mul<$rt<S>> for $t<S> {
                type Output = $ct<S>;
                #[inline(always)]
                fn mul(self, rhs: $rt<S>) -> $ct<S> {
                    self.mul_v(&rhs)
                }
            }
            impl<S: BaseFloat> Mul<$tr<S>> for $t<S> {
                type Output = $om<S>;
                #[inline(always)]
                fn mul(self, rhs: $tr<S>) -> $om<S> {
                    self.mul_m(&rhs)
                }
            }

            impl_assignment_operator!(<S: BaseFloat> MulAssign<$t<S>> for $t<S> {
                fn mul_assign(&mut self, rhs) { self.mul_m(&rhs); }
            });
            impl_assignment_operator!(<S: BaseFloat> MulAssign<S> for $t<S> {
                fn mul_assign(&mut self, scalar) { $(self.$field *= scalar);+ }
            });
            impl_assignment_operator!(<S: BaseFloat> DivAssign<S> for $t<S> {
                fn div_assign(&mut self, scalar) { $(self.$field /= scalar);+ }
            });

            impl<S: BaseFloat> Zero for $t<S> {
                #[inline(always)]
                fn zero() -> $t<S> {
                    $t { $($field: $ct::<S>::zero()), + }
                }
                #[inline(always)]
                fn is_zero(&self) -> bool {
                    $(self.$field.is_zero()) && +
                }
            }
            impl<S: BaseFloat> One for $t<S> {
                #[inline]
                fn one() -> $t<S> {
                    $t::from_value(S::one())
                }
            }

            impl<S: BaseFloat> GenMat<S, $ct<S>> for $t<S> {
                type R = $rt<S>;
                type Transpose = $tr<S>;
                #[inline]
                fn transpose(&self) -> $tr<S> {
                    transpose_unrolled!(self, $ct, $rt)
                }
                #[inline(always)]
                fn mul_c(&self, rhs: &$t<S>) -> $t<S> {
                    $t::new($(self.$field * rhs.$field), +)
                }
            }


            impl_scalar_ops!($t<f32> { $($field),+ });
            impl_scalar_ops!($t<f64> { $($field),+ });
    
            impl<S: BaseFloat> iter::Sum<$t<S>> for $t<S> {
                #[inline]
                fn sum<I: Iterator<Item=$t<S>>>(iter: I) -> $t<S> {
                    iter.fold($t::zero(), Add::add)
                }
            }
    
            impl<'a, S: 'a + BaseFloat> iter::Sum<&'a $t<S>> for $t<S> {
                #[inline]
                fn sum<I: Iterator<Item=&'a $t<S>>>(iter: I) -> $t<S> {
                    iter.fold($t::zero(), Add::add)
                }
            }
    
            impl<S: BaseFloat> iter::Product for $t<S> {
                #[inline]
                fn product<I: Iterator<Item=$t<S>>>(iter: I) -> $t<S> {
                    iter.fold($t::identity(), Mul::mul)
                }
            }
    
            impl<'a, S: 'a + BaseFloat> iter::Product<&'a $t<S>> for $t<S> {
                #[inline]
                fn product<I: Iterator<Item=&'a $t<S>>>(iter: I) -> $t<S> {
                    iter.fold($t::identity(), Mul::mul)
                }
            }
    
    
            impl<S: BaseFloat + NumCast + Copy> $t<S> {
                /// Component-wise casting to another type
                #[inline]
                pub fn cast<T: BaseFloat + NumCast>(&self) -> Option<$t<T>> {
                    $(
                        let $field = match self.$field.cast() {
                            Some(field) => field,
                            None => return None
                        };
                    )+
                    Some($t { $($field),+ })
                }
            }

            
            impl<S: BaseFloat> VectorSpace for $t<S> {
                type Scalar = S;
            }

       )+
    }
}

macro_rules! impl_scalar_ops {
    ($t:ident<$S:ident> { $($field:ident),+ }) => {
        impl_operator!(Mul<$t<$S>> for $S {
            fn mul(scalar, matrix) -> $t<$S> { $t { $($field: scalar * matrix.$field),+ } }
        });
        impl_operator!(Div<$t<$S>> for $S {
            fn div(scalar, matrix) -> $t<$S> { $t { $($field: scalar / matrix.$field),+ } }
        });
        impl_operator!(Rem<$t<$S>> for $S {
            fn rem(scalar, matrix) -> $t<$S> { $t { $($field: scalar % matrix.$field),+ } }
        });
    };
}

impl_matrix! {
    { Matrix2,   Vector2, Vector2, Matrix2,   Matrix2, 2, c0, c1 },

    { Matrix3,   Vector3, Vector3, Matrix3,   Matrix3, 3, c0, c1, c2 },

    { Matrix4,   Vector4, Vector4, Matrix4,   Matrix4, 4, c0, c1, c2, c3 }
}

impl<S: BaseFloat> Matrix2<S> {
    /// Create a new matrix, providing values for each index.
    #[inline]
    pub const fn new_fileds(c0r0: S, c0r1: S, c1r0: S, c1r1: S) -> Matrix2<S> {
        Matrix2::from_cols(Vector2::new(c0r0, c0r1), Vector2::new(c1r0, c1r1))
    }

    /// Create a new matrix, providing columns.
    #[inline]
    pub const fn from_cols(c0: Vector2<S>, c1: Vector2<S>) -> Matrix2<S> {
        Matrix2 { c0: c0, c1: c1 }
    }
}

impl<S: BaseFloat> Matrix3<S> {
    /// Create a new matrix, providing values for each index.
    #[inline]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    pub const fn new_fileds(
        c0r0:S, c0r1:S, c0r2:S,
        c1r0:S, c1r1:S, c1r2:S,
        c2r0:S, c2r1:S, c2r2:S,
    ) -> Matrix3<S> {
        Matrix3::from_cols(
            Vector3::new(c0r0, c0r1, c0r2),
            Vector3::new(c1r0, c1r1, c1r2),
            Vector3::new(c2r0, c2r1, c2r2),
        )
    }

    /// Create a new matrix, providing columns.
    #[inline]
    pub const fn from_cols(c0: Vector3<S>, c1: Vector3<S>, c2: Vector3<S>) -> Matrix3<S> {
        Matrix3 {
            c0: c0,
            c1: c1,
            c2: c2,
        }
    }

    /// Are all entries in the matrix finite.
    pub fn is_finite(&self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }
}

impl<S: BaseFloat> Matrix4<S> {
    /// Create a new matrix, providing values for each index.
    #[inline]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    pub const fn new_fields(
        c0r0: S, c0r1: S, c0r2: S, c0r3: S,
        c1r0: S, c1r1: S, c1r2: S, c1r3: S,
        c2r0: S, c2r1: S, c2r2: S, c2r3: S,
        c3r0: S, c3r1: S, c3r2: S, c3r3: S,
    ) -> Matrix4<S>  {
        Matrix4::from_cols(
            Vector4::new(c0r0, c0r1, c0r2, c0r3),
            Vector4::new(c1r0, c1r1, c1r2, c1r3),
            Vector4::new(c2r0, c2r1, c2r2, c2r3),
            Vector4::new(c3r0, c3r1, c3r2, c3r3),
        )
    }

    /// Create a new matrix, providing columns.
    #[inline]
    pub const fn from_cols(
        c0: Vector4<S>,
        c1: Vector4<S>,
        c2: Vector4<S>,
        c3: Vector4<S>,
    ) -> Matrix4<S> {
        Matrix4 {
            c0: c0,
            c1: c1,
            c2: c2,
            c3: c3,
        }
    }
}

macro_rules! def_alias(
    (
        $({
            $a: ident,          // type alias (e.g., Mat2 for Matrix2<f32>),
            $t: ident,          // type to be aliased,
            $et: ty             // element type,
        }), +
    ) => {
        $(
            pub type $a = $t<$et>;
        )+
    }
);

def_alias! {
    { Mat2,   Matrix2,   f32 },
    { Mat3,   Matrix3,   f32 },
    { Mat4,   Matrix4,   f32 },
    { DMat2,   Matrix2,   f64 },
    { DMat3,   Matrix3,   f64 },
    { DMat4,   Matrix4,   f64 }
}


impl<S: BaseFloat> Matrix for Matrix2<S> {
    type Column = Vector2<S>;
    type Row = Vector2<S>;
    type Transpose = Matrix2<S>;

    #[inline]
    fn row(&self, r: usize) -> Vector2<S> {
        Vector2::new(self[0][r], self[1][r])
    }

    #[inline]
    fn swap_rows(&mut self, a: usize, b: usize) {
        self[0].swap_elements(a, b);
        self[1].swap_elements(a, b);
    }

    #[inline]
    fn swap_columns(&mut self, a: usize, b: usize) {
        unsafe { ptr::swap(&mut self[a], &mut self[b]) };
    }

    #[inline]
    fn swap_elements(&mut self, a: (usize, usize), b: (usize, usize)) {
        let (ac, ar) = a;
        let (bc, br) = b;
        unsafe { ptr::swap(&mut self[ac][ar], &mut self[bc][br]) };
    }

    fn transpose(&self) -> Matrix2<S> {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        Matrix2::new(
            self[0][0], self[1][0],
            self[0][1], self[1][1],
        )
    }
}

impl<S: BaseFloat> SquareMatrix for Matrix2<S> {
    type ColumnRow = Vector2<S>;

    #[inline]
    fn from_value(value: S) -> Matrix2<S> {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        Matrix2::new(
            value, S::zero(),
            S::zero(), value,
        )
    }

    #[inline]
    fn from_diagonal(value: Vector2<S>) -> Matrix2<S> {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        Matrix2::new(
            value.x, S::zero(),
            S::zero(), value.y,
        )
    }

    #[inline]
    fn transpose_self(&mut self) {
        self.swap_elements((0, 1), (1, 0));
    }

    #[inline]
    fn determinant(&self) -> S {
        self[0][0] * self[1][1] - self[1][0] * self[0][1]
    }

    #[inline]
    fn diagonal(&self) -> Vector2<S> {
        Vector2::new(self[0][0], self[1][1])
    }

    #[inline]
    fn invert(&self) -> Option<Matrix2<S>> {
        let det = self.determinant();
        if det == S::zero() {
            None
        } else {
            #[cfg_attr(rustfmt, rustfmt_skip)]
            Some(Matrix2::from_fields(
                self[1][1] / det, -self[0][1] / det,
                -self[1][0] / det, self[0][0] / det,
            ))
        }
    }

    #[inline]
    fn is_diagonal(&self) -> bool {
        ulps_eq!(self[0][1], &S::zero()) && ulps_eq!(self[1][0], &S::zero())
    }

    #[inline]
    fn is_symmetric(&self) -> bool {
        ulps_eq!(self[0][1], &self[1][0]) && ulps_eq!(self[1][0], &self[0][1])
    }
}

impl<S: BaseFloat> Matrix for Matrix3<S> {
    type Column = Vector3<S>;
    type Row = Vector3<S>;
    type Transpose = Matrix3<S>;

    #[inline]
    fn row(&self, r: usize) -> Vector3<S> {
        Vector3::new(self[0][r], self[1][r], self[2][r])
    }

    #[inline]
    fn swap_rows(&mut self, a: usize, b: usize) {
        self[0].swap_elements(a, b);
        self[1].swap_elements(a, b);
        self[2].swap_elements(a, b);
    }

    #[inline]
    fn swap_columns(&mut self, a: usize, b: usize) {
        unsafe { ptr::swap(&mut self[a], &mut self[b]) };
    }

    #[inline]
    fn swap_elements(&mut self, a: (usize, usize), b: (usize, usize)) {
        let (ac, ar) = a;
        let (bc, br) = b;
        unsafe { ptr::swap(&mut self[ac][ar], &mut self[bc][br]) };
    }

    fn transpose(&self) -> Matrix3<S> {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        Matrix3::new(
            self[0][0], self[1][0], self[2][0],
            self[0][1], self[1][1], self[2][1],
            self[0][2], self[1][2], self[2][2],
        )
    }
}

impl<S: BaseFloat> SquareMatrix for Matrix3<S> {
    type ColumnRow = Vector3<S>;

    #[inline]
    fn from_value(value: S) -> Matrix3<S> {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        Matrix3::new(
            value, S::zero(), S::zero(),
            S::zero(), value, S::zero(),
            S::zero(), S::zero(), value,
        )
    }

    #[inline]
    fn from_diagonal(value: Vector3<S>) -> Matrix3<S> {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        Matrix3::new(
            value.x, S::zero(), S::zero(),
            S::zero(), value.y, S::zero(),
            S::zero(), S::zero(), value.z,
        )
    }

    #[inline]
    fn transpose_self(&mut self) {
        self.swap_elements((0, 1), (1, 0));
        self.swap_elements((0, 2), (2, 0));
        self.swap_elements((1, 2), (2, 1));
    }

    fn determinant(&self) -> S {
        self[0][0] * (self[1][1] * self[2][2] - self[2][1] * self[1][2])
            - self[1][0] * (self[0][1] * self[2][2] - self[2][1] * self[0][2])
            + self[2][0] * (self[0][1] * self[1][2] - self[1][1] * self[0][2])
    }

    #[inline]
    fn diagonal(&self) -> Vector3<S> {
        Vector3::new(self[0][0], self[1][1], self[2][2])
    }

    fn invert(&self) -> Option<Matrix3<S>> {
        let det = self.determinant();
        if det == S::zero() {
            None
        } else {
            Some(
                Matrix3::from_cols(
                    self[1].cross(self[2]) / det,
                    self[2].cross(self[0]) / det,
                    self[0].cross(self[1]) / det,
                )
                .transpose(),
            )
        }
    }

    fn is_diagonal(&self) -> bool {
        ulps_eq!(self[0][1], &S::zero())
            && ulps_eq!(self[0][2], &S::zero())
            && ulps_eq!(self[1][0], &S::zero())
            && ulps_eq!(self[1][2], &S::zero())
            && ulps_eq!(self[2][0], &S::zero())
            && ulps_eq!(self[2][1], &S::zero())
    }

    fn is_symmetric(&self) -> bool {
        ulps_eq!(self[0][1], &self[1][0])
            && ulps_eq!(self[0][2], &self[2][0])
            && ulps_eq!(self[1][0], &self[0][1])
            && ulps_eq!(self[1][2], &self[2][1])
            && ulps_eq!(self[2][0], &self[0][2])
            && ulps_eq!(self[2][1], &self[1][2])
    }
}

impl<S: BaseFloat> Matrix for Matrix4<S> {
    type Column = Vector4<S>;
    type Row = Vector4<S>;
    type Transpose = Matrix4<S>;

    #[inline]
    fn row(&self, r: usize) -> Vector4<S> {
        Vector4::new(self[0][r], self[1][r], self[2][r], self[3][r])
    }

    #[inline]
    fn swap_rows(&mut self, a: usize, b: usize) {
        self[0].swap_elements(a, b);
        self[1].swap_elements(a, b);
        self[2].swap_elements(a, b);
        self[3].swap_elements(a, b);
    }

    #[inline]
    fn swap_columns(&mut self, a: usize, b: usize) {
        unsafe { ptr::swap(&mut self[a], &mut self[b]) };
    }

    #[inline]
    fn swap_elements(&mut self, a: (usize, usize), b: (usize, usize)) {
        let (ac, ar) = a;
        let (bc, br) = b;
        unsafe { ptr::swap(&mut self[ac][ar], &mut self[bc][br]) };
    }

    fn transpose(&self) -> Matrix4<S> {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        Matrix4::new(
            self[0][0], self[1][0], self[2][0], self[3][0],
            self[0][1], self[1][1], self[2][1], self[3][1],
            self[0][2], self[1][2], self[2][2], self[3][2],
            self[0][3], self[1][3], self[2][3], self[3][3],
        )
    }
}

impl<S: BaseFloat> SquareMatrix for Matrix4<S> {
    type ColumnRow = Vector4<S>;

    #[inline]
    fn from_value(value: S) -> Matrix4<S> {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        Matrix4::new(
            value, S::zero(), S::zero(), S::zero(),
            S::zero(), value, S::zero(), S::zero(),
            S::zero(), S::zero(), value, S::zero(),
            S::zero(), S::zero(), S::zero(), value,
        )
    }

    #[inline]
    fn from_diagonal(value: Vector4<S>) -> Matrix4<S> {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        Matrix4::new(
            value.x, S::zero(), S::zero(), S::zero(),
            S::zero(), value.y, S::zero(), S::zero(),
            S::zero(), S::zero(), value.z, S::zero(),
            S::zero(), S::zero(), S::zero(), value.w,
        )
    }

    fn transpose_self(&mut self) {
        self.swap_elements((0, 1), (1, 0));
        self.swap_elements((0, 2), (2, 0));
        self.swap_elements((0, 3), (3, 0));
        self.swap_elements((1, 2), (2, 1));
        self.swap_elements((1, 3), (3, 1));
        self.swap_elements((2, 3), (3, 2));
    }

    fn determinant(&self) -> S {
        let tmp = unsafe { det_sub_proc_unsafe(self, 1, 2, 3) };
        tmp.dot(Vector4::new(self[0][0], self[1][0], self[2][0], self[3][0]))
    }

    #[inline]
    fn diagonal(&self) -> Vector4<S> {
        Vector4::new(self[0][0], self[1][1], self[2][2], self[3][3])
    }

    // The new implementation results in negative optimization when used
    // without SIMD. so we opt them in with configuration.
    // A better option would be using specialization. But currently somewhat
    // specialization is too buggy, and it won't apply here. I'm getting
    // weird error msgs. Help wanted.
    #[cfg(not(feature = "simd"))]
    fn invert(&self) -> Option<Matrix4<S>> {
        let det = self.determinant();
        if det == S::zero() {
            None
        } else {
            let inv_det = S::one() / det;
            let t = self.transpose();
            let cf = |i, j| {
                let mat = match i {
                    0 => {
                        Matrix3::from_cols(t.y.truncate_n(j), t.z.truncate_n(j), t.w.truncate_n(j))
                    }
                    1 => {
                        Matrix3::from_cols(t.x.truncate_n(j), t.z.truncate_n(j), t.w.truncate_n(j))
                    }
                    2 => {
                        Matrix3::from_cols(t.x.truncate_n(j), t.y.truncate_n(j), t.w.truncate_n(j))
                    }
                    3 => {
                        Matrix3::from_cols(t.x.truncate_n(j), t.y.truncate_n(j), t.z.truncate_n(j))
                    }
                    _ => panic!("out of range"),
                };
                let sign = if (i + j) & 1 == 1 {
                    -S::one()
                } else {
                    S::one()
                };
                mat.determinant() * sign * inv_det
            };

            #[cfg_attr(rustfmt, rustfmt_skip)]
            Some(Matrix4::from_fields(
                cf(0, 0), cf(0, 1), cf(0, 2), cf(0, 3),
                cf(1, 0), cf(1, 1), cf(1, 2), cf(1, 3),
                cf(2, 0), cf(2, 1), cf(2, 2), cf(2, 3),
                cf(3, 0), cf(3, 1), cf(3, 2), cf(3, 3),
            ))
        }
    }
    #[cfg(feature = "simd")]
    fn invert(&self) -> Option<Matrix4<S>> {
        let tmp0 = unsafe { det_sub_proc_unsafe(self, 1, 2, 3) };
        let det = tmp0.dot(Vector4::new(self[0][0], self[1][0], self[2][0], self[3][0]));

        if det == S::zero() {
            None
        } else {
            let inv_det = S::one() / det;
            let tmp0 = tmp0 * inv_det;
            let tmp1 = unsafe { det_sub_proc_unsafe(self, 0, 3, 2) * inv_det };
            let tmp2 = unsafe { det_sub_proc_unsafe(self, 0, 1, 3) * inv_det };
            let tmp3 = unsafe { det_sub_proc_unsafe(self, 0, 2, 1) * inv_det };
            Some(Matrix4::from_cols(tmp0, tmp1, tmp2, tmp3))
        }
    }

    fn is_diagonal(&self) -> bool {
        ulps_eq!(self[0][1], &S::zero())
            && ulps_eq!(self[0][2], &S::zero())
            && ulps_eq!(self[0][3], &S::zero())
            && ulps_eq!(self[1][0], &S::zero())
            && ulps_eq!(self[1][2], &S::zero())
            && ulps_eq!(self[1][3], &S::zero())
            && ulps_eq!(self[2][0], &S::zero())
            && ulps_eq!(self[2][1], &S::zero())
            && ulps_eq!(self[2][3], &S::zero())
            && ulps_eq!(self[3][0], &S::zero())
            && ulps_eq!(self[3][1], &S::zero())
            && ulps_eq!(self[3][2], &S::zero())
    }

    fn is_symmetric(&self) -> bool {
        ulps_eq!(self[0][1], &self[1][0])
            && ulps_eq!(self[0][2], &self[2][0])
            && ulps_eq!(self[0][3], &self[3][0])
            && ulps_eq!(self[1][0], &self[0][1])
            && ulps_eq!(self[1][2], &self[2][1])
            && ulps_eq!(self[1][3], &self[3][1])
            && ulps_eq!(self[2][0], &self[0][2])
            && ulps_eq!(self[2][1], &self[1][2])
            && ulps_eq!(self[2][3], &self[3][2])
            && ulps_eq!(self[3][0], &self[0][3])
            && ulps_eq!(self[3][1], &self[1][3])
            && ulps_eq!(self[3][2], &self[2][3])
    }
}



macro_rules! fixed_array_conversions {
    ($MatrixN:ident { $($field:ident : $index:expr),+ }, $n:expr) => {
        impl<S: BaseFloat> Into<[[S; $n]; $n]> for $MatrixN<S> {
            #[inline]
            fn into(self) -> [[S; $n]; $n] {
                match self { $MatrixN { $($field),+ } => [$($field.into()),+] }
            }
        }

        impl<S: BaseFloat> AsRef<[[S; $n]; $n]> for $MatrixN<S> {
            #[inline]
            fn as_ref(&self) -> &[[S; $n]; $n] {
                unsafe { mem::transmute(self) }
            }
        }

        impl<S: BaseFloat> AsMut<[[S; $n]; $n]> for $MatrixN<S> {
            #[inline]
            fn as_mut(&mut self) -> &mut [[S; $n]; $n] {
                unsafe { mem::transmute(self) }
            }
        }

        impl<S: BaseFloat + Copy> From<[[S; $n]; $n]> for $MatrixN<S> {
            #[inline]
            fn from(m: [[S; $n]; $n]) -> $MatrixN<S> {
                // We need to use a copy here because we can't pattern match on arrays yet
                $MatrixN { $($field: From::from(m[$index])),+ }
            }
        }

        impl<'a, S: BaseFloat> From<&'a [[S; $n]; $n]> for &'a $MatrixN<S> {
            #[inline]
            fn from(m: &'a [[S; $n]; $n]) -> &'a $MatrixN<S> {
                unsafe { mem::transmute(m) }
            }
        }

        impl<'a, S: BaseFloat> From<&'a mut [[S; $n]; $n]> for &'a mut $MatrixN<S> {
            #[inline]
            fn from(m: &'a mut [[S; $n]; $n]) -> &'a mut $MatrixN<S> {
                unsafe { mem::transmute(m) }
            }
        }

        // impl<S> Into<[S; ($n * $n)]> for $MatrixN<S> {
        //     #[inline]
        //     fn into(self) -> [[S; $n]; $n] {
        //         // TODO: Not sure how to implement this...
        //         unimplemented!()
        //     }
        // }

        impl<S: BaseFloat> AsRef<[S; ($n * $n)]> for $MatrixN<S> {
            #[inline]
            fn as_ref(&self) -> &[S; ($n * $n)] {
                unsafe { mem::transmute(self) }
            }
        }

        impl<S: BaseFloat> AsMut<[S; ($n * $n)]> for $MatrixN<S> {
            #[inline]
            fn as_mut(&mut self) -> &mut [S; ($n * $n)] {
                unsafe { mem::transmute(self) }
            }
        }

        // impl<S> From<[S; ($n * $n)]> for $MatrixN<S> {
        //     #[inline]
        //     fn from(m: [S; ($n * $n)]) -> $MatrixN<S> {
        //         // TODO: Not sure how to implement this...
        //         unimplemented!()
        //     }
        // }

        impl<'a, S: BaseFloat> From<&'a [S; ($n * $n)]> for &'a $MatrixN<S> {
            #[inline]
            fn from(m: &'a [S; ($n * $n)]) -> &'a $MatrixN<S> {
                unsafe { mem::transmute(m) }
            }
        }

        impl<'a, S: BaseFloat> From<&'a mut [S; ($n * $n)]> for &'a mut $MatrixN<S> {
            #[inline]
            fn from(m: &'a mut [S; ($n * $n)]) -> &'a mut $MatrixN<S> {
                unsafe { mem::transmute(m) }
            }
        }
    }
}

fixed_array_conversions!(Matrix2 { c0:0, c1:1 }, 2);
fixed_array_conversions!(Matrix3 { c0:0, c1:1, c2:2 }, 3);
fixed_array_conversions!(Matrix4 { c0:0, c1:1, c2:2, c3:3 }, 4);

#[cfg(feature = "mint")]
macro_rules! mint_conversions {
    ($MatrixN:ident { $($field:ident),+ },{ $($field2:ident),+ }, $MintN:ident) => {
        impl<S: BaseFloat + Clone> Into<mint::$MintN<S>> for $MatrixN<S> {
            #[inline]
            fn into(self) -> mint::$MintN<S> {
                mint::$MintN { $($field2: self.$field.into()),+ }
            }
        }

        impl<S: BaseFloat> From<mint::$MintN<S>> for $MatrixN<S> {
            #[inline]
            fn from(m: mint::$MintN<S>) -> Self {
                $MatrixN { $($field: m.$field2.into()),+ }
            }
        }

    }
}

#[cfg(feature = "mint")]
mint_conversions!(Matrix2 { c0, c1 }, {x, y }, ColumnMatrix2);
#[cfg(feature = "mint")]
mint_conversions!(Matrix3 { c0, c1, c2 },{x, y , z},  ColumnMatrix3);
#[cfg(feature = "mint")]
mint_conversions!(Matrix4 { c0, c1, c2, c3 },{x, y , z, w},  ColumnMatrix4);

impl<S: BaseFloat> From<Matrix2<S>> for Matrix3<S> {
    /// Clone the elements of a 2-dimensional matrix into the top-left corner
    /// of a 3-dimensional identity matrix.
    fn from(m: Matrix2<S>) -> Matrix3<S> {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        Matrix3::new_fileds(
            m[0][0], m[0][1], S::zero(),
            m[1][0], m[1][1], S::zero(),
            S::zero(), S::zero(), S::one(),
        )
    }
}

impl<S: BaseFloat> From<Matrix2<S>> for Matrix4<S> {
    /// Clone the elements of a 2-dimensional matrix into the top-left corner
    /// of a 4-dimensional identity matrix.
    fn from(m: Matrix2<S>) -> Matrix4<S> {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        Matrix4::new_fileds(
            m[0][0], m[0][1], S::zero(), S::zero(),
            m[1][0], m[1][1], S::zero(), S::zero(),
            S::zero(), S::zero(), S::one(), S::zero(),
            S::zero(), S::zero(), S::zero(), S::one(),
        )
    }
}

impl<S: BaseFloat> From<Matrix3<S>> for Matrix4<S> {
    /// Clone the elements of a 3-dimensional matrix into the top-left corner
    /// of a 4-dimensional identity matrix.
    fn from(m: Matrix3<S>) -> Matrix4<S> {
        #[cfg_attr(rustfmt, rustfmt_skip)]
        Matrix4::new_fileds(
            m[0][0], m[0][1], m[0][2], S::zero(),
            m[1][0], m[1][1], m[1][2], S::zero(),
            m[2][0], m[2][1], m[2][2], S::zero(),
            S::zero(), S::zero(), S::zero(), S::one(),
        )
    }
}


// Sub procedure for SIMD when dealing with determinant and inversion
#[inline]
unsafe fn det_sub_proc_unsafe<S: BaseFloat>(
    m: &Matrix4<S>,
    x: usize,
    y: usize,
    z: usize,
) -> Vector4<S> {
    let s: &[S; 16] = m.as_ref();
    let a = Vector4::new(
        *s.get_unchecked(4 + x),
        *s.get_unchecked(12 + x),
        *s.get_unchecked(x),
        *s.get_unchecked(8 + x),
    );
    let b = Vector4::new(
        *s.get_unchecked(8 + y),
        *s.get_unchecked(8 + y),
        *s.get_unchecked(4 + y),
        *s.get_unchecked(4 + y),
    );
    let c = Vector4::new(
        *s.get_unchecked(12 + z),
        *s.get_unchecked(z),
        *s.get_unchecked(12 + z),
        *s.get_unchecked(z),
    );

    let d = Vector4::new(
        *s.get_unchecked(8 + x),
        *s.get_unchecked(8 + x),
        *s.get_unchecked(4 + x),
        *s.get_unchecked(4 + x),
    );
    let e = Vector4::new(
        *s.get_unchecked(12 + y),
        *s.get_unchecked(y),
        *s.get_unchecked(12 + y),
        *s.get_unchecked(y),
    );
    let f = Vector4::new(
        *s.get_unchecked(4 + z),
        *s.get_unchecked(12 + z),
        *s.get_unchecked(z),
        *s.get_unchecked(8 + z),
    );

    let g = Vector4::new(
        *s.get_unchecked(12 + x),
        *s.get_unchecked(x),
        *s.get_unchecked(12 + x),
        *s.get_unchecked(x),
    );
    let h = Vector4::new(
        *s.get_unchecked(4 + y),
        *s.get_unchecked(12 + y),
        *s.get_unchecked(y),
        *s.get_unchecked(8 + y),
    );
    let i = Vector4::new(
        *s.get_unchecked(8 + z),
        *s.get_unchecked(8 + z),
        *s.get_unchecked(4 + z),
        *s.get_unchecked(4 + z),
    );
    let mut tmp = a.mul_element_wise(b.mul_element_wise(c));
    tmp += d.mul_element_wise(e.mul_element_wise(f));
    tmp += g.mul_element_wise(h.mul_element_wise(i));
    tmp -= a.mul_element_wise(e.mul_element_wise(i));
    tmp -= d.mul_element_wise(h.mul_element_wise(c));
    tmp -= g.mul_element_wise(b.mul_element_wise(f));
    tmp
}
