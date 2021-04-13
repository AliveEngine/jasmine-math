// Copyright 2013-2014 The CGMath Developers. For a full listing of the authors,
// refer to the Cargo.toml file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Angle units for type-safe, self-documenting code.

use std::f64;
use std::fmt;
use std::iter;
use std::ops::*;

use num_traits::{cast, Bounded};
#[cfg(feature = "rand")]
use rand::{
    distributions::{uniform::SampleUniform, Distribution, Standard},
    Rng,
};

use structure::*;

use approx;
use num::BaseFloat;

/// An angle, in radians.
///
/// This type is marked as `#[repr(C)]`.
#[repr(C)]
#[derive(Copy, Clone, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Rad<S>(pub S);

/// An angle, in degrees.
///
/// This type is marked as `#[repr(C)]`.
#[repr(C)]
#[derive(Copy, Clone, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Deg<S>(pub S);

impl<S> From<Rad<S>> for Deg<S>
where
    S: BaseFloat,
{
    #[inline]
    fn from(rad: Rad<S>) -> Deg<S> {
        Deg(rad.0 * cast(180.0 / f64::consts::PI).unwrap())
    }
}

impl<S> From<Deg<S>> for Rad<S>
where
    S: BaseFloat,
{
    #[inline]
    fn from(deg: Deg<S>) -> Rad<S> {
        Rad(deg.0 * cast(f64::consts::PI / 180.0).unwrap())
    }
}

macro_rules! impl_angle {
    ($Angle:ident, $fmt:expr, $full_turn:expr, $hi:expr) => {
        impl<S: BaseFloat> Zero for $Angle<S> {
            #[inline]
            fn zero() -> $Angle<S> {
                $Angle(S::zero())
            }

            #[inline]
            fn is_zero(&self) -> bool {
                ulps_eq!(self, &Self::zero())
            }
        }

        impl<S: BaseFloat> iter::Sum<$Angle<S>> for $Angle<S> {
            #[inline]
            fn sum<I: Iterator<Item=$Angle<S>>>(iter: I) -> $Angle<S> {
                iter.fold($Angle::zero(), Add::add)
            }
        }

        impl<'a, S: 'a + BaseFloat> iter::Sum<&'a $Angle<S>> for $Angle<S> {
            #[inline]
            fn sum<I: Iterator<Item=&'a $Angle<S>>>(iter: I) -> $Angle<S> {
                iter.fold($Angle::zero(), Add::add)
            }
        }

        impl<S: BaseFloat> Angle for $Angle<S> {
            type Unitless = S;

            #[inline] fn full_turn() -> $Angle<S> { $Angle(cast($full_turn).unwrap()) }

            #[inline] fn sin(self) -> S { Rad::from(self).0.sin() }
            #[inline] fn cos(self) -> S { Rad::from(self).0.cos() }
            #[inline] fn tan(self) -> S { Rad::from(self).0.tan() }
            #[inline] fn sin_cos(self) -> (S, S) { Rad::from(self).0.sin_cos() }

            #[inline] fn asin(a: S) -> $Angle<S> { Rad(a.asin()).into() }
            #[inline] fn acos(a: S) -> $Angle<S> { Rad(a.acos()).into() }
            #[inline] fn atan(a: S) -> $Angle<S> { Rad(a.atan()).into() }
        //     #[inline] fn atan_yx(y: S, x: S) -> $Angle<S> {
        //         if y.abs() > S::min() {
        //             if x.abs() > S::min() {
        //                 let r = atan(y / x);
        //                 if x < S::zero() {
        //                     if y >= S::zero() {
        //                         r += cast(tau_over_2).unwrap();
        //                     } else {
        //                         r -= cast(tau_over_2).unwrap();
        //                     }
        //                 }
        //                 Rad(r)
        //             } else {
        //                 if y < S::zero() {
        //                     Rad(-tau_over_4).into()
        //                 } else {
        //                     Rad(tau_over_4).into()
        //                 }
        //             }
        //         } else {
        //             if x < S::zero() {
        //                 Rad(cast(tau_over_2).unwrap()).into()
        //             } else {
        //                 Rad(S::zero()).into()
        //             }
        //         }
        //     }
             #[inline] fn atan2(a: S, b: S) -> $Angle<S> { Rad(a.atan2(b)).into() }
        }

        impl<S: BaseFloat> Neg for $Angle<S> {
            type Output = $Angle<S>;

            #[inline]
            fn neg(self) -> $Angle<S> { $Angle(-self.0) }
        }

        impl<'a, S: BaseFloat> Neg for &'a $Angle<S> {
            type Output = $Angle<S>;

            #[inline]
            fn neg(self) -> $Angle<S> { $Angle(-self.0) }
        }

        impl<S: Bounded> Bounded for $Angle<S> {
            #[inline]
            fn min_value() -> $Angle<S> {
                $Angle(S::min_value())
            }

            #[inline]
            fn max_value() -> $Angle<S> {
                $Angle(S::max_value())
            }
        }

        impl_operator!(<S: BaseFloat> Add<$Angle<S> > for $Angle<S> {
            fn add(lhs, rhs) -> $Angle<S> { $Angle(lhs.0 + rhs.0) }
        });
        impl_operator!(<S: BaseFloat> Sub<$Angle<S> > for $Angle<S> {
            fn sub(lhs, rhs) -> $Angle<S> { $Angle(lhs.0 - rhs.0) }
        });
        impl_operator!(<S: BaseFloat> Div<$Angle<S> > for $Angle<S> {
            fn div(lhs, rhs) -> S { lhs.0 / rhs.0 }
        });
        impl_operator!(<S: BaseFloat> Rem<$Angle<S> > for $Angle<S> {
            fn rem(lhs, rhs) -> $Angle<S> { $Angle(lhs.0 % rhs.0) }
        });
        impl_assignment_operator!(<S: BaseFloat> AddAssign<$Angle<S> > for $Angle<S> {
            fn add_assign(&mut self, other) { self.0 += other.0; }
        });
        impl_assignment_operator!(<S: BaseFloat> SubAssign<$Angle<S> > for $Angle<S> {
            fn sub_assign(&mut self, other) { self.0 -= other.0; }
        });
        impl_assignment_operator!(<S: BaseFloat> RemAssign<$Angle<S> > for $Angle<S> {
            fn rem_assign(&mut self, other) { self.0 %= other.0; }
        });

        impl_operator!(<S: BaseFloat> Mul<S> for $Angle<S> {
            fn mul(lhs, scalar) -> $Angle<S> { $Angle(lhs.0 * scalar) }
        });
        impl_operator!(<S: BaseFloat> Div<S> for $Angle<S> {
            fn div(lhs, scalar) -> $Angle<S> { $Angle(lhs.0 / scalar) }
        });
        impl_assignment_operator!(<S: BaseFloat> MulAssign<S> for $Angle<S> {
            fn mul_assign(&mut self, scalar) { self.0 *= scalar; }
        });
        impl_assignment_operator!(<S: BaseFloat> DivAssign<S> for $Angle<S> {
            fn div_assign(&mut self, scalar) { self.0 /= scalar; }
        });

        impl<S: BaseFloat> approx::AbsDiffEq for $Angle<S> {
            type Epsilon = S::Epsilon;

            #[inline]
            fn default_epsilon() -> S::Epsilon {
                S::default_epsilon()
            }

            #[inline]
            fn abs_diff_eq(&self, other: &Self, epsilon: S::Epsilon) -> bool {
                S::abs_diff_eq(&self.0, &other.0, epsilon)
            }
        }

        impl<S: BaseFloat> approx::RelativeEq for $Angle<S> {
            #[inline]
            fn default_max_relative() -> S::Epsilon {
                S::default_max_relative()
            }

            #[inline]
            fn relative_eq(&self, other: &Self, epsilon: S::Epsilon, max_relative: S::Epsilon) -> bool {
                S::relative_eq(&self.0, &other.0, epsilon, max_relative)
            }
        }

        impl<S: BaseFloat> approx::UlpsEq for $Angle<S> {
            #[inline]
            fn default_max_ulps() -> u32 {
                S::default_max_ulps()
            }

            #[inline]
            fn ulps_eq(&self, other: &Self, epsilon: S::Epsilon, max_ulps: u32) -> bool {
                S::ulps_eq(&self.0, &other.0, epsilon, max_ulps)
            }
        }

        #[cfg(feature = "rand")]
        impl<S> Distribution<$Angle<S>> for Standard
            where Standard: Distribution<S>,
                S: BaseFloat + SampleUniform {
            #[inline]
            fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> $Angle<S> {
                $Angle(rng.gen_range(cast::<_, S>(-$hi).unwrap() .. cast::<_, S>($hi).unwrap()))
            }
        }

        impl<S: fmt::Debug> fmt::Debug for $Angle<S> {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(f, $fmt, self.0)
            }
        }
    }
}

impl_angle!(Rad, "{:?} rad", f64::consts::PI * 2.0, f64::consts::PI);
impl_angle!(Deg, "{:?}°", 360, 180);

pub const infinity: u32 = 0x7F800000;
pub const minus_infinity: u32 = 0xFF800000;
pub const tau: f32 =				6.2831853071795864769252867665590f32;
pub const two_tau: f32 =			12.566370614359172953850573533118f32;
pub const three_tau_over_4: f32 =	4.7123889803846898576939650749193f32;
pub const three_tau_over_8: f32 =	2.3561944901923449288469825374596f32;
pub const tau_over_2: f32 =			3.1415926535897932384626433832795f32;
pub const tau_over_3: f32 =			2.0943951023931954923084289221863f32;
pub const two_tau_over_3: f32 =		4.1887902047863909846168578443727f32;
pub const tau_over_4: f32 =			1.5707963267948966192313216916398f32;
pub const tau_over_6: f32 =			1.0471975511965977461542144610932f32;
pub const tau_over_8: f32 =			0.78539816339744830961566084581988f32;
pub const tau_over_12: f32 =		0.52359877559829887307710723054658f32;
pub const tau_over_16: f32 =		0.39269908169872415480783042290994f32;
pub const tau_over_24: f32 =		0.26179938779914943653855361527329f32;
pub const tau_over_40: f32 =		0.15707963267948966192313216916398f32;
pub const one_over_tau: f32 =		1.0f32 / tau;
pub const two_over_tau: f32 =		2.0f32 / tau;
pub const four_over_tau: f32 =		4.0f32 / tau;
pub const one_over_two_tau: f32 =	0.5f32 / tau;

pub const pi: f32 =					3.1415926535897932384626433832795f32;
pub const two_pi: f32 =				6.2831853071795864769252867665590f32;
pub const four_pi: f32 =			12.566370614359172953850573533118f32;
pub const three_pi_over_2: f32 =	4.7123889803846898576939650749193f32;
pub const three_pi_over_4: f32 = 	2.3561944901923449288469825374596f32;
pub const two_pi_over_3: f32 =		2.0943951023931954923084289221863f32;
pub const four_pi_over_3: f32 = 	4.1887902047863909846168578443727f32;
pub const pi_over_2: f32 =			1.5707963267948966192313216916398f32;
pub const pi_over_3: f32 =			1.0471975511965977461542144610932f32;
pub const pi_over_4: f32 =			0.78539816339744830961566084581988f32;
pub const pi_over_6: f32 =			0.52359877559829887307710723054658f32;
pub const pi_over_8: f32 =			0.39269908169872415480783042290994f32;
pub const pi_over_12: f32 =			0.26179938779914943653855361527329f32;
pub const pi_over_20: f32 =			0.15707963267948966192313216916398f32;
pub const one_over_pi: f32 =		1.0f32 / pi;
pub const one_over_two_pi: f32 =	0.5f32/ pi;
pub const one_over_four_pi: f32 =	0.25f32 / pi;

pub const sqrt_2: f32 =				1.4142135623730950488016887242097f32;
pub const sqrt_2_over_2: f32 =		0.70710678118654752440084436210485f32;
pub const sqrt_2_over_3: f32 =		0.47140452079103168293389624140323f32;
pub const sqrt_3: f32 =				1.7320508075688772935274463415059f32;
pub const sqrt_3_over_2: f32 =		0.86602540378443864676372317075294f32;
pub const sqrt_3_over_3: f32 =		0.57735026918962576450914878050196f32;

pub const ln_2: f32 =				0.69314718055994530941723212145818f32;
pub const one_over_ln_2: f32 =		1.4426950408889634073599246810019f32;
pub const ln_10: f32 =				2.3025850929940456840179914546844f32;
pub const one_over_ln_10: f32 =		0.43429448190325182765112891891661f32;
pub const ln_256: f32 =				5.5451774444795624753378569716654f32;