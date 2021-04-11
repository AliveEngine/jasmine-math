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
use trivector4::{Trivector4};

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

impl<S: NumCast + Copy> Bivector4<S> {
    /// Component-wise casting to another type.
    #[inline]
    pub fn cast<T: NumCast>(&self) -> Option<Bivector4<T>> {
        let dir = match self.direction.cast() {
            Some(dir) => dir,
            None => return None
        };
        let mom = match self.moment.cast() {
            Some(mom) => mom,
            None => return None
        };
        Some(Bivector4 { direction: dir, moment: mom})
    }
}

impl<S: BaseFloat> approx::AbsDiffEq for Bivector4<S> {
    type Epsilon = S::Epsilon;

    #[inline]
    fn default_epsilon() -> S::Epsilon {
        S::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: S::Epsilon) -> bool {
        self.direction.abs_diff_eq(&other.direction, epsilon) &&
        self.moment.abs_diff_eq(&other.moment, epsilon)
    }
}

impl<S: BaseFloat> approx::RelativeEq for Bivector4<S> {
    #[inline]
    fn default_max_relative() -> S::Epsilon {
        S::default_max_relative()
    }

    #[inline]
    fn relative_eq(&self, other: &Self, epsilon: S::Epsilon, max_relative: S::Epsilon) -> bool {
        self.direction.relative_eq(&other.direction, epsilon, max_relative) &&
        self.moment.relative_eq(&other.moment, epsilon, max_relative)
    }
}

impl<S: BaseFloat> approx::UlpsEq for Bivector4<S> {
    #[inline]
    fn default_max_ulps() -> u32 {
        S::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: S::Epsilon, max_ulps: u32) -> bool {
        self.direction.ulps_eq(&other.direction, epsilon, max_ulps) &&
        self.moment.ulps_eq(&other.moment, epsilon, max_ulps)
    }
}

#[cfg(feature = "rand")]
impl<S> Distribution<Bivector4<S>> for Standard
    where Standard: Distribution<S>,
        S: BaseFloat {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Bivector4<S> {
        Bivector4{
            direction: self.sample(rng),
            moment: self.sample(rng)
        }
    }
}


macro_rules! impl_scalar_ops {
    (<$S:ident>) => {
        impl_operator!(Mul<Bivector4<$S>> for $S {
            fn mul(scalar, biv4) -> Bivector4<$S> { Bivector4::new_dir_m(
                biv4.direction * scalar,
                biv4.moment * scalar
            ) }
        });
        impl_operator!(Div<Bivector4<$S>> for $S {
            fn div(scalar, biv4) -> Bivector4<$S> { Bivector4::new_dir_m(
                scalar / biv4.direction,
                scalar / biv4.moment
            ) }
        });
        impl_operator!(Rem<Bivector4<$S>> for $S {
            fn rem(scalar, biv4) -> Bivector4<$S> { Bivector4::new_dir_m(
                scalar % biv4.direction,
                scalar % biv4.moment
            ) }
        });
    };
}

impl_scalar_ops!(<usize>);
impl_scalar_ops!(<u8> );
impl_scalar_ops!(<u16>);
impl_scalar_ops!(<u32> );
impl_scalar_ops!(<u64> );
impl_scalar_ops!(<isize> );
impl_scalar_ops!(<i8> );
impl_scalar_ops!(<i16> );
impl_scalar_ops!(<i32> );
impl_scalar_ops!(<i64> );
impl_scalar_ops!(<f32> );
impl_scalar_ops!(<f64> );


impl<S: BaseNum> Bivector4<S> {
    pub fn new_point3_point3(p: Point3<S>, q: Point3<S>) -> Bivector4<S> {
        let dir: Vector3<S> = q - p;
        let m: Bivector3<S> = p.cross(&q);
        Bivector4::new_dir_m(dir, m)
    }

    pub fn new_point3_vec3(p: Point3<S>, v:Vector3<S>) -> Bivector4<S> {
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

    pub fn translate(&self, t: &Vector3<S>) -> Bivector4<S> {
        Bivector4::new_dir_m(
            self.direction, self.moment + (t ^ self.direction)
        )
    }
}

impl<'a, S:BaseFloat> Bivector4<S> {
    pub fn unitize(&'a mut self) -> &'a Bivector4<S> 
    {
        *self *= self.direction.inverse_mag();
        self
    }

}

impl_assignment_operator!(<S:BaseNum> MulAssign<S> for Bivector4<S> {
    fn mul_assign(&mut self, scalar) {
        self.direction *= scalar;
        self.moment *= scalar;
    }
});

impl_assignment_operator!(<S:BaseNum> DivAssign<S> for Bivector4<S> {
    fn div_assign(&mut self, scalar) {
        let s = S::one() / scalar;
        self.direction *= s;
        self.moment *= s;
    }
});

impl<S: Neg<Output = S>> Neg for Bivector4<S> {
    type Output = Bivector4<S>;

    #[inline]
    default_fn!( neg(self) -> Bivector4<S> { 
        Bivector4::new(
            -self.direction.x, -self.direction.y, -self.direction.z, 
            -self.moment.x, -self.moment.y, -self.moment.z)
    });
}

impl_operator!(<S: BaseNum> Mul<S> for Bivector4<S> {
    fn mul(lhs, scalar) -> Bivector4<S> {
        Bivector4::new(
            lhs.direction.x * scalar, lhs.direction.y * scalar, lhs.direction.z * scalar, 
            lhs.moment.x * scalar, lhs.moment.y * scalar, lhs.moment.z * scalar)
    }
});

impl_operator!(<S: BaseNum> Div<S> for Bivector4<S> {
    fn div(lhs, scalar) -> Bivector4<S> {
        Bivector4::new(
            lhs.direction.x / scalar, lhs.direction.y / scalar, lhs.direction.z / scalar, 
            lhs.moment.x / scalar, lhs.moment.y / scalar, lhs.moment.z / scalar)
    }
});


impl_operator!(<S:BaseNum> BitXor<Point3<S>> for Point3<S>{
    fn bitxor(p, q) -> Bivector4<S> {
        Bivector4::new(
            q.x - p.x, q.y - p.y, q.z - p.z, 
            p.y * q.z - p.z * q.y, 
            p.z * q.x - p.x * q.z,
             p.x * q.y - p.y * q.x
        )
    }
});

impl_operator!(<S:BaseNum> BitXor<Vector3<S>> for Point3<S> {
    fn bitxor(p, v) -> Bivector4<S> {
        Bivector4::new(
            v.x, v.y, v.z, p.y * v.z - p.z * v.y, p.z * v.x - p.x * v.z, p.x * v.y - p.y * v.x
        )
    }
});

impl_operator!(<S:BaseNum> BitXor<Trivector4<S>> for Trivector4<S> {
    fn bitxor(f, g) -> Bivector4<S> {
        Bivector4::new(
            f.z * g.y - f.y * g.z, f.x * g.z - f.z * g.x, f.y * g.x - f.x * g.y, f.x * g.w - f.w * g.x, f.y * g.w - f.w * g.y, f.z * g.w - f.w * g.z
        )
    }
});

impl_operator!(<S:BaseFloat> BitXor<Point3<S>> for Bivector4<S> {
    fn bitxor(biv4, p) -> Trivector4<S>
    {
        Trivector4::new(
            biv4.direction.y * p.z - biv4.direction.z * p.y + biv4.moment.x,
            biv4.direction.z * p.x - biv4.direction.x * p.z + biv4.moment.y,
            biv4.direction.x * p.y - biv4.direction.y * p.x + biv4.moment.z,
            -biv4.moment.x * p.x - biv4.moment.y * p.y - biv4.moment.z * p.z
        )
    }
});

impl_operator!(<S:BaseFloat> BitXor<Bivector4<S>> for Point3<S> {
    fn bitxor(p, biv4) -> Trivector4<S> {
        biv4 ^ p
    }
});

impl_operator!(<S:BaseFloat> BitXor<Vector3<S>> for Bivector4<S> {
    fn bitxor(lhs, v) -> Trivector4<S> {
        Trivector4::new(
            lhs.direction.y * v.z - lhs.direction.z * v.y,
            lhs.direction.z * v.x - lhs.direction.x * v.z,
            lhs.direction.x * v.y - lhs.direction.y * v.x,
           -lhs.moment.x * v.x - lhs.moment.y * v.y - lhs.moment.z * v.z
        )
    }
});

impl_operator!(<S:BaseFloat> BitXor<Bivector4<S>> for Vector3<S> {
    fn bitxor(v, trv4) -> Trivector4<S> {
        trv4 ^ v
    }
});

impl_operator!(<S: BaseFloat> BitXor<Trivector4<S>> for Bivector4<S> {
    fn bitxor(biv4, f) -> Vector4<S> {
        Vector4::new(
            biv4.moment.y * f.z - biv4.moment.z * f.y + biv4.direction.x * f.w,
            biv4.moment.z * f.x - biv4.moment.x * f.z + biv4.direction.y * f.w,
            biv4.moment.x * f.y - biv4.moment.y * f.x + biv4.direction.z * f.w,
            -biv4.direction.x * f.x - biv4.direction.y * f.y - biv4.direction.z * f.z
        )
    }
});

impl_operator!(<S: BaseFloat> BitXor<Bivector4<S>> for Trivector4<S> {
    fn bitxor(f, biv4) -> Vector4<S> {
        biv4 ^ f
    }
});

impl_operator!(<S: BaseFloat> BitXor<Bivector4<S>> for Bivector4<S> {
    fn bitxor(lhs_k, rhs_l) -> S {
        -(lhs_k.direction ^ rhs_l.moment) - (lhs_k.moment ^ rhs_l.direction)
    }
});


impl_grassmann_wedge! (<S:BaseNum>, Point3<S>, Point3<S>, Bivector4<S>);
impl_grassmann_wedge! (<S:BaseNum>, Point3<S>, Vector3<S>, Bivector4<S>);
impl_grassmann_antiwedge! (<S:BaseNum>, Trivector4<S>, Trivector4<S>, Bivector4<S>);

impl_grassmann_wedge! (<S:BaseFloat>, Bivector4<S>, Point3<S>, Trivector4<S>);
impl_grassmann_wedge! (<S:BaseFloat>, Point3<S>, Bivector4<S>, Trivector4<S>);
impl_grassmann_wedge! (<S:BaseFloat>, Bivector4<S>, Vector3<S>, Trivector4<S>);
impl_grassmann_wedge! (<S:BaseFloat>, Vector3<S>, Bivector4<S>, Trivector4<S>);

impl_grassmann_antiwedge! (<S:BaseFloat>, Bivector4<S>, Trivector4<S>, Vector4<S>);
impl_grassmann_antiwedge! (<S:BaseFloat>, Trivector4<S>, Bivector4<S>, Vector4<S>);
impl_grassmann_antiwedge! (<S:BaseFloat>, Bivector4<S>, Bivector4<S>, S);

impl<S:BaseNum> ProjectTrait<Bivector4<S>> for Point3<S> {
    fn project(&self, biv4: &Bivector4<S>) -> Point3<S> {
        let d = biv4.direction.dot(self.to_vec());
        Point3::new(
            d * biv4.direction.x + biv4.direction.y * biv4.moment.z - 
            biv4.direction.z * biv4.moment.y,  d * biv4.direction.y + biv4.direction.z * biv4.moment.x - biv4.direction.x * biv4.moment.z, 
            d * biv4.direction.z + biv4.direction.x * biv4.moment.y - biv4.direction.y * biv4.moment.x
        )
    }
}

impl<S:BaseNum> ProjectTrait<Trivector4<S>> for Bivector4<S> {
    fn project(&self, f: &Trivector4<S>) -> Bivector4<S> {
        let xyz = f.xyz();
        let inv_xyz = !xyz;
        Bivector4::new_dir_m(
            self.direction - inv_xyz * (xyz ^ self.direction), 
            xyz * (inv_xyz ^ self.moment) - (inv_xyz ^ self.direction) * f.w
        )
    }
}

impl<S:BaseNum> AntiprojectTrait<Point3<S>> for Bivector4<S>
where 
    S:Neg<Output = S>
{
    fn anti_project(&self, p: &Point3<S> ) -> Bivector4<S> {
        let point = Point3::new(p.x, p.y, p.z);
        Bivector4::new_point3_vec3(point, self.direction)
    }
}

impl<S:BaseNum> AntiprojectTrait<Bivector4<S>> for Trivector4<S> {
    fn anti_project(&self, biv4: &Bivector4<S> ) -> Trivector4<S> {
        let xyz = self.xyz();
        let not_direction = !biv4.direction;
        let biv3 = xyz - not_direction * ( xyz ^ biv4.direction);
        let scalar = biv4.moment ^ not_direction ^ xyz;
        Trivector4::new_bivec3_s(
            &biv3,
            scalar
        )
    }
}






