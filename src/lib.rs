

#![cfg_attr(feature = "simd", feature(specialization))]

#[macro_use]
extern crate approx;

#[cfg(feature = "mint")]
pub extern crate mint;

pub extern crate num_traits;
#[cfg(feature = "rand")]
extern crate rand;

#[cfg(feature = "serde")]
#[macro_use]
extern crate serde;

#[cfg(feature = "simd")]
extern crate simd;


// Re-exports

pub use approx::*;


// #[macro_use]
mod macros;


pub mod core;
pub mod math;
pub mod consts;