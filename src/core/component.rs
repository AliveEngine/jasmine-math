

use super::*;

pub struct Component<T, const COUNT: usize, const INDEX: usize> {
    pub data: [T; COUNT],
}

