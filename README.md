# jasmine-math


# jasmine-math

This is a rust math library containing classes for vectors, matrices, quaternions, and elements of projective geometric algebra. The specific classes are the following:
|--------|--------|
|name|description|
|--------|--------|
|vector2d| A 2D vector (*x*, *y*) that extends to four dimensions as (*x*, *y*, 0, 0). |
|vector3d| A 3D vector (*x*, *y*, *z*) that extends to four dimensions as (*x*, *y*, *z*, 0). |
|vector4d| A 4D vector (*x*, *y*, *z*, *w*).|
|point2d| A 2D point (*x*, *y*) that extends to four dimensions as (*x*, *y*, 0, 1).|
|point3d| A 3D point (*x*, *y*, *z*) that extends to four dimensions as (*x*, *y*, *z*, 1).|
|matrix2d| A 2×2 matrix.|
|matrix3d| A 3×3 matrix.|
|matrix4d| A 4×4 matrix.|
|transform4d| A 4×4 matrix with fourth row always (0, 0, 0, 1).|
|quaternion| A convention quaternion x**i** + y**j** + z**k** + w.|
|bivector3d| A 3D bivector *x* **e**<sub>23</sub> + *y* **e**<sub>31</sub> + *z* **e**<sub>12</sub>.|
|bivector4d| A 4D bivector (line) *v<sub>x</sub>* **e**<sub>41</sub> + *v<sub>y</sub>* **e**<sub>42</sub> + *v<sub>z</sub>* **e**<sub>43</sub> + *m<sub>x</sub>* **e**<sub>23</sub> + *m<sub>y</sub>* **e**<sub>31</sub> + *m<sub>z</sub>* **e**<sub>12</sub>.|
|trivector4d| A 4D trivector (plane) *x* **e**<sub>234</sub> + *y* **e**<sub>314</sub> + *z* **e**<sub>124</sub> + *w* **e**<sub>321</sub>.|
|motor| A 4D motion operator *r<sub>x</sub>* **e**<sub>41</sub> + *r<sub>y</sub>* **e**<sub>42</sub> + *r<sub>z</sub>* **e**<sub>43</sub> + *r<sub>w</sub>* 𝟙 + *u<sub>x</sub>* **e**<sub>23</sub> + *u<sub>y</sub>* **e**<sub>31</sub> + *u<sub>z</sub>* **e**<sub>12</sub> + *u<sub>w</sub>*.|



## Component Swizzling

Vector components can be swizzled using shading-language syntax as long as there are no repeated components. As an example, the following expressions are all valid for a `Vector3D` object `v`:

* `v.x` – The *x* component of `v`.
* `v.xy` – A 2D vector having the *x* and *y* components of `v`.
* `v.yzx` – A 3D vector having the components of `v` in the order (*y*, *z*, *x*).

Rows, columns, and submatrices can be extracted from matrix objects using a similar syntax. As an example, the following expressions are all valid for a `Matrix3D` object `m`:

* `m.m12` – The (1,2) entry of `m`.
* `m.row0` – The first row of `m`.
* `m.col1` – The second column of `m`.
* `m.matrix2D` – The upper-left 2×2 submatrix of `m`.
* `m.transpose` – The transpose of `m`.

All of the above are generally *free operations*, with no copying, when their results are consumed by an expression. For more information, see Eric Lengyel's 2018 GDC talk [Linear Algebra Upgraded](http://terathon.com/gdc18_lengyel.pdf).

## Geometric Algebra

The `^` operator is overloaded for cases in which the wedge or antiwedge product can be applied between vectors, bivectors, points, lines, and planes. (Note that `^` has lower precedence than just about everything else, so parentheses will be necessary.)

The library does not provide operators that directly calculate the geometric product and antiproduct because they would tend to generate inefficient code and produce intermediate results having useless types when something like the sandwich product **Q** ⟇ *p* ⟇ ~**Q** appears in an expression. Instead, there are `Transform()` functions that take some object *p* for the first parameter and the motor **Q** with which to transform it for the second parameter.

See Eric Lengyel's [Projective Geometric Algebra website](http://projectivegeometricalgebra.org) for more information about operations among these types.

## API Documentation

There is API documentation embedded in the header files. The formatted equivalent can be found in the [C4 Engine documentation](http://c4engine.com/docs/Math/index.html).

## references
http://terathon.com/gdc14_lengyel.pdf
http://projectivegeometricalgebra.org
https://medium.com/@Razican/learning-simd-with-rust-by-finding-planets-b85ccfb724c3

this: https://github.com/rustgd/cgmath.git
simd
https://medium.com/@Razican/learning-simd-with-rust-by-finding-planets-b85ccfb724c3
https://doc.rust-lang.org/edition-guide/rust-2018/simd-for-faster-computing.html

https://www.youtube.com/watch?v=tX4H_ctggYo
https://enkimute.github.io/ganja.js/examples/coffeeshop.html#pga2d_points_and_lines


http://terathon.com/blog/
http://mathfor3dgameprogramming.com/

gpa
https://github.com/enkimute/ganja.js
http://terathon.com/blog/projective-geometric-algebra-done-right/
http://terathon.com/blog/symmetries-in-projective-geometric-algebra/

## 

we need imp opt traits : https://doc.rust-lang.org/std/ops/index.html

slug
http://terathon.com/blog/dynamic-glyph-dilation/

## example

```bash
cargo run --example name ..args
```

## features
swizzle
