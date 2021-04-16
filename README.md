# jasmine-math


# jasmine-math

This is a rust math library containing classes for vectors, matrices, quaternions, and elements of projective geometric algebra. The specific classes are the following:
|--------|--------|
|name|description|
|--------|--------|
|vector2| A 2D vector (*x*, *y*) that extends to four dimensions as (*x*, *y*, 0, 0). |
|vector3| A 3D vector (*x*, *y*, *z*) that extends to four dimensions as (*x*, *y*, *z*, 0). |
|vector4| A 4D vector (*x*, *y*, *z*, *w*).|
|point2| A 2D point (*x*, *y*) that extends to four dimensions as (*x*, *y*, 0, 1).|
|point3| A 3D point (*x*, *y*, *z*) that extends to four dimensions as (*x*, *y*, *z*, 1).|
|matrix2| A 2×2 matrix.|
|matrix3| A 3×3 matrix.|
|matrix4| A 4×4 matrix.|
|transform4| A 4×4 matrix with fourth row always (0, 0, 0, 1).|
|quaternion| A convention quaternion x**i** + y**j** + z**k** + w.|
|bivector3| A 3D bivector *x* **e**<sub>23</sub> + *y* **e**<sub>31</sub> + *z* **e**<sub>12</sub>.|
|bivector4| A 4D bivector (line) *v<sub>x</sub>* **e**<sub>41</sub> + *v<sub>y</sub>* **e**<sub>42</sub> + *v<sub>z</sub>* **e**<sub>43</sub> + *m<sub>x</sub>* **e**<sub>23</sub> + *m<sub>y</sub>* **e**<sub>31</sub> + *m<sub>z</sub>* **e**<sub>12</sub>.|
|trivector4| A 4D trivector (plane) *x* **e**<sub>234</sub> + *y* **e**<sub>314</sub> + *z* **e**<sub>124</sub> + *w* **e**<sub>321</sub>.|
|motor| A 4D motion operator *r<sub>x</sub>* **e**<sub>41</sub> + *r<sub>y</sub>* **e**<sub>42</sub> + *r<sub>z</sub>* **e**<sub>43</sub> + *r<sub>w</sub>* 𝟙 + *u<sub>x</sub>* **e**<sub>23</sub> + *u<sub>y</sub>* **e**<sub>31</sub> + *u<sub>z</sub>* **e**<sub>12</sub> + *u<sub>w</sub>*.|


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

## todolist

- [x] 基础内容或功能
- [ ] test
- [ ] 文档
- [ ] sample
- [ ] 性能优化

