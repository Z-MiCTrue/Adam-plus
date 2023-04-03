# Adam-plus

This is a repository including the engineering-oriented improved Adam code and a test script to verify Adam's response characteristics to impulse signals.

## Engineering-oriented changes
1. Calculate derivatives with small approximations. Although a part of accuracy is sacrificed in calculating derivatives with a small approximation, its application scope is expanded (e.g. there is logical judgment in loss function, or there is integer restriction), avoiding complex derivative process.
2. Added parameter range limits to deal with some optimization problems with boundary limits.
3. Use matrix completely, to simultaneously start batch gradient descent from multiple starting points, which to some extents improve the operation efficiency.
4. The inspection process of lr(learning rate) is added to warn when the learning rate is set too low.
5. A process for preserving historical best is added. For the reason that the result at the end of the iteration may not be historically optimal caused by oscillation.