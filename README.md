# Adam-plus

This is a repository including the engineering-oriented improved Adam code and a test script to verify Adam's response characteristics to impulse signals.


## File Interpretation:
Adam-plus: the engineering-oriented improved Adam code
Adam-core: the test script to verify Adam's response characteristics to impulse signals and some figures


## Engineering-oriented changes: 
1. Calculate derivatives with small approximations. Although a part of accuracy is sacrificed in calculating derivatives with a small approximation, its application scope is expanded (e.g. there is logical judgment in loss function, or there is integer restriction), avoiding complex derivative process.
2. Added parameter range limits to deal with some optimization problems with boundary limits.
3. Use matrix completely, to simultaneously start batch gradient descent from multiple starting points, which to some extents improve the operation efficiency.
4. The inspection process of lr(learning rate) is added to warn when the learning rate is set too low.
5. A process for preserving historical best is added. For the reason that the result at the end of the iteration may not be historically optimal caused by oscillation.


## Requirements:
```
numpy
matplotlib [Optional]
```


## Quick start: modify parameters in [`Adam_plus.py`](./Adam_plus/Adam_plus.py)
### Instantiation of the optimizer
```python
Oa = Optimizer_Adam((2, 2), lr=5e-2, integrate=1e-4))
 
# (2, 2) 代表了 (起点数=2, 待优化参数=2)
# lr=5e-2 代表了 学习率=5e-2
# integrate=1e-4 代表了 小量步长=1e-4
```
The integer is exactly the small step size for approximate derivation. If the parameter is required to be an integer (such as optimizing the number of products produced each time), it can be set to 1. If anisotropy is required, it can be modified to an array (e.g. Oa.integrate=np.array ([0.1, 1]))

### Set starting point set
```python
Oa.vector_x = np.array([[0., 0.],  # seed 1
                        [7., 17.]])  # seed 2
```
The initial number of seeds and optimization parameters set here need to be consistent with the settings when instantiating the optimizer, which will simultaneously iterate on them in batches. This option is optional. If not specified, the optimizer will generate a corresponding number of seeds randomly within the parameter constraint range by default.

### Set parameter constraint range
```python
Oa.variable_range = np.array([[-100, 100],  # 参数 1
                              [-100, 100]])  # 参数 2
```
This is an optional setting. If not specified, the optimizer will default to the entire real number field for optimization.

### Set aim function
```python
Oa.aim_function = test_func
```
The function entry address is assigned to the optimizer here for later calls.

### Start optimization
```python
solve = Oa.run(max_time=1e4, min_grade=1e-4)
print(f'solve:\n{solve}')
 
# max_time=1e4 代表最多迭代1e4次
# min_grade=1e-4 代表计算所得梯度小于1e-4时将停止迭代。这里设置的原因是，在梯度小于该值时，估算梯度带来的误差将超过梯度本身，继续迭代没有意义
```

### *Check lr status
```python
if self.loss_log.shape[0] >= 3:
    loss_trend = (self.loss_log[-1] - self.loss_log[-2]) / (self.loss_log[-2] - self.loss_log[-3] + 1e-9)
    if np.min(loss_trend) > 0.1:
        print(f'< warning: lr needs to be set larger -- {loss_trend} >')
```
The purpose of this section of code is to check whether the lr setting is too small when the iteration count reaches the upper limit, resulting in a huge empty space. By setting multiple checkpoints in the iteration, it is determined whether the optimization level of the current checkpoint has significantly decreased compared to the previous checkpoint at the end of the iteration. If not, it indicates that there is still a lot of empty space and the learning rate lr is set too low.


## More instructions
For more information please visit the blog: <http://t.csdn.cn/NnL4n>
