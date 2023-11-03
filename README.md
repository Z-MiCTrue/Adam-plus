# Adam-plus

This is a repository including the engineering-oriented improved Adam code and a test script to verify Adam's response characteristics to impulse signals. At the same time, there is a theoretical introduction from the perspective of information theory.

---

[TOC]

---


## File Interpretation
Adam-plus: the engineering-oriented improved Adam code
Adam-core: the test script to verify Adam's response characteristics to impulse signals and some figures




## Engineering-oriented changes
1. Calculate derivatives with small approximations. Although a part of accuracy is sacrificed in calculating derivatives with a small approximation, its application scope is expanded (e.g. there is logical judgment in loss function, or there is integer restriction), avoiding complex derivative process.
2. Added parameter range limits to deal with some optimization problems with boundary limits.
3. Use matrix completely, to simultaneously start batch gradient descent from multiple starting points, which to some extents improve the operation efficiency.
4. The inspection process of lr(learning rate) is added to warn when the learning rate is set too low.
5. A process for preserving historical best is added. For the reason that the result at the end of the iteration may not be historically optimal caused by oscillation.




## Requirements
```
numpy
matplotlib [Optional]
```



## Quick start: modify parameters in [`Adam.py`](./Adam.py)

### Instantiation of the optimizer
```python
optimizer = Optimizer_Adam((2, 2), lr=1e1, integrate=1e-4)
 
# (2, 2) 代表了 (起点数=2, 待优化参数=2)
# lr=5e-2 代表了 学习率=5e-2
# integrate=1e-4 代表了 小量步长=1e-4
```
The integer is exactly the small step size for approximate derivation. If the parameter is required to be an integer (such as optimizing the number of products produced each time), it can be set to 1. If anisotropy is required, it can be modified to an array (e.g. Oa.integrate=np.array ([0.1, 1]))

### Set starting point set
```python
optimizer.x = np.array([[0., 0.],  # seed 1
                        [-1., -1.]], dtype=np.float32)  # seed 2
```
The initial number of seeds and optimization parameters set here need to be consistent with the settings when instantiating the optimizer, which will simultaneously iterate on them in batches. This option is optional. If not specified, the optimizer will generate a corresponding number of seeds randomly within the parameter constraint range by default.

### Set parameter constraint range
```python
optimizer.border_alarmer.variable_range = np.array([[-15, 15],  # 参数 1
                                                    [-30, 30]], dtype=np.float32)  # 参数 2
```
This is an optional setting. If not specified, the optimizer will default to the entire real number field for optimization.

### Set lr attenuation

```
optimizer.milestone = np.array([500, 750], dtype=np.float32)
```

Set a milestone for attenuation and when the number of iterations reaches a certain value, the learning rate will decay exponentially.

### Set aim function

```python
optimizer.aim_function = test_func
```
The function entry address is assigned to the optimizer here for later calls.

### Start optimization
```python
solve = optimizer.run(max_time=1e4, min_grade=1e-4)
print(f'solve:\n{solve}')
 
# max_time=1e4 代表最多迭代1e4次
# min_grade=1e-4 代表计算所得梯度小于1e-4时将停止迭代。这里设置的原因是，在梯度小于该值时，估算梯度带来的误差将超过梯度本身，继续迭代没有意义
```

### *Check lr status
```python
if self.loss_log.shape[0] >= 3:
    loss_trend = (self.loss_log[-1] - self.loss_log[-2] + self.eps) / (self.loss_log[-2] - self.loss_log[-3] + self.eps)
    if np.min(loss_trend) > 0.1:
        print(f'< warning: lr needs to be set larger -- {loss_trend} >')
else:
    print(f'< warning: end prematurely, lr status cannot be determined >')
```
The purpose of this section of code is to check whether the lr setting is too small when the iteration count reaches the upper limit, resulting in a huge empty space. By setting multiple checkpoints in the iteration, it is determined whether the optimization level of the current checkpoint has significantly decreased compared to the previous checkpoint at the end of the iteration. If not, it indicates that there is still a lot of empty space and the learning rate lr is set too low.



## Theoretical introduction from the perspective of information theory

> The Adam optimization algorithm actually implements an IIR digital filter to filter gradient signals

### 1. Overall process

<img src="./Figures/overall process.png" alt="pseudocode" />

### 2. Essentially: IIR digital filters filter gradient signals

If Momentum-SGD incorporates the concept of momentum from physics into gradient descent, I would prefer to believe that Adam takes into account the concept of IIR filters in digital signal processing:

![](http://latex.codecogs.com/png.latex?m_{t}\leftarrow\beta_{1}\cdot{m_{t-1}}+(1-\beta_{1})\cdot{g_{t}}\\v_{t}\leftarrow\beta_{2}\cdot{v_{t-1}}+(1-\beta_{2})\cdot{g_{t}^{2}})

Iterations after iterations, these two lines are the core of Adam, if $\beta_ 1=0.9$, unfold it to obtain:

![](http://latex.codecogs.com/png.latex?m_{100}=0.1\theta_{100}+0.1*0.9\theta_{99}+0.1*(0.9)^2\theta_{98}+0.1*(0.9)^3\theta_{97}+0.1*(0.9)^4\theta_{96})

If the gradient calculated after each iteration is regarded as an impulse signal, then as the number of iterations increases, the impulse signal multiplied by its corresponding weight will appear as an exponential decay signal in time as follows:

<img src="./Figures/exponential decay signal.png" alt="exponential decay signal" style="zoom:50%;" />

From this perspective, this equation is equivalent to: convolving the impulse sequence (historical gradient) with the exponential decay sequence;

In other words, assuming this is a linear time invariant system, **the response of $m_t, v_t$ to the unit pulse signal (system function) is an exponential decay function**.

Meanwhile, the system function of a system where the unit impulse response is an exponential decay function is expressed as:

<img src="./Figures/system function.jpg" alt="system function" style="zoom:50%;" />

The system function representation of IIR digital filters is as follows:

![](http://latex.codecogs.com/png.latex?H(z)=\frac{Y(z)}{X(z)}=\frac{\sum_{i=0}^Ma_iz^{-i}}{1-\sum_{i=1}^Nb_iz^{-i}}=a_0\frac{\prod_{i=1}^M(1-c_iz^{-1})}{\prod_{i=1}^N(1-d_iz^{-1})})

The processing of historical gradients in Adam can be seen as an IIR digital filtering process with only low order terms, which involves filtering gradients and filtering gradients squared (response amplitude). In my opinion, the correction of errors is mainly aimed at correcting the impact of zero state response.

**So from this perspective, there are two ways to increase Adam's sensitivity to gradients (and vice versa):**

- **Add the number of terms during iteration, even if it becomes a higher-order filter**

- **Increase the value of $\beta$;**

### 3. Summary

Without adding an offset correction term, the equivalent formula for Adam can be derived from the previous section as follows:

![](http://latex.codecogs.com/png.latex?S(t)=-lr\cdot\frac{(1-\beta_1)\cdot[g(t)*\beta_1^t]}{\sqrt{(1-\beta_2)\cdot[g^2(t)*\beta_2^t]}})


> Where S(t) is the final step sequence; g(t) is a gradient sequence; * Represents convolution;

To verify the correctness of this equation and investigate the unit impulse response (system function) brought by the unit impulse signal to the entire system, I used the following code:

```python
import numpy as np
import matplotlib.pyplot as plt


class Adam_Core:
    def __init__(self):
        self.lr = 5e-2
        self.eps = 1e-9
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.grad_smooth = 0
        self.velocity_smooth = 0

    # 迭代函数
    def forward(self, x):
        y = []
        for n, grad in enumerate(x):
            self.grad_smooth = self.beta_1 * self.grad_smooth + (1 - self.beta_1) * grad
            # 偏移修正项
            self.velocity_smooth = self.beta_2 * self.velocity_smooth + (1 - self.beta_2) * np.power(grad, 2)
            self.grad_smooth = self.grad_smooth / (1 - np.power(self.beta_1, n + 1))
            self.velocity_smooth = self.velocity_smooth / (1 - np.power(self.beta_1, n + 1))
            step = self.lr * (self.grad_smooth + self.eps) / (np.power(self.velocity_smooth, 1 / 2) + self.eps)
            y.append(step)
        return y


# 等效函数
def adam_core_conv(x, lr, beta1=0.9, beta2=0.999, eps=1e-9):
    beta1_list = np.logspace(0, len(x)-1, num=len(x), endpoint=True, base=beta1)
    beta2_list = np.logspace(0, len(x) - 1, num=len(x), endpoint=True, base=beta2)
    g = np.convolve(x, beta1_list, 'full')
    v = np.convolve(np.power(x, 2), beta2_list, 'full')
    y = lr * ((1 - beta1) * g) / (np.power((1 - beta2) * v, 0.5) + eps)
    return y


if __name__ == '__main__':
    # 构造单位冲激函数
    test_x = np.append(np.array([0, 0, 1]), np.zeros(80))
    adam_core = Adam_Core()
    res = adam_core.forward(test_x)
    # res = adam_core_conv(test_x, lr=5e-2)
    plt.bar(range(len(test_x)), test_x, color='r')
    plt.bar(range(len(res)), res, color='b')
    plt.savefig('1.png')
    plt.show()
```

Results calculated using iterative methods:

<img src="./Figures/using iterative methods.png" alt="using iterative methods" style="zoom: 67%;" />

The result calculated using  the equivalent calculation formula (as it is a complete convolution, the length of the result sequence is twice the input length):

<img src="./Figures/using the equivalent calculation formula.png" alt="img" style="zoom:67%;" />

Result of adding offset correction term:

<img src="./Figures/adding offset correction term.png" alt="adding offset correction term" style="zoom:67%;" />

> Red represents the unit impulse sequence, and Blue represents the unit impulse response sequence

The parameters used for the above results are:

```
lr = 5e-2
eps = 1e-9
beta_1 = 0.9
beta_2 = 0.999
```

So the following conclusion can be drawn:

- The equivalent calculation formula is correct;
- The offset correction term has a significant impact on the unit impulse response of the overall system, as follows:
    1. From the perspective of amplitude, adding the offset correction term significantly improves the sensitivity to gradients compared to not adding it, with a maximum of **2** times after adding it, while not adding it is only about **0.18** times;
    2. **From the perspective of response time, adding the offset correction term has a delay of about 15 iterations compared to not adding it in response time, which means that the impulse response does not immediately reach its peak after the arrival of the impulse. In other words, it still has a certain amount of momentum, improving the escape ability at the local minimum point**

> Due to the inclusion of offset correction terms, the equivalent calculation formula is too complex (including accumulation and quadratic convolution), so it is not given here

**In summary, the following conclusions can be drawn:**

1. **Increasing the number of terms during iteration to make it a higher-order filter** or **increasing the value of $\beta$** can **increase Adam's sensitivity to gradients**;
2. **The offset correction term** not only **increases Adam's sensitivity to gradients**, but also **increases its escape ability at local minimum points**;
3. **The impact of learning rate on optimization performance is crucial**, as it limits the upper limit of the actual iteration step size. So **setting it too small will cause the algorithm to fail to converge** even after tens of thousands of iterations, while **setting it too large will cause it to be too sensitive to gradients and also affect convergence**. When using, appropriate choices need to be made based on specific scenarios.

Reference:

[1] Kingma D ,  Ba J . Adam: A Method for Stochastic Optimization[J]. Computer Science, 2014.
