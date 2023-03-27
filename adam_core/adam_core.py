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

    def forward(self, x):
        y = []
        for n, grad in enumerate(x):
            self.grad_smooth = self.beta_1 * self.grad_smooth + (1 - self.beta_1) * grad
            self.velocity_smooth = self.beta_2 * self.velocity_smooth + (1 - self.beta_2) * np.power(grad, 2)
            self.grad_smooth = self.grad_smooth / (1 - np.power(self.beta_1, n + 1))
            self.velocity_smooth = self.velocity_smooth / (1 - np.power(self.beta_1, n + 1))
            step = (self.lr * self.grad_smooth) / (np.power(self.velocity_smooth, 1 / 2) + self.eps)
            y.append(step)
        return y


def adam_core_conv(x, lr, beta1=0.9, beta2=0.999, eps=1e-9):
    beta1_list = np.logspace(0, len(x)-1, num=len(x), endpoint=True, base=beta1)
    beta2_list = np.logspace(0, len(x) - 1, num=len(x), endpoint=True, base=beta2)
    g = np.convolve(x, beta1_list, 'full')
    v = np.convolve(np.power(x, 2), beta2_list, 'full')
    y = lr * ((1 - beta1) * g) / (np.power((1 - beta2) * v, 0.5) + eps)
    return y


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    test_x = np.append(np.array([0, 0, 1]), np.zeros(80))
    adam_core = Adam_Core()
    res = adam_core.forward(test_x)
    # res = adam_core_conv(test_x, lr=5e-2)
    plt.bar(range(len(test_x)), test_x, color='r')
    plt.bar(range(len(res)), res, color='b')
    plt.savefig('1.png')
    plt.show()

