from copy import deepcopy

import numpy as np


def test_func(x):
    return np.power(x[0] - 10, 2) + np.power(x[1] - 20, 2)


class Border_alarmer:
    def __init__(self):
        self.variable_range = None

    def __call__(self, x, step):
        if self.variable_range is not None:
            x = x + step  # 非原址操作不可自加
            x_ = np.maximum(np.minimum(x, self.variable_range[:, 1]), self.variable_range[:, 0])
            step = x - x_
            step[step != 0] = 1
            return 1 - step
        else:
            return None


class Optimizer_Adam:
    def __init__(self, variable_shape, lr=1e-3, integrate=1e-4):
        self.lr = lr
        self.eps = 1e-9
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.integrate = integrate
        self.variable_shape = variable_shape
        # ------------- auto ------------- #
        # schedule
        self.epoch = 0
        self.gamma = 1e-1
        self.milestone = np.empty(0, dtype=int)
        # calculate
        self.aim_function = None
        self.x = np.full(variable_shape, np.nan, dtype=np.float32)
        self.grad = np.zeros(variable_shape, dtype=np.float32)
        self.grad_smooth = np.zeros(variable_shape, dtype=np.float32)
        self.velocity_smooth = np.zeros(variable_shape, dtype=np.float32)
        self.border_alarmer = Border_alarmer()
        # log
        self.loss = np.full(variable_shape[0], np.nan, dtype=np.float32)
        self.loss_log = np.empty((0, variable_shape[0]), dtype=np.float32)
        self.min_loss = np.full(variable_shape[0], np.inf, dtype=np.float32)
        self.best_solve = np.full(variable_shape, np.nan, dtype=np.float32)

    def reinitialize(self):
        self.epoch = 0
        self.grad = np.zeros(self.variable_shape, dtype=np.float32)
        self.grad_smooth = np.zeros(self.variable_shape, dtype=np.float32)
        self.velocity_smooth = np.zeros(self.variable_shape, dtype=np.float32)
        self.loss = np.full(self.variable_shape[0], np.nan, dtype=np.float32)
        self.loss_log = np.empty((0, self.variable_shape[0]), dtype=np.float32)
        self.min_loss = np.full(self.variable_shape[0], np.inf, dtype=np.float32)
        self.best_solve = np.full(self.variable_shape, np.nan, dtype=np.float32)

    def step_forward(self):
        # 计算梯度
        for j in range(self.variable_shape[1]):
            pathfinder_1 = deepcopy(self.x)  # must 自加原址操作
            pathfinder_1[:, j] -= self.integrate
            pathfinder_2 = deepcopy(self.x)  # must 自加原址操作
            pathfinder_2[:, j] += self.integrate
            # 中心差分计算导数
            for i in range(self.variable_shape[0]):
                self.grad[i, j] = (self.aim_function(pathfinder_2[i]) -
                                   self.aim_function(pathfinder_1[i])) / (2 * self.integrate)
        # 指数平滑计算
        self.grad_smooth = self.beta_1 * self.grad_smooth + (1 - self.beta_1) * self.grad
        self.velocity_smooth = self.beta_2 * self.velocity_smooth + (1 - self.beta_2) * np.power(self.grad, 2)
        # 误差修正
        self.grad_smooth = self.grad_smooth / (1 - np.power(self.beta_1, self.epoch + 1))
        self.velocity_smooth = self.velocity_smooth / (1 - np.power(self.beta_1, self.epoch + 1))
        # 衰减步进 + 保持边界
        attenuation_rate = np.power(self.gamma, np.searchsorted(self.milestone, self.epoch))
        step = -attenuation_rate * self.lr * (self.grad_smooth + self.eps) / (np.power(self.velocity_smooth, 1 / 2) + self.eps)
        step_valve = self.border_alarmer(self.x, step)
        if step_valve is not None:
            self.x += (step_valve * step)
        else:
            self.x += step
        # 更新 loss 及历史最佳结果
        for i in range(self.variable_shape[0]):
            self.loss[i] = self.aim_function(self.x[i])
        better_id = self.loss < self.min_loss
        self.min_loss = np.minimum(self.min_loss, self.loss)
        self.best_solve[better_id] = self.x[better_id]
        # 定时输出并记录loss
        if (self.epoch + 1) % 10 == 0:
            self.loss_log = np.append(self.loss_log, [self.loss], axis=0)
            print(f'check point -- {self.epoch + 1}:\nvector_x =\n{self.x}\nresult =\n{self.loss}\n')
        # 进入下个 epoch
        self.epoch += 1

    def check_lr(self):
        if self.loss_log.shape[0] >= 3:
            loss_trend = (self.loss_log[-1] - self.loss_log[-2] + self.eps) / (self.loss_log[-2] - self.loss_log[-3] + self.eps)
            if np.min(loss_trend) > 0.1:
                print(f'< warning: lr needs to be set larger -- {loss_trend} >')
        else:
            print(f'< warning: end prematurely, lr status cannot be determined >')

    def run(self, max_time=1e4, min_grade=1e-4):
        self.reinitialize()
        for n in range(int(max_time)):
            self.step_forward()
            # 梯度消失时终止
            if np.max(np.abs(self.grad)) <= min_grade:
                print(f'< warning: the gradient has disappeared -- grade = {self.grad} >')
                break
        self.check_lr()
        return deepcopy(self.best_solve)


if __name__ == '__main__':
    optimizer = Optimizer_Adam((2, 2), lr=1e1, integrate=1e-4)
    # 设置初始值
    optimizer.x = np.array([[0., 0.],  # seed 1
                            [-1., -1.]], dtype=np.float32)  # seed 2
    # 设置边界
    optimizer.border_alarmer.variable_range = np.array([[-15, 15],  # 参数 1
                                                        [-30, 30]], dtype=np.float32)  # 参数 2
    # 设置 lr 衰减
    optimizer.milestone = np.array([500, 750], dtype=np.float32)
    # 设置优化函数
    optimizer.aim_function = test_func
    # 开始解算
    solve = optimizer.run(max_time=1e3, min_grade=1e-4)
    print(f'solve:\n{solve}')
