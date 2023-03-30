from copy import deepcopy

import numpy as np


def test_func(x):
    return (x[:, 0] - 10) ** 2 + (x[:, 1] - 20) ** 2


class Optimizer_Adam:
    def __init__(self, variable_shape, lr=1e-3, integrate=1e-4):
        self.lr = lr
        self.eps = 1e-9
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        # ------------- auto ------------- #
        # calculate
        self.aim_function = None
        self.integrate = np.empty(variable_shape[1], dtype=np.float32)
        self.integrate.fill(integrate)
        self.variable_range = np.empty((variable_shape[1], 2), dtype=np.float32)
        self.variable_range[:, 0], self.variable_range[:, 1] = -np.inf, np.inf
        self.vector_x = np.empty(variable_shape, dtype=np.float32)
        self.vector_x.fill(np.nan)
        self.grad = np.zeros(variable_shape, dtype=np.float32)
        self.grad_smooth = np.zeros(variable_shape, dtype=np.float32)
        self.velocity_smooth = np.zeros(variable_shape, dtype=np.float32)
        # log
        self.min_loss = np.inf
        self.best_solve = np.empty(variable_shape, dtype=np.float32)
        self.best_solve.fill(np.nan)
        self.loss_log = np.empty((0, variable_shape[0]), dtype=np.float32)
        self.variable_shape = variable_shape

    def reinitialize(self):
        self.grad = np.zeros(self.variable_shape, dtype=np.float32)
        self.grad_smooth = np.zeros(self.variable_shape, dtype=np.float32)
        self.velocity_smooth = np.zeros(self.variable_shape, dtype=np.float32)
        self.min_loss = np.inf
        self.best_solve = np.empty(self.variable_shape, dtype=np.float32)
        self.best_solve.fill(np.nan)
        self.loss_log = np.empty((0, self.variable_shape[0]), dtype=np.float32)

    def run(self, max_time=1e4, min_grade=1e-4):
        # 初始化起点
        if np.any(np.isnan(self.vector_x)):
            original_lower = np.maximum(-1e2, self.variable_range[:, 0])
            original_upper = np.minimum(1e2, self.variable_range[:, 1])
            self.vector_x = np.random.uniform(original_lower, original_upper, self.variable_shape)
        for n in range(int(max_time)):
            # 计算梯度(可指定各向异性的间隔)
            for j in range(self.variable_shape[1]):
                pathfinder_x_1 = deepcopy(self.vector_x)  # must 否则更改 pathfinder_x_1 时 self.vector_x 也会更改
                pathfinder_x_1[:, j] += self.integrate[j]
                pathfinder_x_2 = deepcopy(self.vector_x)  # must 否则更改 pathfinder_x_2 时 self.vector_x 也会更改
                pathfinder_x_2[:, j] -= self.integrate[j]
                # 中心差分计算导数
                self.grad[:, j] = (self.aim_function(pathfinder_x_1) - self.aim_function(pathfinder_x_2)) / \
                                  (2 * self.integrate[j])
            # 指数平滑计算
            self.grad_smooth = self.beta_1 * self.grad_smooth + (1 - self.beta_1) * self.grad
            self.velocity_smooth = self.beta_2 * self.velocity_smooth + (1 - self.beta_2) * np.power(self.grad, 2)
            # 误差修正
            self.grad_smooth = self.grad_smooth / (1 - np.power(self.beta_1, n + 1))
            self.velocity_smooth = self.velocity_smooth / (1 - np.power(self.beta_1, n + 1))
            step = -(self.lr * self.grad_smooth) / (np.power(self.velocity_smooth, 1 / 2) + self.eps)
            # 步进+保持边界
            self.vector_x += step
            # over_step_1 = (self.vector_x + step) > self.variable_range[:, 1]
            # over_step_2 = (self.vector_x + step) < self.variable_range[:, 0]
            # self.grad[over_step_1 | over_step_2] = 0  # 将撞到边界的方向梯度清零, 但假如 lr 设置过大则可能导致提前误终止迭代
            self.vector_x = np.minimum(self.vector_x, self.variable_range[:, 1])
            self.vector_x = np.maximum(self.vector_x, self.variable_range[:, 0])
            # 更新 loss
            loss = self.aim_function(self.vector_x)
            # 更新历史最佳结果
            better_id = loss < self.min_loss
            self.min_loss = np.minimum(self.min_loss, loss)
            self.best_solve[better_id] = self.vector_x[better_id]
            # 梯度消失时终止
            if np.max(np.abs(self.grad)) <= min_grade:
                print(f'< warning: the gradient has disappeared -- grade = {self.grad} >')
                break
            # 定时输出并记录loss
            if (n + 1) % (max_time // 10) == 0:
                self.loss_log = np.append(self.loss_log, [loss], axis=0)
                print(f'check point -- {n + 1}:\nvector_x =\n{self.vector_x}\nresult =\n{loss}\n')
        # 检查lr状态
        if self.loss_log.shape[0] >= 3:
            loss_trend = (self.loss_log[-1] - self.loss_log[-2]) / (self.loss_log[-2] - self.loss_log[-3] + 1e-9)
            if np.min(loss_trend) > 0.1:
                print(f'< warning: lr needs to be set larger -- {loss_trend} >')
        else:
            print(f'< warning: end prematurely, lr status cannot be determined >')
        return deepcopy(self.best_solve)


if __name__ == '__main__':
    Oa = Optimizer_Adam((2, 2), lr=5e-2, integrate=1e-4)
    # 设置初始值
    Oa.vector_x = np.array([[0., 0.],  # seed 1
                            [7., 17.]])  # seed 2
    # 设置范围
    Oa.variable_range = np.array([[-100, 100],  # 参数 1
                                  [-100, 100]])  # 参数 2
    Oa.aim_function = test_func
    solve = Oa.run(max_time=1e4, min_grade=1e-4)
    print(f'solve:\n{solve}')
