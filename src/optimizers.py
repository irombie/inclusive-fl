# -*- coding:utf-8 -*-
"""
@Time: 2022/03/02 13:34
@Author: KI
@File: ScaffoldOptimizer.py
@Motto: Hungry And Humble
"""
from torch.optim import Optimizer


class ScaffoldOptimizer(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(ScaffoldOptimizer, self).__init__(params, defaults)

    def step(self, server_controls, client_controls):

        for group in self.param_groups:
            for p, c, ci in zip(group['params'], server_controls.values(), client_controls.values()):
                if p.grad is None:
                    continue
                dp = p.grad.data + c.data - ci.data
                p.data = p.data - dp.data * group['lr']