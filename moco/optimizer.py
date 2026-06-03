# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch


class LARS(torch.optim.Optimizer):
    """
    LARS optimizer, no rate scaling or weight decay for parameters <= 1D.
    """

    def __init__(
        self, params, lr=0, weight_decay=0, momentum=0.9, trust_coefficient=0.001
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            trust_coefficient=trust_coefficient,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                # 参数梯度
                dp = p.grad

                if dp is None:
                    continue
                # if not normalization gamma/beta or bias
                if p.ndim > 1:
                    dp = dp.add(p, alpha=g["weight_decay"])
                    # 标量，L2正则化
                    param_norm = torch.norm(p)
                    # 梯度L2范数
                    update_norm = torch.norm(dp)
                    # 标量
                    one = torch.ones_like(param_norm)
                    # 自适应缩放，根据参数大小和梯度信号大小确定缩放比例
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0,
                            (g["trust_coefficient"] * param_norm / update_norm),
                            one,
                        ),
                        one,
                    )
                    # 缩放后的梯度
                    dp = dp.mul(q)
                # 参数状态
                param_state = self.state[p]
                # 动量缓冲区：过去更新的梯度
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                # 过去平均梯度*动量系数+当前梯度
                mu.mul_(g["momentum"]).add_(dp)
                # 梯度更新
                p.add_(mu, alpha=-g["lr"])
