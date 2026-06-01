# MoCo-v3 代码深度解析

> 基于 https://github.com/facebookresearch/moco-v3 的 PyTorch 实现
> 论文: *An Empirical Study of Training Self-Supervised Vision Transformers* (ICCV 2021)

---

## 1. `moco/builder.py` — MoCo 模型核心建构

**目的**: 定义 MoCo-v3 的双分支对比学习架构，包含基础编码器、动量编码器、投影头(projector)、预测头(predictor)以及对称对比损失函数。

---

### 类 `MoCo` (基类)

```python
class MoCo(nn.Module):
```

#### `__init__(self, base_encoder, dim=256, mlp_dim=4096, T=1.0)`

```python
def __init__(self, base_encoder, dim=256, mlp_dim=4096, T=1.0):
    super(MoCo, self).__init__()
    self.T = T

    # ── 双分支编码器 ─────────────────────────────────────
    # 论文 §3.1: query encoder (可训练) 和 key encoder (动量更新)
    # base_encoder 是一个 partial 函数，传入 num_classes=mlp_dim
    self.base_encoder = base_encoder(num_classes=mlp_dim)       # ← 梯度更新
    self.momentum_encoder = base_encoder(num_classes=mlp_dim)   # ← 动量+EMA更新

    # 替换/添加 MLP 投影头 + 预测头 (由子类实现)
    self._build_projector_and_predictor_mlps(dim, mlp_dim)

    # ── 初始化动量编码器 = 基础编码器，并冻结 ──────────────
    # 论文 §3.1: momentum_encoder 不接收梯度，参数通过 EMA 更新
    for param_b, param_m in zip(self.base_encoder.parameters(),
                                self.momentum_encoder.parameters()):
        param_m.data.copy_(param_b.data)      # 初始化：动量编码器 = 基础编码器
        param_m.requires_grad = False          # 冻结：不参与反向传播
```

**关键点**:
- 两个编码器**结构相同但角色不同**：`base_encoder` 通过反向传播更新，`momentum_encoder` 通过指数移动平均(EMA)更新
- 动量编码器参数 `requires_grad = False`，从不直接接收梯度
- `base_encoder` 以 partial 形式传入（如 `partial(vit_base, stop_grad_conv1=True)`），这允许传递特定于 ViT 的参数

#### `_build_mlp(num_layers, input_dim, mlp_dim, output_dim, last_bn=True)`

```python
def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
    mlp = []
    for l in range(num_layers):
        dim1 = input_dim if l == 0 else mlp_dim
        dim2 = output_dim if l == num_layers - 1 else mlp_dim
        mlp.append(nn.Linear(dim1, dim2, bias=False))
        if l < num_layers - 1:
            mlp.append(nn.BatchNorm1d(dim2))
            mlp.append(nn.ReLU(inplace=True))
        elif last_bn:
            # 仿照 SimCLR 设计：最后一层 BN 去掉 affine (无 gamma/beta)
            # 防止投影头的输出坍塌到全零
            mlp.append(nn.BatchNorm1d(dim2, affine=False))
    return nn.Sequential(*mlp)
```

#### `_update_momentum_encoder(m)` — 论文 §3.1 动量更新

```python
@torch.no_grad()   # ← 关键：不记录梯度
def _update_momentum_encoder(self, m):
    """Momentum update of the momentum encoder"""
    for param_b, param_m in zip(self.base_encoder.parameters(),
                                self.momentum_encoder.parameters()):
        # EMA: θ_k ← m * θ_k + (1 - m) * θ_q
        # 论文公式 (1): θ_k ← m·θ_k + (1-m)·θ_q
        param_m.data = param_m.data * m + param_b.data * (1. - m)
```

**与论文对应**: MoCo 系列核心机制，动量系数 `m` 在 v3 中采用余弦退火调度（从 0.99 → 1.0）

#### `contrastive_loss(q, k)` — InfoNCE 损失

```python
def contrastive_loss(self, q, k):
    # ── 1. L2 归一化 ──────────────────────────────────
    q = nn.functional.normalize(q, dim=1)   # shape: [N, dim]
    k = nn.functional.normalize(k, dim=1)   # shape: [N, dim]

    # ── 2. 跨 GPU 聚合所有 key ──────────────────────────
    # 关键：在分布式训练中，每个 GPU 只看到自己的 batch
    # concat_all_gather 收集所有 GPU 的 key，形成更大的负样本池
    k = concat_all_gather(k)                # shape: [N*world_size, dim]

    # ── 3. 计算相似度矩阵（InfoNCE logits）─────────────
    # Einstein求和: 'nc,mc->nm' 计算 q 与所有 k 的点积
    # 结果形状: [N, N*world_size]
    logits = torch.einsum('nc,mc->nm', [q, k]) / self.T

    # ── 4. 构造标签 ──────────────────────────────────
    # 正样本对是 q[i] 与 k[i] —— 但 k 已经是 all_gather 后的
    # 所以正样本索引为: rank*N + i
    N = logits.shape[0]  # batch size per GPU
    labels = (torch.arange(N, dtype=torch.long) +
              N * torch.distributed.get_rank()).cuda()

    # ── 5. 交叉熵损失 × (2 * T) ─────────────────────
    # 乘 2T 的原因：对称损失中有两项，每项用温度 T 缩放后，
    # 梯度会缩小 1/T 倍，乘回来保持梯度尺度一致性
    return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)
```

**容易误解的点**:
- `labels` 的构造假设：每个 query `q[i]` 的正样本是 `k[i + N*rank]`，因为 `concat_all_gather` 将各 GPU 的 k 按 rank 顺序拼接
- 乘以 `2*T` 不是论文中的标准操作，而是工程上的梯度尺度补偿。因为对称损失总共两项，不加这个系数梯度会太小

#### `forward(x1, x2, m)` — 对称对比学习前向传播

```python
def forward(self, x1, x2, m):
    """
    x1: 第一组增强视图  [N, 3, 224, 224]
    x2: 第二组增强视图  [N, 3, 224, 224]
    m:  当前动量系数 (float)
    """

    # ── query 分支（base_encoder + predictor）────────────
    # 论文 §3.3: 引入 predictor 是 MoCo-v3 相比 v2 的关键改进
    # （借鉴 BYOL 的 asymmetric 设计）
    q1 = self.predictor(self.base_encoder(x1))   # q1: [N, dim=256]
    q2 = self.predictor(self.base_encoder(x2))   # q2: [N, dim=256]

    with torch.no_grad():   # ← key 分支不参与反向传播
        # 先更新动量编码器参数
        self._update_momentum_encoder(m)

        # ── key 分支（momentum_encoder，无 predictor）─────
        k1 = self.momentum_encoder(x1)   # k1: [N, dim=256]
        k2 = self.momentum_encoder(x2)   # k2: [N, dim=256]

    # ── 对称对比损失 ──────────────────────────────────
    # 论文 §3.2: 对称损失是 MoCo-v3 的重要改进：
    # x1→query 对 x2→key + x2→query 对 x1→key
    return self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)
```

**v3 与 v1/v2 的核心差异**:
| 特性 | v1/v2 | v3 |
|------|-------|----|
| 对称损失 | 否（单一方向） | 是（双向） |
| Predictor | 无 | 有（借鉴 BYOL） |
| 动量调度 | 固定/手动 | 余弦退火 |
| 队列(queue) | 有（v1） | 无 |
| 编码器结构 | ResNet | ResNet + ViT |

---

### 类 `MoCo_ViT(MoCo)` — ViT 特化

```python
class MoCo_ViT(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.head.weight.shape[1]  # ViT的head层
        del self.base_encoder.head, self.momentum_encoder.head

        # ── Projector: 3 层 MLP（相比 ResNet 的 2 层） ──
        # ViT 的特征维度较大 (768 for base)，需要更深投影头
        self.base_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)

        # ── Predictor: 2 层 MLP ─────────────────────────
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim)
```

**与 ResNet 版的区别**: ViT 投影头为 3 层（ResNet 为 2 层），因为 ViT `[CLS]` token 的特征维度更高。

---

### 辅助函数 `concat_all_gather`

```python
@torch.no_grad()
def concat_all_gather(tensor):
    """跨 GPU 收集张量并拼接"""
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output
```

**作用**: 分布式训练中，每张 GPU 只处理 `batch_size / num_gpus` 个样本。`concat_all_gather` 将所有 GPU 的 key 收集到一起，使得每个 query 能与全局所有 key 计算对比损失。这样负样本池大小 = 总 batch_size，而不是单卡 batch_size。

---

## 2. `main_moco.py` — 预训练主程序

**目的**: 实现完整的自监督预训练流程，包括分布式训练、数据加载、学习率与动量调度、checkpoint 保存。

---

### 从 `main` 到 `main_worker`

```python
def main():
    args = parser.parse_args()
    # 分布式训练初始化
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # 每张 GPU 一个进程，用 torch.multiprocessing.spawn 启动
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)
```

`main_worker` 中的关键操作:
1. **分布式初始化**: `dist.init_process_group(...)`
2. **模型创建**: 根据 `--arch` 选择 `MoCo_ViT` 或 `MoCo_ResNet`
3. **学习率缩放**: `args.lr = args.lr * args.batch_size / 256` — 线性缩放规则
4. **SyncBN**: 将 BN 转为同步 BN `torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)`，这对大 batch 训练至关重要
5. **AMP 混合精度**: 使用 `torch.cuda.amp.GradScaler()` 和 `autocast`

### 数据增强 — 双视图变换

```python
# ── 论文 §4.1: BYOL 风格增强 ───────────────────────
# 两视图使用不同的增强流水线
augmentation1 = [
    transforms.RandomResizedCrop(224, scale=(args.crop_min, 1.)),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=1.0),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
]

augmentation2 = augmentation1 类似，但是:
- GaussianBlur p=0.1 (而非 1.0)
- 增加 Solarize p=0.2 (来自 BYOL)
```

**为什么两视图不对称**：借鉴 BYOL 的经验，让两个视图的增强强度略有不同，有助于学习更鲁棒的特征。

### `train` 函数

```python
def train(train_loader, model, optimizer, scaler, summary_writer, epoch, args):
    # ...
    for i, (images, _) in enumerate(train_loader):
        # 1. 学习率余弦退火调度（含 warmup）
        lr = adjust_learning_rate(optimizer, epoch + i / iters_per_epoch, args)
        # 2. 动量系数余弦退火
        if args.moco_m_cos:
            moco_m = adjust_moco_momentum(epoch + i / iters_per_epoch, args)
        # 3. AMP 前向传播
        with torch.cuda.amp.autocast(True):
            loss = model(images[0], images[1], moco_m)
        # 4. 反向传播
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

### 学习率调度 `adjust_learning_rate`

```python
def adjust_learning_rate(optimizer, epoch, args):
    if epoch < args.warmup_epochs:      # warmup 阶段：线性增长
        lr = args.lr * epoch / args.warmup_epochs
    else:                               # cosine 衰减
        lr = args.lr * 0.5 * (1. + math.cos(math.pi *
            (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    # ...
```

**warmup** 对 ViT 训练至关重要：ViT 对初始学习率敏感，直接使用高学习率容易导致训练发散。

### 动量调度 `adjust_moco_momentum` — 论文 §3.2

```python
def adjust_moco_momentum(epoch, args):
    """余弦退火调度动量系数 m: 从 moco_m 逐渐增加到 1.0"""
    m = 1. - 0.5 * (1. + math.cos(math.pi * epoch / args.epochs)) * (1. - args.moco_m)
    return m
```

当 `moco_m=0.99, epochs=300` 时:
- epoch 0: m = 0.99
- epoch 150: m ≈ 0.995
- epoch 300: m → 1.0

**工程意义**: 训练早期 m 较小，动量编码器能更快地吸收 query 编码器的新知识；后期 m 接近 1，动量编码器趋近稳定，提供更一致的 target 表示。

---

## 3. `vits.py` — ViT 模型定义与 `stop_grad_conv1`

**目的**: 定义 MoCo-v3 中使用的 ViT 变体（small/base/conv variants），以及关键创新 `stop_grad_conv1`。

---

### 类 `VisionTransformerMoCo(VisionTransformer)`

```python
class VisionTransformerMoCo(VisionTransformer):
    def __init__(self, stop_grad_conv1=False, **kwargs):
        super().__init__(**kwargs)
        # ── 固定 2D sin-cos 位置编码 ────────────────────
        self.build_2d_sincos_position_embedding()

        # ── 自定义初始化 ──────────────────────────────
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if 'qkv' in name:
                    # QKV 权重的特殊初始化
                    val = math.sqrt(6. / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                else:
                    nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.cls_token, std=1e-6)

        if isinstance(self.patch_embed, PatchEmbed):
            val = math.sqrt(6. / float(3 * reduce(mul, self.patch_embed.patch_size, 1) + self.embed_dim))
            nn.init.uniform_(self.patch_embed.proj.weight, -val, val)
            nn.init.zeros_(self.patch_embed.proj.bias)

            # ⭐ 关键创新：冻结 patch embedding 层
            if stop_grad_conv1:
                self.patch_embed.proj.weight.requires_grad = False
                self.patch_embed.proj.bias.requires_grad = False
```

### `stop_grad_conv1` — 论文 §4.3 训练稳定性改进

**动机**: MoCo-v3 作者发现，ViT 在**早期训练阶段**（前几个 epoch）会遇到**训练坍塌**问题——对比损失急剧下降但准确率也大幅下降，信息瓶颈在 patch embedding 层。

**解决方案**: 冻结第一层（patch embedding 卷积）的参数，不让其参与反向传播。

**为什么有效**: 
1. Patch embedding 层只是一个 16×16 卷积，参数量少但梯度方差大
2. 冻结它相当于为 ViT 的自监督训练提供了稳定的初始特征提取
3. 实验表明即使不冻结，随着训练进行 patch embedding 最终也会稳定，但冻结可以避免早期的坍塌风险

**使用方式**: `python main_moco.py --arch vit_base --stop-grad-conv1`

### 固定位置编码 `build_2d_sincos_position_embedding`

```python
def build_2d_sincos_position_embedding(self, temperature=10000.):
    h, w = self.patch_embed.grid_size  # e.g., 14, 14 for 224/16
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
    # 2D sin-cos 编码: 分别对行和列位置编码
    # ...
    self.pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
    self.pos_embed.requires_grad = False  # 固定不动
```

**与 DeiT/BEiT 的区别**: 使用固定的 sin-cos 位置编码（而非可学习的），增强了 ViT 的外推能力。

### ConvStem — 论文 §4.4 早期卷积帮助 ViT

```python
class ConvStem(nn.Module):
    """4层卷积堆叠替代 patch embedding，来自 Early Conv ViT (Tete et al., 2021)"""
    def forward(self, x):
        x = self.proj(x)   # 4个 stride=2 卷积 + 1个 1x1 卷积
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x
```

**效果**: 用 4 层 3×3 卷积（步长 2）替代单层 16×16 卷积，提供更平滑的下采样和更好的低级特征学习。

### 模型变体

| 函数 | embed_dim | depth | heads | 特点 |
|------|-----------|-------|-------|------|
| `vit_small` | 384 | 12 | 12 | 小型 |
| `vit_base` | 768 | 12 | 12 | 标准 |
| `vit_conv_small` | 384 | 11 | 12 | 少一层ViT+ConvStem |
| `vit_conv_base` | 768 | 11 | 12 | 少一层ViT+ConvStem |

---

## 4. `main_lincls.py` — 线性评估

**目的**: 在冻结的骨干网络上训练一个线性分类头，评估自监督预训练的质量。

---

### 关键流程

```python
def main_worker(gpu, ngpus_per_node, args):
    # 1. 创建 backbone 模型（不加 MoCo wrapper）
    if args.arch.startswith('vit'):
        model = vits.__dict__[args.arch]()
    else:
        model = torchvision_models.__dict__[args.arch]()

    # 2. 冻结所有层，只保留最后一层 fc/head 可训练
    for name, param in model.named_parameters():
        if name not in ['%s.weight' % linear_keyword, '%s.bias' % linear_keyword]:
            param.requires_grad = False

    # 3. 加载预训练权重（去掉 MoCo wrapper 前缀）
    if args.pretrained:
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # 只保留 'module.base_encoder.' 开头的键
            # 去掉 'module.base_encoder.' 前缀
            if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % linear_keyword):
                state_dict[k[len("module.base_encoder."):]] = state_dict[k]
            del state_dict[k]
        model.load_state_dict(state_dict, strict=False)
```

### 冻结 BN 的运行统计

```python
# 在 train 函数中:
model.eval()   # ← 重要！
```

**原因**: 即使 BN 层不接收梯度（被冻结），`model.train()` 模式下 BN 仍会更新 running mean/std。线性评估协议要求只能训练分类头，不能修改 backbone 的任何参数或统计量。

### Checkpoint 权重的映射

```
预训练 checkpoint 中的键: module.base_encoder.head.weight
线性评估模型中的键:      head.weight
```
映射规则：去掉 `module.base_encoder.` 前缀，保留分类头之前的权重，丢弃分类头权重（因为线性评估需要新的分类头）。

### 训练参数

| 参数 | 值 |
|------|------|
| 优化器 | SGD (momentum=0.9) |
| 学习率 | 0.1 × batch_size / 256 |
| 权重衰减 | 0 (线性分类头不需要正则化) |
| 调度 | Cosine decay, 90 epochs |
| Batch size | 1024 |

---

## 5. `moco/loader.py` — 数据加载器

**目的**: 提供双视图变换和 BYOL 风格增强。

---

### `TwoCropsTransform`

```python
class TwoCropsTransform:
    """对同一张图像应用两次随机变换，产生两个视图"""
    def __call__(self, x):
        im1 = self.base_transform1(x)   # 使用 augmentation1
        im2 = self.base_transform2(x)   # 使用 augmentation2 (含 Solarize)
        return [im1, im2]               # 返回列表，由 DataLoader 的 collate 处理
```

### `GaussianBlur` 和 `Solarize`

继承自 SimCLR 和 BYOL 的增强策略：
- `GaussianBlur`: 随机 σ ∈ [0.1, 2.0] 的高斯模糊
- `Solarize`: 反转像素值（在 BYOL 中被发现有益）

---

## 6. `moco/optimizer.py` — LARS 优化器

**目的**: LARS (Layer-wise Adaptive Rate Scaling) 优化器，适合大 batch 训练。

```python
class LARS(torch.optim.Optimizer):
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad
                if p.ndim > 1:
                    # 为权重大于1D的参数（卷积、线性层）添加权重衰减
                    dp = dp.add(p, alpha=g['weight_decay'])
                    # 逐层自适应学习率缩放
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    q = (g['trust_coefficient'] * param_norm / update_norm)
                    dp = dp.mul(q)   # 缩放梯度
                # 动量更新
                mu.mul_(g['momentum']).add_(dp)
                p.add_(mu, alpha=-g['lr'])
```

**关键设计**:
- 对偏置和 BN 参数 (ndim ≤ 1) 不进行权重衰减和 LARS 缩放
- `trust_coefficient` (默认 0.001) 控制 LARS 信任系数，防止缩放过大
- LARS 是 MoCo-v3 大 batch 训练 (4096) 的默认优化器

---

## 总结：MoCo-v3 相较于 v1/v2 的三大创新

| 创新点 | 代码位置 | 论文章节 |
|--------|----------|----------|
| **对称对比损失** | `builder.py:forward()` — 双向计算 loss | §3.2 |
| **Predictor** (借鉴 BYOL) | `builder.py:_build_projector_and_predictor_mlps()` | §3.3 |
| **ViT 冻结 patch projection** | `vits.py:VisionTransformerMoCo.__init__()` — `stop_grad_conv1` | §4.3 |

## 如何跑通代码

```bash
# 预训练 (8 GPUs)
python main_moco.py \
    --arch vit_base \
    --stop-grad-conv1 \
    --epochs 300 \
    --batch-size 256 \
    --lr 0.6 \
    --moco-m-cos \
    --crop-min 0.2 \
    /path/to/imagenet

# 线性评估
python main_lincls.py \
    --arch vit_base \
    --pretrained checkpoint_0299.pth.tar \
    --epochs 90 \
    --batch-size 1024 \
    /path/to/imagenet
```