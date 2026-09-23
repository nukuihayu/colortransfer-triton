# colortransfer-triton

[English](README.md) · [安装](#快速开始) · [API](#python) · [算法](#算法) · [Benchmark](#benchmark)

ColorTransfer Triton 使用 Triton GPU 内核实现 AdaIN、Wavelet、Sliced OT、Sinkhorn
和 ColorFM-O 颜色迁移，提供 PyTorch 张量接口，无需预训练权重。

## 定性示例

![两组同构颜色迁移对比](assets/aligned.png)

**图 1.** 同构：Astronaut 冷色恢复、Chelsea 暖色迁移。[PDF](assets/aligned.pdf)

![两组异构颜色迁移对比](assets/unaligned.png)

**图 2.** 异构：Astronaut → Coffee、Coffee → Chelsea。[PDF](assets/unaligned.pdf)

## 快速开始

需要 Linux、Python 3.10+、NVIDIA GPU、CUDA PyTorch 2.6+ 和兼容的 Triton 3.2+。

```bash
python -m pip install -e .
```

### Python

```python
import torch
import colortransfer_triton as ct

source = torch.rand(1, 3, 512, 512, device="cuda")  # 替换为 RGB 图像
reference = torch.rand_like(source)
result = ct.adain(source, reference)
result = ct.transfer(source, reference, method="sliced_ot")

lut = ct.fit_colorfm(source, reference, steps=700)
result = lut(source, strength=0.8)  # 后续复用，不再拟合
```

| 张量 | 约定 |
|---|---|
| `source` | NCHW RGB，有限值 `[0,1]`，CUDA float16/bfloat16/float32 |
| `reference` | 同设备，batch 为 1 或与输入一致；除 Wavelet 外允许不同空间尺寸 |
| 输出 | 与输入同形状、同设备，连续 float32 |

图像接口仅用于推理，含梯度的输入须先 `detach()`。

## 算法

| 方法 | 原理 | 适用场景 |
|---|---|---|
| `ct.adain` | 独立匹配 RGB 各通道的均值与方差 | 全局色偏与对比度调整 |
| `ct.wavelet` | 保留输入高频，替换低频 | 用空间对齐的参考图恢复颜色 |
| `ct.sliced_ot` | 多次随机投影，对排序后的颜色进行匹配 | 超出通道统计量的颜色分布匹配 |
| `ct.sinkhorn` | 熵正则最优传输，通过重心映射生成颜色 | 联合 RGB 分布匹配，可调平滑程度 |
| `ct.partial_sinkhorn` | 只传输部分分布质量 | 参考图中部分颜色不适合强制匹配 |
| `ct.colorfm` | 拟合颜色速度场，再积分得到映射 | 通过逐图优化构建非线性颜色映射 |

Wavelet 使用不降采样的膨胀二项式滤波金字塔，并非 Haar/DWT 变换，要求参考图同尺寸且空间对齐。
其余方法不要求空间对应，匹配的是颜色分布，不区分物体语义。

ColorFM 实现无语义分割的 [ColorFM-O](https://github.com/cszn/ColorFM) 变体，
对每对图片的 RGB 采样从零训练小型 MLP：无需预训练权重或分割模型，但需要优化过程。
AdaIN、Wavelet 和 OT 方法不训练神经网络。

OT/ColorFM 使用采样拟合与 3D LUT 近似，通过 `ct.fit_lut` 或 `ct.fit_colorfm` 复用映射。
`strength` 控制混合强度，`clamp=True` 裁剪至 `[0,1]`。

### 复用颜色映射

```python
lut = ct.fit_lut(
    source,
    reference,
    method="sinkhorn",
    samples=1024,
    lut_size=33,
    epsilon=0.03,
    iterations=100,
    seed=0,
)
result = lut(source, strength=0.8)
```

拟合在采样颜色上完成，全图通过 Triton 三线性 LUT 查表处理，并融合混合与裁剪，
避免在大图上构建逐像素传输矩阵。映射可用于其他分辨率，但仍对应拟合时的颜色分布；
场景或参考图改变时应重新拟合。

### 关键参数

| 参数 | 默认值 | 作用 |
|---|---|---|
| `strength` | `1.0` | 输出混合比例 `[0,1]`；`0` 保留输入，`1` 完整迁移 |
| Wavelet `levels` | `5` | 层数越多，替换的颜色变化越偏向粗尺度 |
| OT `samples` | `1024` | 增加采样可改善分布覆盖；Sinkhorn 矩阵显存按平方增长 |
| `lut_size` | `33` | RGB 网格边长，表大小按立方增长 |
| OT `iterations` | `32` / `100` | Sliced OT 投影次数 / Sinkhorn 迭代次数 |
| Sinkhorn `epsilon` | `0.03` | 越大越平滑；较小值通常需要更多迭代 |
| Partial Sinkhorn `mass` | `0.8` | 参与传输的分布质量比例，不是输出混合强度 |
| ColorFM `steps` / `samples` | `700` / `16384` | 优化步数与颜色采样数 |
| ColorFM `ode_steps` | `5` | 构建 LUT 时的中点法积分步数 |

Sinkhorn 的收敛残差可通过 `lut.diagnostics` 中的 `marginal_error` 查看；固定迭代次数不保证收敛。

## 配图复现

```bash
python -m pip install -e ".[figures]"
python examples/color_transfer.py --suite --output assets --save-images
python examples/color_transfer.py --source source.png --reference reference.png --save-images
```

## Benchmark

```bash
python benchmarks/benchmark.py --height 2160 --width 3840 --repeats 30
python benchmarks/benchmark.py --colorfm --repeats 5
```

## 测试

```bash
python -m pip install -e ".[test,figures]"
python -m pytest -q
```

## 许可证

[MIT](LICENSE)。示例照片来自 scikit-image：Astronaut（NASA，公有领域）、Chelsea（Stefan van der Walt，CC0）、Coffee（Rachel Michetti，CC0）。
