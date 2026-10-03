# FlyerDash PNG 导出内核

只处理图像层与嵌套组的 PNG 导出内核。纯 Python + NumPy，PNG 编码使用标准库
`zlib`/`struct` 实现，无其他运行时依赖。不做文字排版，不做任务调度。

## 数据模型（`flyerdash.model`）

```python
from flyerdash import Affine, GroupNode, ImageNode, Project
```

- `Project(width, height, root)`：画布宽高以**点**计（72 点 = 1 英寸），背景透明。
  `root` 为根 `GroupNode`。
- `GroupNode(transform, opacity, children)`：有序组。`children` 索引 0 在最底层，
  兄弟节点从下到上依次 source-over 合成。组无额外裁切区域。
- `ImageNode(transform, opacity, width, height, pixels)`：已解码 RGBA8 图像，
  `pixels` 为 `width*height*4` 字节的行主序字节串；源图像 1 像素 = 1 个局部坐标单位。
- `Affine(a, b, c, d, e, f)`：可逆二维仿射变换，把节点局部坐标映射到父坐标：
  `(x, y) -> (a*x + c*y + e, b*x + d*y + f)`。`Affine()` 即恒等变换。
- 每个节点的 `opacity` 取值 0 到 1。组的透明度在子节点合成完毕后整体施加，
  嵌套组逐层递归遵守此规则。

## 接口（`flyerdash.api`）

```python
from flyerdash import render, render_region, output_size, SUPPORTED_DPIS
```

### `render(project, dpi, output_path) -> None`

生成 RGBA PNG。`dpi` 支持 72、150、300（`SUPPORTED_DPIS`）。输出宽高为
`ceil(画布宽高 * dpi / 72)`。按横向条带流式处理并增量写出，不分配整画布像素
缓冲区，不使用临时图文件或内存映射。

### `render_region(project, dpi, rect) -> bytes`

`rect = (x, y, w, h)` 为输出像素坐标（全局），必须在画布内且 `w, h <= 512`。
返回 `w*h*4` 字节的 RGBA8，与完整导出对应区域**逐位一致**。只计算请求的
像素，不生成完整导出，也不扫描请求区域以外的画布。

### `output_size(project, dpi) -> (int, int)`

返回输出像素宽高。

## 像素语义

- 在输出像素中心取样；源像素中心位于半整数坐标。
- 双线性插值作用于预乘 alpha 后的颜色；越界源像素按透明黑参与采样。
- 按 sRGB 通道数值直接运算，不做 gamma 校正。
- source-over 合成；中间结果为 float32 预乘 alpha，不量化为 RGBA8。
- 最终转回未预乘 RGBA8 并四舍五入；alpha 为 0 时 RGB 也为 0。
  与实数计算语义相比每通道误差不超过 1。
- 每个输出像素独立求值：结果不随请求区域的大小、位置或内部条带边界改变。

## 资源约束

- 支持最大 12000x12000 输出、100 个节点、8 层嵌套、每组 50 个子节点。
- 源图像解码后总大小不超过 16 MiB；完整渲染进程峰值 RSS 不超过 192 MiB
  （含解释器与依赖，由 `tests/test_large.py` 实测验证）。

## 运行测试

```bash
python3 -m pytest tests/            # 全部（含约 1 分钟的 12000x12000 实测）
python3 -m pytest tests/ -k "not large"   # 仅快速测试
```

测试覆盖：

- `tests/test_compositing.py`：红蓝重叠 + 节点/组透明度 0.5 得 `(85, 0, 170, 96)`；
  第二层嵌套组验证递归透明度；组透明度不分摊到子节点；兄弟从下到上合成。
- `tests/test_transforms.py`：旋转、非均匀缩放、负坐标、带非零 RGB 的全透明
  源像素；纯 Python 独立参考实现逐像素核对；多种区域切分拼接与完整导出逐位一致。
- `tests/test_large.py`：独立进程实际生成 12000x12000 PNG（跨画布大图 +
  重叠半透明组），实测峰值 RSS，Pillow 解码验证，抽样区域与 `render_region` 核对。
