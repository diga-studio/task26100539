# FlyerDash PNG 导出内核

只处理图像层与嵌套组的 PNG 导出内核。不做文字排版与任务调度。

## 数据模型

```python
from flyerdash import Project, GroupNode, ImageNode

# 图像层: pixels 为调用方已解码的 RGBA8 字节 (行优先, 长度 width*height*4)。
# 源图像的一个像素对应一个局部坐标单位, 源像素 (i, j) 覆盖局部坐标
# [i, i+1) x [j, j+1), 像素中心位于半整数坐标。
img = ImageNode(pixels=rgba_bytes, width=w, height=h,
                transform=(a, b, c, d, e, f), opacity=0.8)

# 组: children 自底向上有序 (索引 0 在最底层), 无额外裁切区域。
grp = GroupNode(children=[img], transform=(a, b, c, d, e, f), opacity=0.5)

# 项目: 画布宽高以点计 (72 点 = 1 英寸), 背景透明, nodes 自底向上有序。
project = Project(width=5760.0, height=5760.0, nodes=[grp])
```

- 仿射变换 `(a, b, c, d, e, f)` 将节点局部坐标映射到父坐标:
  `px = a*lx + c*ly + e`, `py = b*lx + d*ly + f`。输入保证可逆。
- 每个节点的 `opacity` 取值 0 到 1。

## 接口

### `render(project, dpi, output_path) -> None`

将项目渲染为 RGBA PNG 文件。

- `dpi`: 仅支持 72 / 150 / 300。
- 输出宽高 = 画布宽高 × dpi ÷ 72, 向上取整。
- 按 32 行行带流式渲染并写出, 不分配整画布像素缓冲区,
  峰值内存与画布面积无关 (实测 12000×12000 导出峰值 RSS < 192 MiB)。

### `render_region(project, dpi, rect) -> bytes`

渲染指定输出像素矩形, 返回 RGBA8 字节 (行优先, 长度 `w*h*4`)。

- `rect = (x, y, width, height)`: 全局输出像素坐标, 须在画布内且不超过 512×512。
- 结果与 `render` 完整导出后截取对应区域逐字节一致;
  不生成完整导出, 也不逐像素扫描请求区域以外的画布。

### 辅助

- `output_size(project, dpi) -> (width, height)`: 计算输出像素尺寸。
- `affine_compose(outer, inner)` / `affine_invert(m)`: 仿射变换运算。

## 像素语义

- 在输出像素中心取样; 双线性插值作用于预乘 alpha 后的颜色,
  越界源像素按透明黑参与; 按 sRGB 通道数值直接运算, 不做 gamma 校正。
- source-over 合成; 组内先完成子节点合成, 再把组作为整体施加组透明度,
  嵌套组逐层遵守此规则; 组变换只改变坐标, 不引入额外重采样。
- 中间结果为 float32 预乘 alpha, 不量化为 RGBA8; 最终转回未预乘 RGBA8
  并四舍五入, 输出 alpha 为 0 时 RGB 也为 0。与实数计算语义相比
  每通道误差不超过 1。

## 约束与规模

- 支持最大 12000×12000 输出、100 个节点、8 层嵌套、每组 50 个子节点。
- 所有源图像解码后的总大小不超过 16 MiB。
- 生产路径不分配整画布像素缓冲区, 不使用临时图文件或内存映射保存
  整画布中间结果; 最终 PNG 文件不受此限制。
- 同一项目像素的结果不因请求区域的大小、位置或内部处理边界而改变。

## 测试

```bash
python3 -m pytest tests/ -q
```

- `tests/test_opacity.py`: 透明度与嵌套组递归合成 (含规范用例 85,0,170,96)。
- `tests/test_transforms.py`: 旋转/非均匀缩放/负坐标/全透明源像素,
  独立参考实现核对, 区域切分拼接一致性。
- `tests/test_large.py`: 独立进程实测 12000×12000 导出的峰值 RSS,
  验证 PNG 可解码并与 `render_region` 抽样核对。
