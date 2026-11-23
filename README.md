# pairwise-cascade-shadow-stable

## 领域

游戏、图形与多媒体

## 主题

级联阴影稳定性

## 说明

纹素对齐、切换滞回与阴影矩阵同帧捕获。

初始状态下 `graphics/cascade_shadow_stable.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
