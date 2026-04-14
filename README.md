# rough_marking_v1_0

复杂构件粗划线/基准约束配准演示项目。

## 1. 项目说明

当前主程序入口是 `main.py`，主流程已经接入 `src/for_step_feature_fit` 的核心能力，用它来完成扫描侧特征拟合。

已经并入 `main.py` 的能力：
- STEP 解析面读取
- STEP 到 STL 的整体粗配准
- 围绕 STEP 面的局部支持区搜索
- 基于 STL 三角片的局部重拟合
- 全局三角片唯一归属
- 将扫描侧拟合结果转换为主程序可直接使用的平面/圆柱特征

没有并入 `main.py` 的内容：
- `python -m src.for_step_feature_fit.demo`
- `src/for_step_feature_fit/view.py` 里的独立调试界面
- 面向圆锥 / 球 / 环面的独立调试展示

也就是说：`main.py` 已经使用了 `for_step_feature_fit` 的核心拟合链路，但没有把这个目录里的独立调试工具整体嵌进去。

## 2. 当前项目真实依赖

以下依赖只针对当前主流程和 `for_step_feature_fit`，不包含 `src/for_stl_feature` 的历史功能。

必需依赖：
- Python `3.10`
- `numpy`
- `scipy`
- `open3d`
- `pythonocc-core`

说明：
- `open3d` 负责 STL 读写、可视化、GUI、拾取与渲染。
- `scipy` 用于最小二乘优化、旋转表示和 `cKDTree`。
- `pythonocc-core` 用于 STEP 读取与解析面提取；没有它，`main.py` 和 `for_step_feature_fit` 都无法正常工作。
- 当前主流程不需要 `PySide6`。
- 当前主流程不需要 `pybind11`、`gmp`、`mpfr`，这些属于 `for_stl_feature` 的历史构建依赖，不在本文档范围内。

## 3. 环境搭建

推荐直接使用 Conda 环境文件创建环境：

```powershell
conda env create -f environment.yml
conda activate cgal_env
```

这是当前项目最推荐的安装方式，因为：
- `pythonocc-core` 通过 `conda-forge` 安装最稳定
- `environment.yml` 已经覆盖当前主流程所需完整环境

如果你已经有自己的 Conda 环境，也可以手动安装：

```powershell
conda create -n cgal_env python=3.10 -y
conda activate cgal_env
conda install -c conda-forge numpy>=1.24 scipy>=1.10 pythonocc-core>=7.7.2 -y
pip install -r requirements.txt
```

补充说明：
- `requirements.txt` 只保留适合通过 `pip` 安装的依赖。
- 完整环境请优先使用 `environment.yml`，不要把它理解成“纯 pip 即可完整安装”。
- 当前项目主要按 Windows + Conda 环境整理，其他平台尤其 macOS 上 `pythonocc-core` 可能更难安装。

## 4. 文件说明

- `environment.yml`
  - 当前项目完整推荐环境
- `requirements.txt`
  - 当前项目可通过 `pip` 安装的补充依赖
- `config.py`
  - 输入数据路径和主流程参数配置

## 5. 数据准备

`data/` 目录已被 `.gitignore` 忽略，不随仓库一起分发。拿到项目的人需要自行准备 STEP 和 STL 数据。

请修改 [`config.py`](config.py) 中的以下字段：
- `CAD_STEP_PATH`
- `SCAN_STL_PATH`

当前默认配置为：

```text
./data/2_orginal.step
./data/2.stl
```

## 6. 运行主程序

```powershell
conda activate cgal_env
python main.py
```

主流程如下：
1. 从 STEP 中提取 CAD 侧解析平面/圆柱。
2. 调用 `src.for_step_feature_fit` 对扫描 STL 做 STEP 引导的局部拟合。
3. 进入主程序现有的双视图特征配对界面。
4. 构建约束并求解刚体优化。
5. 显示最终配准结果。

## 7. 独立调试入口

如果需要单独调试 `for_step_feature_fit` 的局部拟合效果，可以运行：

```powershell
conda activate cgal_env
python -m src.for_step_feature_fit.demo
```

这个入口主要用于：
- 观察支持区选择是否合理
- 观察内点/外点划分是否合理
- 调整 `support_gap_mm`、`plane_tol_mm`、`cylinder_tol_mm`、`generic_tol_mm`

它是调试工具，不等同于主程序交付入口。

## 8. 注意事项

- 当前仓库首页文档只描述主流程和 `for_step_feature_fit` 相关依赖。
- `src/for_stl_feature` 里的历史代码不计入当前环境说明。
- 由于项目使用 Open3D GUI，请在有图形界面的本地环境中运行。
