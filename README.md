# DVI & Lyapunov 可视化模拟程序操作文档

欢迎使用 **DVI & Lyapunov 可视化模拟程序**。本程序旨在帮助用户构建电力网络模型，选择驱动节点，并计算与可视化每个节点的动态脆弱性指数（DVI）。以下是详细的操作指南，帮助您高效使用程序的各项功能。

## 目录

1. [界面概览](#1-界面概览)
2. [添加节点](#2-添加节点)
3. [移除节点](#3-移除节点)
4. [链接节点](#4-链接节点)
5. [编辑节点参数](#5-编辑节点参数)
6. [选择驱动节点](#6-选择驱动节点)
7. [设置功率谱密度 S(ω)](#7-设置功率谱密度-sω)
8. [计算DVI](#8-计算DVI)
9. [查看DVI结果](#9-查看DVI结果)
10. [画布操作](#10-画布操作)

---

## 1. 界面概览

程序主界面分为两部分：

- **左侧菜单栏**：包含节点操作按钮、驱动节点选择、功率谱密度设置及DVI计算按钮。
- **右侧画布区域**：用于显示和操作网络图，包括节点和连接线。

此外，菜单栏下方有一个 **DVI结果表格**，用于展示每对驱动节点与目标节点之间的DVI值。

![界面概览](https://i.imgur.com/placeholder.png) *(此处应有界面截图)*

---

## 2. 添加节点

**步骤：**

1. 在左侧菜单栏中，点击 **“增加节点”** 按钮。
2. 鼠标指针将变为添加模式。
3. 在画布的任意位置点击，程序将在该位置创建一个新的节点。

**注意事项：**

- 每次点击画布，都会生成一个新的节点，并自动为其分配一个唯一的名称（如“节点 1”、“节点 2”等）。

---

## 3. 移除节点

**步骤：**

1. 在左侧菜单栏中，点击 **“减少节点”** 按钮。
2. 鼠标指针将变为移除模式。
3. 在画布上点击要移除的节点，程序将删除该节点及其所有连接线。

**注意事项：**

- 移除节点后，驱动节点选择列表将自动更新，确保列表中仅包含当前网络中的节点。

---

## 4. 链接节点

**步骤：**

1. 在左侧菜单栏中，点击 **“链接节点”** 按钮。
2. 鼠标指针将变为链接模式。
3. 依次点击要连接的两个节点，程序将在它们之间创建一条连接线。

**注意事项：**

- 每对节点之间只能存在一条连接线。
- 如果尝试重复连接同一对节点，程序将提示已存在连接。

---

## 5. 编辑节点参数

**步骤：**

1. 在画布上，右键点击要编辑的节点。
2. 弹出 **“编辑节点参数”** 对话框。
3. 在对话框中，输入新的 **P_i**（功率参数）和 **alpha**（阻尼参数）值。
4. 点击 **“保存”** 按钮，参数将被更新。

**注意事项：**

- 输入的参数必须为有效的数值。
- 更新参数后，节点的DVI计算将基于最新的参数值。

---

## 6. 选择驱动节点

**步骤：**

1. 在左侧菜单栏中，找到 **“选择驱动节点”** 列表。
2. 列表中将显示当前网络中的所有节点名称。
3. 勾选一个或多个节点作为驱动节点。

**注意事项：**

- 至少需要选择一个驱动节点才能进行DVI计算。
- 驱动节点列表会根据网络的变化自动更新，确保仅包含当前存在的节点。

---

## 7. 设置功率谱密度 S(ω)

**步骤：**

1. 在左侧菜单栏中，找到 **“功率谱密度 S(ω)”** 下拉菜单。
2. 选择 **“白噪声 S(ω) = 1”** 或 **“自定义 S(ω)”**。
   - **白噪声 S(ω) = 1**：默认设置，适用于均匀驱动信号。
   - **自定义 S(ω)**：允许用户输入自定义的功率谱密度值。
3. 如果选择 **“自定义 S(ω)”**，在下方的输入框中输入所需的 **S(ω)** 值。

**注意事项：**

- 自定义 S(ω) 仅支持单一的常数值输入。
- 输入框在选择 **“自定义 S(ω)”** 后才会启用。

---

## 8. 计算DVI

**步骤：**

1. 完成网络构建、节点参数设置、驱动节点选择及功率谱密度配置后，点击左侧菜单栏中的 **“计算DVI”** 按钮。
2. 程序将基于选择的驱动节点和 S(ω) 值，计算每对驱动节点与目标节点之间的DVI值。
3. 计算完成后，DVI结果将显示在菜单栏下方的 **“DVI 结果表格”** 中，同时节点的颜色和大小也会根据累计DVI值进行调整。

**注意事项：**

- 计算过程中，程序会根据特征值和特征向量进行数值积分，确保DVI值的准确性。
- DVI值的累积用于节点的颜色和大小可视化，DVI值越高，节点颜色越红，大小越大。

---

## 9. 查看DVI结果

**步骤：**

1. 在左侧菜单栏下方，找到 **“DVI 结果表格”**。
2. 表格中将显示每对驱动节点与目标节点之间的具体DVI值。
   - 第一列为驱动节点名称。
   - 后续列为各目标节点的DVI值。
3. 节点上的 **DVI文本** 显示了该节点的累计DVI值。

**表格示例：**

| 驱动节点 | 节点1 | 节点2 | 节点3 |
|----------|-------|-------|-------|
| 节点1    | 0.50  | 0.71  | 0.50  |
| 节点3    | 0.50  | 0.71  | 0.50  |


## 10. 画布操作

**缩放与平移：**

- **缩放**：
  - 使用鼠标滚轮向上滚动以放大画布。
  - 使用鼠标滚轮向下滚动以缩小画布。
  
- **平移**：
  - 按住鼠标左键并拖动，可以移动画布视图，实现画布的平移。

**节点交互：**

- **移动节点**：
  - 点击并拖动节点，可以重新定位节点在画布上的位置。
  - 节点移动后，连接线将自动更新以保持连接。

- **查看节点名称与DVI值**：
  - 节点上方显示了节点名称和累计DVI值，便于识别和分析。

---

## 使用示例

### 构建一个简单的三节点线性网络

1. **添加节点**：
   - 点击 **“增加节点”** 按钮，分别在画布上点击三次，生成三个节点，命名为“节点 1”、“节点 2”和“节点 3”。

2. **链接节点**：
   - 点击 **“链接节点”** 按钮，依次点击“节点 1”和“节点 2”以创建连接线。
   - 再次点击 **“链接节点”** 按钮，依次点击“节点 2”和“节点 3”以创建连接线。

3. **选择驱动节点**：
   - 在 **“选择驱动节点”** 列表中，勾选 **“节点 1”** 作为驱动节点。

4. **设置功率谱密度**：
   - 选择 **“白噪声 S(ω) = 1”**，无需输入额外值。

5. **计算DVI**：
   - 点击 **“计算DVI”** 按钮。
   - 程序将计算 **节点 1** 对 **节点 1**、**节点 2** 和 **节点 3** 的DVI值，并在表格中显示。
   - 节点的颜色和大小将根据累计DVI值进行调整。

6. **查看结果**：
   - 在 **“DVI 结果表格”** 中，查看 **节点 1** 对各节点的DVI值。
  
