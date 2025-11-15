Project1 说明文档
本项目包含三个主要任务：Task1、Task2 与 Task3。

Task1：基础分割模型（Baseline UNet）
目标：
使用 domain1 的训练数据训练 UNet，并在 domain1–5 上评估跨域分割性能。
主要流程：
加载 FAZ_h5 数据
UNet 模型训练
多域测试（Dice / HD95 / ASSD）
涉及文件与作用：
dataset_faz.py
读取 FAZ_h5 数据集（imgs、masks），并统一处理为 256×256 的灰度图像。支持 domain 切换与 train/test 切分。
unet.py
实现标准 2D UNet 模型，包括下采样、上采样、跳跃连接等结构。
metrics.py
包含分割任务使用的度量函数：Dice 系数、Dice Loss、HD95、ASSD 等。
train_task1.py
使用 domain1 的训练集训练 UNet，并保存最佳模型权重。
eval_task1.py
加载 Task1 模型，对 domain1–5 的测试集分别评估性能。

Task2：领域可视化与风格迁移
目标：
分析 domain gap，执行两种 domain1→domain3 的风格迁移（FedDG、CycleGAN），并评估迁移后的图像差异。
主要内容：
使用 FFT 分析不同域图像的频域特征
使用 UMAP 可视化 domain1–5 的特征分布
使用 FedDG 进行频域风格迁移
使用 CycleGAN 进行深度学习风格迁移
使用特征距离检验迁移后图像与目标域的相似度
涉及文件与作用：
fft_visualize.py
对单张图像进行 2D FFT，展示原图、幅度谱、相位谱。
fft_batch.py
批量处理 domain1–5 的图像，生成 FFT 结果并保存。
fed_dg.py
实现 FedDG（低频替换）风格迁移，将 domain1 图像迁移为接近 domain3 风格。
cyclegan_train.py
轻量版 CycleGAN 实现，用于训练 domain1→domain3 的风格映射模型。
cyclegan_infer.py
使用训练好的 CycleGAN 模型生成迁移后的图像。
tsne_vis.py
使用 UMAP 方法可视化 domain1–5 的特征分布，展示 domain gap。
eval_domain_shift.py
提取原图、FedDG 图像、CycleGAN 图像与 domain3 的 UNet 编码特征，并计算它们之间的特征空间距离，量化迁移效果。

Task3：分割模型与风格迁移结合（Domain Generalization）
目标：
利用 domain1 的标签和多风格图像（原图 + FedDG + CycleGAN）重新训练 UNet，以提升模型跨域鲁棒性，并对 domain1–5 进行再次评估。
涉及文件与作用：
dataset_task3.py
为 Task3 构建的数据集类。每个样本随机从三类图像来源中选择一种（domain1 原图、FedDG 图像、CycleGAN 图像），但标签始终使用 domain1 的 mask，符合“不使用 domain2–5 标签”的要求。
train_task3.py
使用 Task3 数据集重新训练一个增强版 UNet，并保存最佳模型权重。
eval_task3.py
对 domain1–5 的测试集再次评估 Task3 模型，并与 Task1 作指标对比。