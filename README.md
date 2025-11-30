# Project1 — README

说明  
本仓库为 Project1 实现。该分支基于 master，为加分任务在原项目基础上扩展与整理；

分支说明
- 此 branch 用于实现并验证加分任务（domain generalization 相关增强与评估），代码和说明在 master 的基础上进行增补与重构。
- 所有实验依赖的原始训练/测试数据位于 FAZ_h5 数据路径（请参考 dataset 脚本的路径配置）。

加分任务概况
1) 探索其他图像分割模型或方法（替代/对比 UNet）
- 目的：评估更强表达能力或结构化设计是否能提升跨域性能。
- 实现脚本：
    - UNet++: unet.py
    - TransUNet / ViT-based: transunet.py
    - nnU-Net（配置化实验）: nnunet.py

2) 尝试其他领域泛化 / 领域迁移 / 风格转换方法以改进 Task2 并在 Task3 中验证性能 (*)
- 目的：扩大风格集合与迁移质量，提升 Task3 的训练样本多样性与跨域泛化。
- 实现脚本：
    - CUT（Contrastive Unpaired Translation）：cut_utils.py、cut_train.py、cut_infer.py
- 评估：使用 eval_domain_shift.py 量化源-目标特征距离，并在 Task3 的训练集/验证集上比较最终分割性能。

3) 尝试使用其他半监督学习方法完成 Task3 并获得更好的性能 (*)
- 目的：利用未标注目标域或合成风格图像的无标签信息提升模型鲁棒性。
- 实现脚本：
    - Mean Teacher: train_task3_with_MeanTeacher.py、eval_task3.py

相比Master文件更改（简要）
- dataset_task3.py — 添加排除迁移前图像的flag。
- unet.py — 加入Nested UNet 实现。
- transunet.py — 加入TransUNet 实现。
- nnunet.py — nnU-Net 实验配置与调用。
- model_zoo.py — 模型库，方便调用。
- train_task1.py / eval_task1.py — 加入其他三个图像分割模型训练与评估。
- cut_utils.py / cut_train.py / cut_infer.py — CUT相关工具函数与训练推理。
- eval_domain_shift.py — 加入CUT相关评估。
- eval_task3.py — 加入 Mean Teacher 评估。
- train_task3_with_MeanTeacher.py — Task3 Mean Teacher半监督训练入口。