manas队伍


1. 必要的代码级样例展示
无
2. 算法思路
（1）使用VQVAE作为baseline。量化效果比较好。
（2）使用MLP Mixer作为骨干网络，模型小、速度快但容量大，拟合效果好。
（3）归一化：对每个样本计算l2范数，除以范数后再进行编码，解码后的结果再乘以范数。
（4）降低过拟合：使用数据增强和adamw等。
（5）模型融合：使用EMA做自融合；对范数进行扰动，选择最优的进行编码。

3. 亮点解读
（1）矢量量化：矢量量化效果很好，码率接近限制字节数。动态标量量化可以有更高的码率，但保存的信息没有矢量量化多，解码效果不如矢量量化。
（2）MLP Mixer：网络表现力很强，模型小，速度快，比同量级的transformer和CNN收敛都快，但容易过拟合。
（3）多模型融合：在编码器中添加解码器，可以选择最优的解码方案。

4. 建模算力与环境
a. 项目运行环境

i. 项目所需的工具包/框架
numpy==1.18.5
torch=1.9.0+cu111

ii. 项目运行的资源环境
windows11
1个3090显卡

b. 项目运行办法
i. 项目的文件结构
-tasks
    -config.py : 配置和日志
    -Model_define_pytorch.py : 模型定义及提交文件
    -train.py  :训练文件
    -Model_evaluation_encoder.py: 官方提供的评估文件
    -Model_evaluation_decoder.py: 官方提供的评估文件

ii. 项目的运行步骤
运行train.py：
cd ./tasks
python train.py
在./tasks/Modelsave文件夹中含有生成的结果
运行结果的位置
./tasks/Modelsave/encoder.pth.tar 文件
./tasks/Modelsave/decoder.pth.tar 文件
5. 使用的预训练模型相关论文及模型下载链接
无
6. 其他补充资料（如有）
参考文献：
Neural Discrete Representation Learning: https://arxiv.org/abs/1711.00937.
Generating Diverse High-Fidelity Images with VQ-VAE-2.
