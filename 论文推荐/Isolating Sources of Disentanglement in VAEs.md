# Isolating Sources of Disentanglement in VAEs

论文地址:<https://arxiv.org/abs/1802.04942>

![image-20190413161118241](https://ws4.sinaimg.cn/large/006tNc79ly1g212oae4nvj31940tuq8j.jpg)

#### 作者:Ricky T. Q. Chen, Xuechen Li, Roger Grosse, David Duvenaud University of Toronto, Vector Institute 

我们分解了变分下界，展示了潜变量中的全相关值(Total Correlation)的存在，并设计了β-TCVAE算法。β-TCVAE算法是一个精炼的，可以替代β-VAE来完成解纠缠（Disentanglement）表示算法，在训练过程中不需要额外的参数。我们并进一步提出了一种不需要分类器的解纠缠度量方法，叫做互信息间隔（Mutual Information Gap）。我们展示了大量的和高质量的实验，使用我们的模型在一些限制条件和非线性条件下的设置中，证明了全相关和解纠缠之间重要的关系。

