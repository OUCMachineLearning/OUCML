### VAE系解纠缠：从VAE到βVAE，再到β-TCVAE
>> “演讲中，Bengio 以去年发布在 arXiv 的研究计划论文「有意识先验」（The consciousness prior）为主旨，重申了他与 Yann Lecun 十年前提出的解纠缠（disentangle）观念：我们应该以「关键要素需要彼此解纠缠」为约束，学习用于描述整个世界的高维表征（unconscious state）、用于推理的低维特征（conscious state），以及从高维到低维的注意力机制——这正是深度学习通往人类水平 AI 的挑战。”[Yoshua Bengio首次中国演讲：深度学习通往人类水平AI的挑战](https://t.cj.sina.com.cn/articles/view/3996876140/ee3b7d6c01900cccm)

解纠缠（Disentanglement），也叫做解耦，就是将原始数据空间中纠缠着的数据变化，变换到一个好的表征空间中，在这个空间中，不同要素的变化是可以彼此分离的。比如，人脸数据集经过编码器，在潜变量空间Z中，我们就会获得人脸是否微笑、头发颜色、方位角等信息的分离表示，我们把这些分离表示称为Factors。
解纠缠的变量通常包含可解释的语义信息，并且能够反映数据变化中的分离的因子。在生成模型中，我们就可以根据这些分布进行特定的操作，比如改变人脸宽度、添加眼镜等操作。

![591555144950_.pic_hd](https://ws4.sinaimg.cn/large/006tNc79ly1g23pd0y1mqj30pw081jx9.jpg)

解纠缠的方法不仅限于VAE，像传统的机器学习算法PCA也可以理解为一种解纠缠的方法。

#### VAE
[Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
VAE是将概率图模型的思想和AE（Auto Encoder）的结合体。

![601555144951_.pic_hd](https://ws2.sinaimg.cn/large/006tNc79ly1g23pd62fasj30du056aae.jpg)

看左图的一个概率图模型，假设有一个已知的先验分布P(z)，和能观察到的数据x。我们想要推断后验分布p(z|x),他的计算公式就是下面这个贝叶斯公式。但是当z的维度很高的时候，p(x)的计算是不可行的。这时候我们就设计一个q(z|x)去近似p(z|x)。这个度量就由KL散度给出，通过最小化KL散度，我们就能得到近似的p(z|x)。通过变换，我们就得到下面这个式子,这个式子就等于logp(x)的期望，这个式子也叫做变分下界。最大化这个式子，我们就能近似得到p(x)。
将这个概率图模型和自编码器结合，我们就得到了右图中的变分自编码器结构。变分自编码器的encoder就可以表示为近似方法q(z|x).
因为VAE的最终目的就是要拟合出x的分布，所以同样的,VAE也要去最大化这个变分下界。写成的loss的形式就是，最小化这个下界。
它包含两个值，一个是对生成的x的期望，一个是拟合的q(z|x)和先验p(z)的KL散度。
这里的q和p分别被phi 和 theta参数化。
VAE的loss:

$$
\begin{aligned} L_{\mathrm{VAE}}(\theta, \phi) &=-\log p_{\theta}(\mathbf{x})+D_{\mathrm{KL}}\left(q_{\phi}(\mathbf{z} | \mathbf{x}) \| p_{\theta}(\mathbf{z} | \mathbf{x})\right) \\ &=-\mathbb{E}_{\mathbf{z} \sim q_{\phi}(\mathbf{z} | \mathbf{x})} p_{\theta}(\mathbf{x} | \mathbf{z})+D_{\mathrm{KL}}\left(q_{\phi}(\mathbf{z} | \mathbf{x}) \| p_{\theta}(\mathbf{z})\right) \\ \theta^{*}, \phi^{*} &=\arg \min _{\theta, \phi} L_{\mathrm{VAE}} \end{aligned}
$$

#### β-VAE
[beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl)
β-VAE是VAE的变体，增强了VAE模型表示解纠缠的能力。回顾VAE，我们希望最大化生成真实数据的概率值和最小化真实和估计后验分布的KL散度。
相应的拉格朗日函数为：
$$
\begin{aligned} \mathcal{F}(\theta, \phi, \beta) &=\mathbb{E}_{\mathbf{z} \sim q_{\phi}(\mathbf{z} | \mathbf{x})} \log p_{\theta}(\mathbf{x} | \mathbf{z})-\beta\left(D_{\mathrm{KL}}\left(q_{\phi}(\mathbf{z} | \mathbf{x}) \| p_{\theta}(\mathbf{z})\right)-\delta\right) \\ &=\mathbb{E}_{\mathbf{z} \sim q_{\phi}(\mathbf{z} | \mathbf{x})} \log p_{\theta}(\mathbf{x} | \mathbf{z})-\beta D_{\mathrm{KL}}\left(q_{\phi}(\mathbf{z} | \mathbf{x}) \| p_{\theta}(\mathbf{z})\right)+\beta \delta \\ & \geq \mathbb{E}_{\mathbf{z} \sim q_{\phi}(\mathbf{z} | \mathbf{x})} \log p_{\theta}(\mathbf{x} | \mathbf{z})-\beta D_{\mathbf{K L}}\left(q_{\phi}(\mathbf{z} | \mathbf{x}) \| p_{\theta}(\mathbf{z})\right) \end{aligned}
$$
这样上式的优化问题就可以表示成最大化这个拉格朗日方程，loss函数就可以写成

$$
L_{\mathrm{BETA}}(\phi, \beta)=-\mathbb{E}_{\mathbf{z} \sim q_{\phi}(\mathbf{z} | \mathbf{x})} \log p_{\theta}(\mathbf{x} | \mathbf{z})+\beta D_{\mathrm{KL}}\left(q_{\phi}(\mathbf{z} | \mathbf{x}) \| p_{\theta}(\mathbf{z})\right)
$$
这里，拉格朗日乘子beta就是一个超参数，当beta为1的时候，它就是标准的VAE。一个较高的beta值，就使得前变量空间z表示信息的丰富度降低，但同时模型的解纠缠能力增加。所以beta可以作为表示能力和解纠缠能力之间的平衡因子。


#### β-TCVAE
[ Isolating Sources of Disentanglement in VAEs](https://arxiv.org/abs/1802.04942)
接下来就是betatcvae的工作，作者认为虽然βvae很好的提升了vae解纠缠能力，但是并没有解释清楚为什么加重惩罚KL项就可以提升解纠缠能力。
##### 分解ELBO
于是作者对变分下界中的KL散度进行了分解。首先对每个训练样本指定了唯一的索引，并且定义一个在1..N上的均匀的随机变量与训练样本相关联。这个n就代表了索引信息。q(z)就被称为聚合后验，它表示了潜变量的聚合构成。
经过作者的分解，变分下届中的KL散度就被分解成了三个子KL散度。中间的变换过程就不看了。直接看下这三个子KL散度的含义。
$$
\frac{1}{N} \sum_{n=1}^{N} \operatorname{KL}\left(q\left(z | x_{n}\right) \| p(z)\right)=\mathbb{E}_{p(n)}[\operatorname{KL}(q(z | n) \| p(z))]
$$

$$
\begin{equation}
=\mathbb{E}_{p(n)}\left[\mathbb{E}_{q(z | n)}\left[\log q(z | n)-\log p(z)+\log q(z)-\log q(z)+\log \prod_{j} q\left(z_{j}\right)-\log \prod_{j} q\left(z_{j}\right)\right]\right]
\end{equation}
$$

$$
=\mathbb{E}_{q(z, n)}\left[\log \frac{q(z | n)}{q(z)}\right]+\mathbb{E}_{q(z)}\left[\log \frac{q(z)}{\prod_{j} q\left(z_{j}\right)}\right]+\mathbb{E}_{q(z)}\left[\sum_{j} \log \frac{q\left(z_{j}\right)}{p\left(z_{j}\right)}\right]
$$

$$
=\mathbb{E}_{q(z, n)}\left[\log \frac{q(z | n) p(n)}{q(z) p(n)}\right]+\mathbb{E}_{q(z)}\left[\log \frac{q(z)}{\prod_{j} q\left(z_{j}\right)}\right]+\sum_{j} \mathbb{E}_{q(z)}\left[\log \frac{q\left(z_{j}\right)}{p\left(z_{j}\right)}\right]
$$

$$
=\mathbb{E}_{q(z, n)}\left[\log \frac{q(z | n) p(n)}{q(z) p(n)}\right]+\mathbb{E}_{q(z)}\left[\log \frac{q(z)}{\prod_{j} q\left(z_{j}\right)}\right]+\sum_{j} \mathbb{E}_{q\left(z_{j}\right) q\left(z_{\langle j} | z_{j}\right)}\left[\log \frac{q\left(z_{j}\right)}{p\left(z_{j}\right)}\right]
$$

$$
=\mathbb{E}_{q(z, n)}\left[\log \frac{q(z | n) p(n)}{q(z) p(n)}\right]+\mathbb{E}_{q(z)}\left[\log \frac{q(z)}{\prod_{j} q\left(z_{j}\right)}\right]+\sum_{j} \mathbb{E}_{q\left(z_{j}\right)}\left[\log \frac{q\left(z_{j}\right)}{p\left(z_{j}\right)}\right]
$$

![image-20190415230033849](https://ws2.sinaimg.cn/large/006tNc79ly1g23pqo8anqj30po04274y.jpg)


1. 第一项是索引编码互信息，它与数据变量和潜变量之间的互信息相关。更高的MI表示更好的解纠缠效果。有些研究认为越高的互信息值代表着越好的解纠缠效果，所以可以完全放弃对该项的惩罚。但是有些研究认为对该项惩罚也可以鼓励更好的解纠缠效果。
2. 第二项是全相关。这是信息论里的概念，它代表着潜变量空间中变量之间的相互依赖程度，是一种冗余度的测量。这里的$z_j$代表潜变量的第j维。全相关作为惩罚使得模型在分布中去找统计独立性因子，对TC更重的惩罚，代表着后验概率分布中语义的统计独立性更强，加强解纠缠的能力。当$q(z_j)$都独立时，此项为0，可以理解为最理想的解纠缠效果。
3. 第三项维度KL散度，代表每个潜变量的维度与先验潜变量的维度不能偏离太大。

通过分解ELBO，作者认为betaVAE成功的原因就在于TC项的作用。它鼓励更低的TC值，同时惩罚了index-code MI，更低的TC是βVAE表现优异的核心所在。
作者认为在解纠缠的目的中，最重要的就是对TC项进行惩罚，在后续的实验中作者会证实，只惩罚TC项，就可以得到很好的解纠缠效果。

##### 用小批量权重采样来训练
在作者的分解形式中，存在着q(z)不能直接计算出来的问题，它需要整个训练集，所以作者借鉴了重要性采样的思路，提出了批量权重采样的方法来估计q(z)。
这里作者采样M个数据，估计方程就可以写成下式：

$$
\mathbb{E}_{q(z)}[\log q(z)] \approx \frac{1}{M} \sum_{i=1}^{M}\left[\log \frac{1}{N M} \sum_{j=1}^{M} q\left(z\left(n_{i}\right) | n_{j}\right)\right]
$$
其中z(ni)是从条件分布q(z|ni)中的采样。并且这个估计值是有偏的。同时，计算它并不需要额外的超参数。

这样在计算出了分解项之后，就可以对分解后的变分下界分解项进行单独的权重赋值。这就引出了βTCVAE的算法。β-TCVAE的loss如下式:

$$
\mathcal{L}_{\beta-\mathrm{TC}} :=\mathbb{E}_{q(z | n) p(n)}[\log p(n | z)]-\alpha I_{q}(z ; n)-\beta \operatorname{KL}\left(q(z) \| \prod_{j} q\left(z_{j}\right)\right)-\gamma \sum_{j} \mathrm{KL}\left(q\left(z_{j}\right) \| p\left(z_{j}\right)\right)
$$
作者在实验后证明只调节beta，可以取得最好的结果，证实了TC项是在解纠缠中的重要性。所以作者把α和γ设置成了1，只调节β。

##### MIG(Mutual Information Gap)

β-TCVAE的文章中提出了一种新的解纠缠的度量方法-MIG互信息间隔，取代了传统的分类度量的方法。 潜变量$z_j$和一个真实因子$v_k$之间的经验互信息可以用一个联合分布$q(z_j,v_k)=\sum_{n=1}^{N}p(v_k)p(n|v_k)q(z_j|n)$。假设潜变量因子$p(v_k)$和生成过程已知，互信息就是下面这个式子。其中$Hz_j$是潜变量$z_j$的香农熵。

$$
I_{n}\left(z_{j} ; v_{k}\right)=\mathbb{E}_{q\left(z_{j}, v_{k}\right)}\left[\log \sum_{n \in \mathcal{X}_{v_{k}}} q\left(z_{j} | n\right) p\left(n | v_{k}\right)\right]+H\left(z_{j}\right)
$$
但是可能会出现一种情况，就是一个factor Vk，可能与多个Zj有很高的互信息，可是我们只希望有一个最大的互信息值，那么就使用下面的公式，每对互信息值减去第二大的互信息值，让其他的Z变小。

$$
\frac{1}{K} \sum_{k=1}^{K} \frac{1}{H\left(v_{k}\right)}\left(I_{n}\left(z_{j^{(k)}} ; v_{k}\right)-\max _{j \neq j^{(k)}} I_{n}\left(z_{j} ; v_{k}\right)\right)
$$
这个公式就是最终的度量函数。作者认为他的这种方法的优点就是轴对齐的，就是一个潜变量$z_j$之和一个真实的$v_k$相关。

#### 对比结果
在β-TCVAE中为使用了两个数据集在变分下界和解耦能力上的测验，分别是2d的形状数据集和3D的人脸数据集。这两个数据集的真实因子如图表所示。

![image-20190415230308041](https://ws3.sinaimg.cn/large/006tNc79ly1g23ptctchsj30e205q0uh.jpg)

从下面的结果图可以看出，当β值升高，互信息的惩罚在βvae中更大，但是他也阻碍了潜变量中更多有用的信息，所以通过互信息度量可以看出β-tcvae有更高的值。 这里facflow没有去了解。

![image-20190415230323401](https://ws4.sinaimg.cn/large/006tNc79ly1g23ptlyl9ej30m8084q6w.jpg)

然后是隐变量中因子之间分布的相关性和独立性对互信息的影响。作者在两个因子的分布上设置了四种不同的取值，作为模拟的噪声，测试了模型对噪声的鲁棒性，从右箱线图中可以看出β-TCVAE的优越性。

![image-20190415230329696](https://ws3.sinaimg.cn/large/006tNc79ly1g23ptq5aznj30ku08wq61.jpg)

然后是一些定性的比较，从图六中可以看到在3D椅子数据集上，βVAE只可以学到方位角、尺寸等四种属性，而βTCVEA可以多学到椅子材质、椅子腿的旋转的属性。

![image-20190415230335717](https://ws2.sinaimg.cn/large/006tNc79ly1g23pttwtv4j30mq09jtcg.jpg)

---
#### 参考资料

[From Autoencoder to Beta-VAE](https://lilianweng.github.io/lil-log/2018/08/12/from-autoencoder-to-beta-vae.html)
[Intuitively Understanding Variational Autoencoders](https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf)
[Variational autoencoders](https://www.jeremyjordan.me/variational-autoencoders/)
[β-TCVAE -Isolating Sources of Disentanglement in Variational Autoencoders](https://blog.csdn.net/qq_31239495/article/details/82702303)