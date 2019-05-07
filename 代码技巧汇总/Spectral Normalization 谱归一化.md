# Spectral Normalization 谱归一化



本文主要介绍谱归一化这项技术，详细论文参考下面的链接。

Spectral Normalization For GANsarxiv.org

本文主要对论文中的基础知识和遗漏的细节做出补充，以便于更好地理解谱归一化。部分内容参考并整合了如下两篇博客：Spectral Normalization Explainedchristiancosgrove.com![图标](https://ws4.sinaimg.cn/large/006tNc79ly1g2sjxo7xhnj303c03cq2z.jpg)

GAN 的谱归一化(Spectral Norm)和矩阵的奇异值分解(Singular Value Decompostion)kaiz.xyz![图标](https://ws1.sinaimg.cn/large/006tNc79ly1g2sjxn8zeej303c03cq2s.jpg)

## **Lipschitz continuity**

在 GAN 中，假设我们有一个判别器 ![D: I\rightarrow\mathbb{R}](https://www.zhihu.com/equation?tex=D%3A+I\rightarrow\mathbb{R}) ，其中 ![I](https://www.zhihu.com/equation?tex=I) 是图像空间。如果判别器是 K-Lipschitz continuous 的，那么对图像空间中的任意 x 和 y，有：

![||D(https://www.zhihu.com/equation?tex=||D(x)-D(y)||\leq+K||x-y||\tag{1})-D(y)||\leq K||x-y||\tag{1}](https://www.zhihu.com/equation?tex=%7C%7CD%28x%29-D%28y%29%7C%7C%5Cleq+K%7C%7Cx-y%7C%7C%5Ctag%7B1%7D)

其中 ![||\cdot||](https://www.zhihu.com/equation?tex=||\cdot||) 为 ![L_2](https://www.zhihu.com/equation?tex=L_2) norm，如果 K 取到最小值，那么 K 被称为 Lipschitz constant。

直观地来说，Lipschitz 条件限制了函数变化的剧烈程度，即函数的梯度。在一维空间中，很容易看出 y=sin(x) 是 1-Lipschitz的，它的最大斜率是 1。

那么，为什么要使判别器函数具有 Lipschitz continuity 呢？Wasserstein GAN 提出了用 wasserstein 距离取代之前的 KL 散度或者 JS 散度，作为 GAN 判别器的损失函数：

![W(https://www.zhihu.com/equation?tex=W(P_r%2C+P_g)%3D\inf_{\gamma\sim\prod(P_r%2C+P_g)}\mathbb{E}_{(x%2Cy)\sim\gamma}||x-y||\tag{2})=\inf_{\gamma\sim\prod(P_r, P_g)}\mathbb{E}_{(x,y)\sim\gamma}||x-y||\tag{2}](https://www.zhihu.com/equation?tex=W%28P_r%2C+P_g%29%3D%5Cinf_%7B%5Cgamma%5Csim%5Cprod%28P_r%2C+P_g%29%7D%5Cmathbb%7BE%7D_%7B%28x%2Cy%29%5Csim%5Cgamma%7D%7C%7Cx-y%7C%7C%5Ctag%7B2%7D)

其中 ![P_r](https://www.zhihu.com/equation?tex=P_r) ![P_g](https://www.zhihu.com/equation?tex=P_g) 分别为真实数据和生成的数据的分布函数，wasserstein 距离衡量了这两个分布函数的差异性。直观地理解，就是根据这两个分布函数分别生成一堆数据 ![x_1, x_2, ... , x_n](https://www.zhihu.com/equation?tex=x_1%2C+x_2%2C+...+%2C+x_n) 和另一堆数据 ![y_1, y_2, ... , y_n](https://www.zhihu.com/equation?tex=y_1%2C+y_2%2C+...+%2C+y_n) , 然后计算这两堆数据之间的距离。距离的算法是找到一种一一对应的配对方案 ![\gamma\sim\prod(https://www.zhihu.com/equation?tex=\gamma\sim\prod(P_r%2C+P_g))](https://www.zhihu.com/equation?tex=%5Cgamma%5Csim%5Cprod%28P_r%2C+P_g%29) ，把 ![x_i](https://www.zhihu.com/equation?tex=x_i) 移动到 ![y_j](https://www.zhihu.com/equation?tex=y_j) ，求总移动距离的最小值。由于在 GAN 中， ![P_r](https://www.zhihu.com/equation?tex=P_r) 和 ![P_g](https://www.zhihu.com/equation?tex=P_g) 都没有显式的表达式，只能是从里面不停地采样，所以不可能找到这样的 ![\gamma](https://www.zhihu.com/equation?tex=\gamma) ，无法直接优化公式 (2) 。W-GAN 的做法是根据 Kantorovich-Rubinstein duality，将公式 (2) 转化成公式 (3)，过程详见：

Wasserstein GAN and the Kantorovich-Rubinstein Dualityvincentherrmann.github.io![图标](https://ws2.sinaimg.cn/large/006tNc79ly1g2sjxos0kgj305003cjr9.jpg)

![W(https://www.zhihu.com/equation?tex=W(P_r%2C+P_g)%3D\sup_{||f||_L\leq+1}\mathbb{E}_{x\sim+P_r}[f(x)]-\mathbb{E}_{y\sim+P_g}[f(y)]\tag{3})=\sup_{||f||_L\leq 1}\mathbb{E}_{x\sim P_r}[f(x)]-\mathbb{E}_{y\sim P_g}[f(y)]\tag{3}](https://www.zhihu.com/equation?tex=W%28P_r%2C+P_g%29%3D%5Csup_%7B%7C%7Cf%7C%7C_L%5Cleq+1%7D%5Cmathbb%7BE%7D_%7Bx%5Csim+P_r%7D%5Bf%28x%29%5D-%5Cmathbb%7BE%7D_%7By%5Csim+P_g%7D%5Bf%28y%29%5D%5Ctag%7B3%7D)

其中 ![f](https://www.zhihu.com/equation?tex=f) 即为判别器函数。只有当判别器函数满足 1-Lipschitz 约束时，(2) 才能转化为 (3)。除此之外，正如上文所说，Lipschitz continuous 的函数的梯度上界被限制，因此函数更平滑，在神经网络的优化过程中，参数变化也会更稳定，不容易出现梯度爆炸，因此Lipschitz continuity 是一个很好的性质。

为了让判别器函数满足 1-Lipschitz continuity，W-GAN 和之后的 W-GAN GP 分别采用了 weight-clipping 和 gradient penalty 来约束判别器参数。这里的谱归一化，则是另一种让函数满足 1-Lipschitz continuity 的方式。

## **矩阵的 Lipschitz continuity**

众所周知，矩阵的乘法是线性映射。对线性映射来说，如果它在零点处是 K-Lipschitz 的，那么它在整个定义域上都是 K-Lipschitz 的。想象一条过零点的直线，它的斜率是固定的，只要它上面任何一点是 K-Lipschitz 的，那么它上面所有点都是 K-Lipschitz 的。因此，对矩阵 ![A:\mathbb{R}^n\rightarrow\mathbb{R}^m](https://www.zhihu.com/equation?tex=A%3A\mathbb{R}^n\rightarrow\mathbb{R}^m)来说，它满足 K-Lipschitz 的充要条件是：

![||Ax||\leq K||x||, \forall x\in \mathbb{R}^n\tag{4}](https://www.zhihu.com/equation?tex=||Ax||\leq+K||x||%2C+\forall+x\in+\mathbb{R}^n\tag{4})

对其做如下变换：

![\begin{align} ||Ax|| &\leq K||x|| \\ \langle Ax, Ax \rangle &\leq K^2\langle x, x \rangle \\ \langle (https://www.zhihu.com/equation?tex=\begin{align}+||Ax||+%26\leq+K||x||+\\+\langle+Ax%2C+Ax+\rangle+%26\leq+K^2\langle+x%2C+x+\rangle+\\+\langle+(A^TA-K^2)x%2Cx\rangle+%26\leq+0+\tag{5}+\end{align})x,x\rangle &\leq 0 \tag{5} \end{align}](https://www.zhihu.com/equation?tex=%5Cbegin%7Balign%7D+%7C%7CAx%7C%7C+%26%5Cleq+K%7C%7Cx%7C%7C+%5C%5C+%5Clangle+Ax%2C+Ax+%5Crangle+%26%5Cleq+K%5E2%5Clangle+x%2C+x+%5Crangle+%5C%5C+%5Clangle+%28A%5ETA-K%5E2%29x%2Cx%5Crangle+%26%5Cleq+0+%5Ctag%7B5%7D+%5Cend%7Balign%7D)

其中 ![\langle \cdot , \cdot \rangle](https://www.zhihu.com/equation?tex=\langle+\cdot+%2C+\cdot+\rangle) 表示两个向量的内积。由于矩阵 ![A^TA](https://www.zhihu.com/equation?tex=A^TA) 是半正定矩阵，它的所有特征值均为非负。我们假设它的特征向量构成的基底为 ![\nu_1, \nu_2, ... , \nu_n ](https://www.zhihu.com/equation?tex=\nu_1%2C+\nu_2%2C+...+%2C+\nu_n+) ，对应的特征值为 ![\lambda_1, \lambda_2, ... , \lambda_n ](https://www.zhihu.com/equation?tex=\lambda_1%2C+\lambda_2%2C+...+%2C+\lambda_n+) ，令 ![x=x_1\cdot\nu_1+x_2\cdot\nu_2+ ... +x_n\cdot \nu_n ](https://www.zhihu.com/equation?tex=x%3Dx_1\cdot\nu_1%2Bx_2\cdot\nu_2%2B+...+%2Bx_n\cdot+\nu_n+) 。那么，公式 (5) 的左半部分可以转化为：

![\begin{align} \langle(https://www.zhihu.com/equation?tex=\begin{align}+\langle(A^TA-K^2)x%2C+x\rangle+%26%3D+\langle(A^TA-K^2)\sum\limits_{i%3D1}^nx_i\cdot\nu_i%2C+\sum\limits_{j%3D1}^nx_j\cdot\nu_j\rangle\\+%26%3D\sum\limits_{i%3D1}^n\sum\limits_{j%3D1}^nx_ix_j\langle(A^TA-K^2)\nu_i%2C+\nu_j\rangle+\\+%26%3D\sum\limits_{i%3D1}^n(\lambda_i-K^2)x_i^2\leq+0+\\+%26\Rightarrow\sum\limits_{i%3D1}^n(K^2-\lambda_i)x_i^2+\geq0\tag{6}+\end{align})x, x\rangle &= \langle(A^TA-K^2)\sum\limits_{i=1}^nx_i\cdot\nu_i, \sum\limits_{j=1}^nx_j\cdot\nu_j\rangle\\ &=\sum\limits_{i=1}^n\sum\limits_{j=1}^nx_ix_j\langle(A^TA-K^2)\nu_i, \nu_j\rangle \\ &=\sum\limits_{i=1}^n(\lambda_i-K^2)x_i^2\leq 0 \\ &\Rightarrow\sum\limits_{i=1}^n(K^2-\lambda_i)x_i^2 \geq0\tag{6} \end{align}](https://www.zhihu.com/equation?tex=%5Cbegin%7Balign%7D+%5Clangle%28A%5ETA-K%5E2%29x%2C+x%5Crangle+%26%3D+%5Clangle%28A%5ETA-K%5E2%29%5Csum%5Climits_%7Bi%3D1%7D%5Enx_i%5Ccdot%5Cnu_i%2C+%5Csum%5Climits_%7Bj%3D1%7D%5Enx_j%5Ccdot%5Cnu_j%5Crangle%5C%5C+%26%3D%5Csum%5Climits_%7Bi%3D1%7D%5En%5Csum%5Climits_%7Bj%3D1%7D%5Enx_ix_j%5Clangle%28A%5ETA-K%5E2%29%5Cnu_i%2C+%5Cnu_j%5Crangle+%5C%5C+%26%3D%5Csum%5Climits_%7Bi%3D1%7D%5En%28%5Clambda_i-K%5E2%29x_i%5E2%5Cleq+0+%5C%5C+%26%5CRightarrow%5Csum%5Climits_%7Bi%3D1%7D%5En%28K%5E2-%5Clambda_i%29x_i%5E2+%5Cgeq0%5Ctag%7B6%7D+%5Cend%7Balign%7D)

要使公式 (6) 对任意 ![x_i](https://www.zhihu.com/equation?tex=x_i) 恒成立，且 ![\lambda_i](https://www.zhihu.com/equation?tex=\lambda_i) 非负，则必有 ![K^2-\lambda_i\geq0, \forall i=1,2, ... , n](https://www.zhihu.com/equation?tex=K^2-\lambda_i\geq0%2C+\forall+i%3D1%2C2%2C+...+%2C+n) 。若 ![\lambda_1](https://www.zhihu.com/equation?tex=\lambda_1)为![A^TA](https://www.zhihu.com/equation?tex=A^TA)最大的特征值，只需要满足 ![K\geq\sqrt{\lambda_1}](https://www.zhihu.com/equation?tex=K\geq\sqrt{\lambda_1}) ，这里 ![\sqrt{\lambda_1}](https://www.zhihu.com/equation?tex=\sqrt{\lambda_1}) 即为矩阵 ![A](https://www.zhihu.com/equation?tex=A) 的 spectral norm。

综上所述，映射 ![A:\mathbb{R}^n\rightarrow\mathbb{R}^m](https://www.zhihu.com/equation?tex=A%3A%5Cmathbb%7BR%7D%5En%5Crightarrow%5Cmathbb%7BR%7D%5Em) 满足 K-Lipschitz 连续，K 的最小值为 ![\sqrt{\lambda_1}](https://www.zhihu.com/equation?tex=%5Csqrt%7B%5Clambda_1%7D)。那么，要想让矩阵 ![A](https://www.zhihu.com/equation?tex=A) 满足 1-Lipschitz 连续，只需要在A的所有元素上同时除以 ![\sqrt{\lambda_1}](https://www.zhihu.com/equation?tex=%5Csqrt%7B%5Clambda_1%7D) 即可（观察公式 (4)左侧是线性映射）。

通过上面的讨论，我们得出了这样的结论：**矩阵 ![A](https://www.zhihu.com/equation?tex=A) 除以它的 spectral norm（ ![A^TA](https://www.zhihu.com/equation?tex=A%5ETA) 最大特征值的开根号 ![\sqrt{\lambda_1}](https://www.zhihu.com/equation?tex=%5Csqrt%7B%5Clambda_1%7D) ）可以使其具有 1-Lipschitz continuity**。

## **矩阵的奇异值分解**

上文提到的矩阵的 spectral norm 的另一个称呼是矩阵的最大**奇异值**。回顾矩阵的 SVD 分解：

矩阵 ![A:\mathbb{R}^n\rightarrow\mathbb{R}^m](https://www.zhihu.com/equation?tex=A%3A%5Cmathbb%7BR%7D%5En%5Crightarrow%5Cmathbb%7BR%7D%5Em) 存在这样的一种分解：

![A=U\Sigma V^T\tag{7}](https://www.zhihu.com/equation?tex=A%3DU\Sigma+V^T\tag{7})

其中：

- U 是一个 ![m\times m](https://www.zhihu.com/equation?tex=m\times+m) 的单位正交基矩阵
- ![\Sigma](https://www.zhihu.com/equation?tex=\Sigma) 是一个 ![m\times n](https://www.zhihu.com/equation?tex=m\times+n) 的对角阵，对角线上的元素为**奇异值**，非对角线上的元素为0
- V 是一个 ![n\times n](https://www.zhihu.com/equation?tex=n\times+n) 的单位正交基矩阵

![img](https://ws2.sinaimg.cn/large/006tNc79ly1g2sjxpm2uvj30k006mmx5.jpg)SVD 分解

（上图来自：[奇异值分解(SVD)原理与在降维中的应用 - 刘建平Pinard - 博客园](https://link.zhihu.com/?target=https%3A//www.cnblogs.com/pinard/p/6251584.html)）

由于 U 和 V 都是单位正交基，因此可以把矩阵乘以向量分成三步：旋转，拉伸，旋转。一前一后的两步旋转不改变向量的模长，唯一改变向量模长的是中间的拉伸，即与 ![\Sigma](https://www.zhihu.com/equation?tex=%5CSigma) 相乘的那一步。而矩阵的 Lipschitz continuity 关心的正是矩阵对向量模长的改变，不关心旋转。因此，只需要研究中间的 ![\Sigma](https://www.zhihu.com/equation?tex=%5CSigma) 即可。而 ![\Sigma](https://www.zhihu.com/equation?tex=%5CSigma) 又是一个对角矩阵，因此，**它对向量的模长拉长的最大值，就是对角线上的元素的最大值**。也就是说，矩阵的最大奇异值即为它的 spectral norm。

根据公式 (7) ，以及 SVD 分解中 U V 都是单位正交基，单位正交基的转置乘以它本身为单位矩阵，有：

![\begin{align} A^TA &= (https://www.zhihu.com/equation?tex=\begin{align}+A^TA+%26%3D+(U\Sigma+V^T)^T(U\Sigma+V^T)\\+%26%3DV\Sigma^TU^TU\Sigma+V^T+\\+%26%3DV\Sigma^TI\Sigma+V^T+\\+%26%3DV\Sigma^2V^T\tag{8}+\end{align})^T(U\Sigma V^T)\\ &=V\Sigma^TU^TU\Sigma V^T \\ &=V\Sigma^TI\Sigma V^T \\ &=V\Sigma^2V^T\tag{8} \end{align}](https://www.zhihu.com/equation?tex=%5Cbegin%7Balign%7D+A%5ETA+%26%3D+%28U%5CSigma+V%5ET%29%5ET%28U%5CSigma+V%5ET%29%5C%5C+%26%3DV%5CSigma%5ETU%5ETU%5CSigma+V%5ET+%5C%5C+%26%3DV%5CSigma%5ETI%5CSigma+V%5ET+%5C%5C+%26%3DV%5CSigma%5E2V%5ET%5Ctag%7B8%7D+%5Cend%7Balign%7D)

因此，只需要求出![A^TA](https://www.zhihu.com/equation?tex=A%5ETA) 的最大特征值，再开根号 ，就求出了矩阵的最大奇异值，也就是矩阵的 spectral norm，和上一小节的推导殊途同归。

## **神经网络的 Spectral Normalization**

对于复合函数，我们有这样的定理：

![||g\circ f||_{Lip}\leq ||g||_{Lip}\cdot ||f||_{Lip}\tag{9}](https://www.zhihu.com/equation?tex=||g\circ+f||_{Lip}\leq+||g||_{Lip}\cdot+||f||_{Lip}\tag{9})

而多层神经网络，正是多个复合函数嵌套的操作。最常见的嵌套是：一层卷积，一层激活函数，再一层卷积，再一层激活函数，这样层层包裹起来。而激活函数通常选取的 ReLU，Leaky ReLU 都是 1-Lipschitz 的，带入到 (9) 中相乘不影响总体的 Lipschitz constant，我们只需要保证卷积的部分是 1-Lipschitz continuous 的，就可以保证整个神经网络都是 1-Lipschitz continuous 的。

而在图像上每个位置的卷积操作，正好可以看成是一个矩阵乘法。因此，我们只需要约束各层卷积核的参数 ![W](https://www.zhihu.com/equation?tex=W) ，使它是 1-Lipschitz continuous 的，就可以满足整个神经网络的 1-Lipschitz continuity。而我们已经知道，想让矩阵满足 1-Lipschitz continuous，只需要让它所有元素同时除以它的最大奇异值，或者说是它的 spectural norm。因此，下一步的问题是如何计算 ![W](https://www.zhihu.com/equation?tex=W) 的最大奇异值。

对大矩阵做 SVD 分解运算量很大，我们不希望在优化神经网络的过程中，每步都对卷积核矩阵做一次 SVD 分解。一个近似的解决方案是 **power iteration** 算法。

## **Power Iteration**

Power iteration 是用来近似计算矩阵最大的特征值（dominant eigenvalue 主特征值）和其对应的特征向量（主特征向量）的。

假设矩阵 A 是一个 ![n\times n](https://www.zhihu.com/equation?tex=n%5Ctimes+n) 的满秩的方阵，它的单位特征向量为 ![\nu_1, \nu_2, ... , \nu_n ](https://www.zhihu.com/equation?tex=%5Cnu_1%2C+%5Cnu_2%2C+...+%2C+%5Cnu_n+) ，对应的特征值为 ![\lambda_1, \lambda_2, ... , \lambda_n ](https://www.zhihu.com/equation?tex=%5Clambda_1%2C+%5Clambda_2%2C+...+%2C+%5Clambda_n+) 。那么对任意向量 ![x=x_1\cdot\nu_1+x_2\cdot\nu_2+ ... +x_n\cdot \nu_n ](https://www.zhihu.com/equation?tex=x%3Dx_1%5Ccdot%5Cnu_1%2Bx_2%5Ccdot%5Cnu_2%2B+...+%2Bx_n%5Ccdot+%5Cnu_n+) ，有：

![\begin{align} Ax &= A(https://www.zhihu.com/equation?tex=\begin{align}+Ax+%26%3D+A(x_1\cdot\nu_1%2Bx_2\cdot\nu_2%2B+...+%2Bx_n\cdot+\nu_n)+\\+%26%3Dx_1(A\nu_1)%2Bx_2(A\nu_2)%2B...%2Bx_n(A\nu_n)\\+%26%3Dx_1(\lambda_1\nu_1)%2Bx_2(\lambda_2\nu_2)%2B...%2Bx_n(\lambda_n\nu_n)\tag{10}+\end{align}) \\ &=x_1(A\nu_1)+x_2(A\nu_2)+...+x_n(A\nu_n)\\ &=x_1(\lambda_1\nu_1)+x_2(\lambda_2\nu_2)+...+x_n(\lambda_n\nu_n)\tag{10} \end{align}](https://www.zhihu.com/equation?tex=%5Cbegin%7Balign%7D+Ax+%26%3D+A%28x_1%5Ccdot%5Cnu_1%2Bx_2%5Ccdot%5Cnu_2%2B+...+%2Bx_n%5Ccdot+%5Cnu_n%29+%5C%5C+%26%3Dx_1%28A%5Cnu_1%29%2Bx_2%28A%5Cnu_2%29%2B...%2Bx_n%28A%5Cnu_n%29%5C%5C+%26%3Dx_1%28%5Clambda_1%5Cnu_1%29%2Bx_2%28%5Clambda_2%5Cnu_2%29%2B...%2Bx_n%28%5Clambda_n%5Cnu_n%29%5Ctag%7B10%7D+%5Cend%7Balign%7D)

我们经过 k 次迭代：

![\begin{align} A^k x &= x_1(https://www.zhihu.com/equation?tex=\begin{align}+A^k+x+%26%3D+x_1(\lambda_1^k\nu_1)%2Bx_2(\lambda_2^k\nu_2)%2B...%2Bx_n(\lambda_n^k\nu_n)+\\+%26%3D+\lambda_1^k\left[x_1\nu_1%2Bx_2\left(\frac{\lambda_2}{\lambda_1}\right)^k\nu_2%2B...%2Bx_n\left(\frac{\lambda_n}{\lambda_1}\right)^k\nu_n\right]\tag{11}+\end{align})+x_2(\lambda_2^k\nu_2)+...+x_n(\lambda_n^k\nu_n) \\ &= \lambda_1^k\left[x_1\nu_1+x_2\left(\frac{\lambda_2}{\lambda_1}\right)^k\nu_2+...+x_n\left(\frac{\lambda_n}{\lambda_1}\right)^k\nu_n\right]\tag{11} \end{align}](https://www.zhihu.com/equation?tex=%5Cbegin%7Balign%7D+A%5Ek+x+%26%3D+x_1%28%5Clambda_1%5Ek%5Cnu_1%29%2Bx_2%28%5Clambda_2%5Ek%5Cnu_2%29%2B...%2Bx_n%28%5Clambda_n%5Ek%5Cnu_n%29+%5C%5C+%26%3D+%5Clambda_1%5Ek%5Cleft%5Bx_1%5Cnu_1%2Bx_2%5Cleft%28%5Cfrac%7B%5Clambda_2%7D%7B%5Clambda_1%7D%5Cright%29%5Ek%5Cnu_2%2B...%2Bx_n%5Cleft%28%5Cfrac%7B%5Clambda_n%7D%7B%5Clambda_1%7D%5Cright%29%5Ek%5Cnu_n%5Cright%5D%5Ctag%7B11%7D+%5Cend%7Balign%7D)

由于 ![\lambda_1>\lambda_2>...>\lambda_n](https://www.zhihu.com/equation?tex=\lambda_1>\lambda_2>...>\lambda_n) (不考虑两个特征值相等的情况，因为太少见了)。可知，经过 k 次迭代后 ![\lim_{k\rightarrow+\infty}(https://www.zhihu.com/equation?tex=\lim_{k\rightarrow%2B\infty}(\lambda_i%2F\lambda_1)^k%3D0)^k=0](https://www.zhihu.com/equation?tex=%5Clim_%7Bk%5Crightarrow%2B%5Cinfty%7D%28%5Clambda_i%2F%5Clambda_1%29%5Ek%3D0) ( ![i\neq 1](https://www.zhihu.com/equation?tex=i\neq+1) )。因此：

![\lim_{k\to +\infty}A^kx=\lambda_1^kx_1\nu_1\tag{12}](https://www.zhihu.com/equation?tex=\lim_{k\to+%2B\infty}A^kx%3D\lambda_1^kx_1\nu_1\tag{12})

也就是说，经过 k 次迭代后，我们将得到矩阵主特征向量的线性放缩，只要把这个向量归一化，就得到了该矩阵的单位主特征向量，进而可以解出矩阵的主特征值。

而我们在神经网络中，想求的是权重矩阵 ![W](https://www.zhihu.com/equation?tex=W) 的最大奇异值，根据上面几节的推导，知道这个奇异值正是 ![W^TW](https://www.zhihu.com/equation?tex=W^TW) 最大特征值的开方 ![\sqrt{\lambda_1}](https://www.zhihu.com/equation?tex=\sqrt{\lambda_1}) 。因此，我们可以采用 power iteration 的方式求解 ![W^TW](https://www.zhihu.com/equation?tex=W^TW) 的单位主特征向量，进而求出最大特征值 ![\lambda_1](https://www.zhihu.com/equation?tex=\lambda_1) 。论文中给出的算法是这样的：

![\tilde{v}:= \frac{W^T\tilde{u}}{||W^T\tilde{u}||_2}\\ \tilde{u}:=\frac{W\tilde{v}}{||W\tilde{v}||_2}\tag{13}](https://www.zhihu.com/equation?tex=\tilde{v}%3A%3D+\frac{W^T\tilde{u}}{||W^T\tilde{u}||_2}\\+\tilde{u}%3A%3D\frac{W\tilde{v}}{||W\tilde{v}||_2}\tag{13})



如果单纯看分子，我们发现这两步合起来就是 ![\tilde{v}=W^TW\tilde{v}](https://www.zhihu.com/equation?tex=\tilde{v}%3DW^TW\tilde{v}) ，反复迭代 (13) 中上下两个式子 ，即可得到矩阵 ![W^TW](https://www.zhihu.com/equation?tex=W^TW) 的单位主特征向量 ![\tilde{v}](https://www.zhihu.com/equation?tex=\tilde{v}) 。只不过 (13) 的每算“半”步都归一化一次。其实这种归一化并不影响向量 ![\tilde{v}](https://www.zhihu.com/equation?tex=\tilde{v}) 的方向收敛到主特征向量的方向，而只影响特征向量前面的系数。而每步归一化一次的好处是，每步都可以得到单位主特征向量的近似解。

那么，知道 ![W^TW](https://www.zhihu.com/equation?tex=W%5ETW) 的单位主特征向量 ![\tilde{v}](https://www.zhihu.com/equation?tex=%5Ctilde%7Bv%7D) 后，如何求出最大特征值 ![\lambda_1](https://www.zhihu.com/equation?tex=%5Clambda_1) 呢？

![\begin{align} &W^TW\tilde{v}=\lambda_1 v , ||v||_2=1\\ &\Rightarrow\tilde{v}^TW^TW\tilde{v}=\lambda_1v^Tv=\lambda_1 \\  &\Rightarrow \langle W\tilde{v}, W\tilde{v}\rangle=\lambda_1 \\ &\Rightarrow ||W\tilde{v}||_2=\sqrt{\lambda_1}\tag{14} \end{align}](https://www.zhihu.com/equation?tex=\begin{align}+%26W^TW\tilde{v}%3D\lambda_1+v+%2C+||v||_2%3D1\\+%26\Rightarrow\tilde{v}^TW^TW\tilde{v}%3D\lambda_1v^Tv%3D\lambda_1+\\++%26\Rightarrow+\langle+W\tilde{v}%2C+W\tilde{v}\rangle%3D\lambda_1+\\+%26\Rightarrow+||W\tilde{v}||_2%3D\sqrt{\lambda_1}\tag{14}+\end{align})

而将公式 (13) 的第二个式子两边同时左乘 ![\tilde{u}^T](https://www.zhihu.com/equation?tex=\tilde{u}^T) :

![\begin{align} \tilde{u}^T\tilde{u}&=\frac{\tilde{u}^TW\tilde{v}}{||W\tilde{v}||_2}\\ 1&=\frac{\tilde{u}^TW\tilde{v}}{\sqrt{\lambda_1}}\\ \sqrt{\lambda_1}&=\tilde{u}^TW\tilde{v}\tag{15} \end{align} ](https://www.zhihu.com/equation?tex=\begin{align}+\tilde{u}^T\tilde{u}%26%3D\frac{\tilde{u}^TW\tilde{v}}{||W\tilde{v}||_2}\\+1%26%3D\frac{\tilde{u}^TW\tilde{v}}{\sqrt{\lambda_1}}\\+\sqrt{\lambda_1}%26%3D\tilde{u}^TW\tilde{v}\tag{15}+\end{align}+)

最终，(15) 即为论文中提出的权重矩阵 ![W](https://www.zhihu.com/equation?tex=W) 的 spectral norm 公式。

而在具体的代码实现过程中，可以随机初始化一个噪声向量 ![\tilde{u}_0](https://www.zhihu.com/equation?tex=\tilde{u}_0) 代入公式 (13) 。由于每次更新参数的 step size 很小，矩阵 ![W](https://www.zhihu.com/equation?tex=W) 的参数变化都很小，矩阵可以长时间维持不变。因此，可以把参数更新的 step 和求矩阵最大奇异值的 step 融合在一起，即每更新一次权重 ![W](https://www.zhihu.com/equation?tex=W) ，更新一次 ![\tilde{u}](https://www.zhihu.com/equation?tex=\tilde{u}) 和 ![\tilde{v}](https://www.zhihu.com/equation?tex=\tilde{v})，并将矩阵归一化一次（除以公式 (15) 近似算出来的 spectral norm）。

具体代码见：

christiancosgrove/pytorch-spectral-normalization-gangithub.com![图标](https://ws3.sinaimg.cn/large/006tNc79ly1g2sjxq34pkj303c03ca9v.jpg)



编辑于 2019-01-23

生成对抗网络（GAN）

计算机视觉



# 对角占优矩阵

　　对角占优矩阵是计算数学中应用非常广泛的矩阵类，它较多出现于经济价值模型和反网络系统的系数矩阵及解某些确定微分方程的数值解法中，在信息论、系统论、现代经济学、网络、算法和程序设计等众多领域都有着十分重要的应用。　　定义：n阶方阵A，如果其主对角线元素的绝对值大于同行其他元素绝对值之和，则称A是对角占优的。

 

```
如果A的每个对角元的绝对值都比所在行的非对角元的绝对值的和要大，即 |a_ii|>sum{j!=i}|a_ij| 对所有的i成立，那么称A是（行）严格对角占优阵。 如果A'是行严格对角占优阵，那么称A是列严格对角占优阵。 习惯上如果不指明哪种类型的话就认为是行对角占优。
 
严格对角占优矩阵一定正定吗？ 不一定，比如 负三阶单位矩阵
 
实对称矩阵是高等代数中一个重要的内容, 所谓定型实 对称矩阵是指正定、负定、半正定和半负定矩阵, 我们首先回 顾一下本文将用到的有关实对称矩阵的一些结论: 性质1: 一个实对称矩阵A正定的充要条件是存在可逆方 阵C, 使得A=C′C。 性质2: 一个实对称矩阵A半正定的充要条件是它的所有 主子式都大于等于零。 性质3: 一个实对称矩阵A负定( 半负定) 的充要条件是- A 为正定( 半正定) 。 性质4: n维欧氏空间中, 一组基ε1,ε2, ⋯,εn 的度量矩阵A= (aij), 其中aij=(εi,εj)为实对称矩阵, 而且矩阵A是正定的。 性质5: n维欧氏空间中, 两组基ε1,ε2, ⋯,εn 和η1 ,η2, ⋯,ηn 的度量矩阵分别为A和B, 那么A和B是合同的, 即若(η1,η2 , ⋯,ηn ) =(ε1,ε2, ⋯,εn)C, 则有B=C′AC。 本文要证明的主要定理为: 定理1: A=(aij)为n阶正定矩阵, 则有detA≤ n k=1 ∏akk
关于定型实对称矩阵的行列式的一个结论( 长江师范学院数学系, 重庆408100)杨世显  
对称正定矩阵对角线上的元素必须相同吗？ 不必须，例如所有满足对角线元素都是正数的对角矩阵都是对称正定的。
 
为什么hermite正定矩阵的模最大的元素一定位于主对角线上?
利用正定矩阵的任何主子阵正定。 如果模最大的元素A(i,j)不在对角线上，那么二阶主子阵 A(i,i) A(i,j) A(j,i) A(j,j) 不是正定的。
对称正定矩阵的绝对值最大元为什么是对角元？
正定矩阵的所有主子式大于0 则 aij=aji <= max {aii ajj} 就是说，对所有aij，在对角线上都有比他大的 所以毕在对角线上取得最大值
 
 
 
对角线元素均为0的对称矩阵，它是半正定的吗？
半正定矩阵的对角线元素是非负的，而且零矩阵是一个特殊矩阵。请问对角线元素为0一定能推出是半正定吗？
半正定，等价于所有主子式>=0 ，主对角元是一阶主子式>=0，但其他主子式不一定>=0，故不一定。
有，零矩阵就是半正定的。
 
 
正定矩阵对角线的各元素都大于0吗？
直接用正定的定义就可以了。 取x=(0,0,...,1,...,0)'，即第i个元素为1，其余为0的列向量，那么 x'Ax=a_{ii}>0。
都是非负数，可以有0，但是不能全部是0。
应该是正确的。我们可以利用数学归纳法来证明这个结论。 首先，n＝1时，是显然成立的。 假设，n＝k时成立。 则，当n＝k+1时。则考虑其一个n阶主子式，其也是正定的。其对角元的元素之和全都大于0。再考察另一个n阶主子式，则其对角元的元素也全大于零。综上知，其所有的对角元的元素都大于0。 综上知，命题得证。
 
 
正定矩阵对角线的元素aii都大于0吗？
取x为单位阵的第i列，由x'Ax>0即得。
 
纯量阵就是A=aE 其中a为常数，E为单位矩阵 正定矩阵的所有的特征值都是大于零的， 而矩阵的迹(即：主对角线元素之和)=所有特征值的和>0
 
 
对角线元素均为0的对称矩阵，它是半正定的吗？
半正定矩阵的对角线元素是非负的，而且零矩阵是一个特殊矩阵。请问对角线元素为0一定能推出是半正定吗？
线性代数中什么叫纯量？为什么正定矩阵的主对角线上的元素都大于0？
半正定，等价于所有主子式>=0 ，主对角元是一阶主子式>=0，但其他主子式不一定>=0，故不一定。
有，零矩阵就是半正定的。
 
对角线元素为0、非对角线元素大于等于0的对称矩阵，它是半正定的吗？
不一定是，最简单的，二阶矩阵，角线元素为0、非对角线元素为1。而它的行列式为-1。
 
 
 
 
 
请问，对角线元素为0、非对角线元素大于等于0的对称矩阵，它是半正定的吗？
不一定是，最简单的，二阶矩阵，角线元素为0、非对角线元素为1。而它的行列式为-1
 
 
 
为什么说半正定矩阵的行列式大于等于0?
因为半正定矩阵的特征值>=0 特别附加定义，半正定矩阵为对称矩阵 所以可以对角化(定理) A=P*B*P^-1 |A|=|B|>=0
 
```

