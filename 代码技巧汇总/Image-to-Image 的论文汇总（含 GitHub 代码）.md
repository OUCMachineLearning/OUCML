# Image-to-Image 的论文汇总（含 GitHub 代码）

图像生成一直是计算机视觉领域非常有意思的方向，图像到图像的变换是其中一个非常重要的应用，使用图像到图像的变换，可以完成非常多有趣的应用，可以把黑熊变成熊猫，把你的照片换成别人的表情，还可以把普通的照片变成毕加索风格的油画，自从GAN横空出世之后，这方面的应用也越来越多，下面是对这个领域的相关论文的一个整理，而且大部分都有代码！

github地址：https://github.com/ExtremeMart/image-to-image-papers

这是一个图像到图像的论文的汇总。论文按照arXiv上第一次提交时间排序。

# 监督学习

| Note            | Model                                     | Paper                                                        | Conference | paper link                                     | code link                                                    |
| :-------------- | :---------------------------------------- | :----------------------------------------------------------- | :--------- | :--------------------------------------------- | :----------------------------------------------------------- |
|                 | pix2pix                                   | Image-to-Image Translation with Conditional Adversarial Networks | CVPR 2017  | [1611.07004](https://arxiv.org/abs/1611.07004) | [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) |
| texture guided  | TextureGAN                                | TextureGAN: Controlling Deep Image Synthesis with Texture Patches | CVPR 2018  | [1706.02823](https://arxiv.org/abs/1706.02823) | [janesjanes/Pytorch-TextureGAN](https://github.com/janesjanes/Pytorch-TextureGAN) |
|                 | Contextual GAN                            | Image Generation from Sketch Constraint Using Contextual GAN | ECCV 2018  | [1711.08972](https://arxiv.org/abs/1711.08972) |                                                              |
|                 | pix2pix-HD                                | High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs | CVPR 2018  | [1711.11585](https://arxiv.org/abs/1711.11585) | [NVIDIA/pix2pixHD](https://github.com/NVIDIA/pix2pixHD)      |
| one-to-many     | BicycleGAN                                | Toward Multimodal Image-to-Image Translation                 | NIPS 2017  | [1711.11586](https://arxiv.org/abs/1711.11586) | [junyanz/BicycleGAN](https://github.com/junyanz/BicycleGAN)  |
| keypoint guided | G2-GAN                                    | Geometry Guided Adversarial Facial Expression Synthesis      | MM 2018    | [1712.03474](https://arxiv.org/abs/1712.03474) |                                                              |
|                 | contour2im                                | Smart, Sparse Contours to Represent and Edit Images          | CVPR 2018  | [1712.08232](https://arxiv.org/abs/1712.08232) | [website](https://contour2im.github.io/)                     |
| disentangle     | Cross-domain disentanglement networks     | Image-to-image translation for cross-domain disentanglement  | NIPS 2018  | [1805.09730](https://arxiv.org/abs/1805.09730) |                                                              |
| video           | vid2vid                                   | Video-to-Video Synthesis                                     | NIPS 2018  | [1808.06601](https://arxiv.org/abs/1808.06601) | [NVIDIA/vid2vid](https://github.com/NVIDIA/vid2vid)          |
| video           | pix2pix-HD + Temporal Smoothing + faceGAN | Everybody Dance Now                                          | ECCVW 2018 | [1808.07371](https://arxiv.org/abs/1808.07371) | [website](https://carolineec.github.io/everybody_dance_now/) |



# 非监督学习

**非监督学习-通用**

| Note                                | Model        | Paper                                                        | Conference            | paper link                                                   | code link                                                    |
| :---------------------------------- | :----------- | :----------------------------------------------------------- | :-------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
|                                     | DTN          | Unsupervised Cross-Domain Image Generation                   | ICLR 2017             | [1611.02200](https://arxiv.org/abs/1611.02200)               | [yunjey/domain-transfer-network (unofficial)](https://github.com/yunjey/domain-transfer-network) |
|                                     | UNIT         | Unsupervised image-to-image translation networks             | NIPS 2017             | [1703.00848](https://arxiv.org/abs/1703.00848)               | [mingyuliutw/UNIT](https://github.com/mingyuliutw/UNIT)      |
|                                     | DiscoGAN     | Learning to Discover Cross-Domain Relations with Generative Adversarial Networks | ICML 2017             | [1703.05192](https://arxiv.org/abs/1703.05192)               | [SKTBrain/DiscoGAN](https://github.com/SKTBrain/DiscoGAN)    |
|                                     | CycleGAN     | Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks | ICCV 2017             | [1703.10593](https://arxiv.org/abs/1703.10593)               | [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) |
|                                     | DualGAN      | DualGAN: Unsupervised Dual Learning for Image-to-Image Translation | ICCV 2017             | [1704.02510](https://arxiv.org/abs/1704.02510)               | [duxingren14/DualGAN](https://github.com/duxingren14/DualGAN) |
|                                     | DistanceGAN  | One-Sided Unsupervised Domain Mapping                        | NIPS 2017             | [1706.00826](https://arxiv.org/abs/1706.00826)               | [sagiebenaim/DistanceGAN](https://github.com/sagiebenaim/DistanceGAN) |
| semi supervised                     | Triangle GAN | Triangle Generative Adversarial Networks                     | NIPS 2017             | [1709.06548](https://arxiv.org/abs/1709.06548)               | [LiqunChen0606/Triangle-GAN](https://github.com/LiqunChen0606/Triangle-GAN) |
|                                     | CartoonGAN   | CartoonGAN: Generative Adversarial Networks for Photo Cartoonization | CVPR 2018             | [thecvf](http://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_CartoonGAN_Generative_Adversarial_CVPR_2018_paper.pdf) | [unofficial test](https://github.com/Yijunmaverick/CartoonGAN-Test-Pytorch-Torch), [unofficial pytorch](https://github.com/znxlwm/pytorch-CartoonGAN) |
| non-adversarial                     | NAM          | NAM: Non-Adversarial Unsupervised Domain Mapping             | ECCV 2018             | [1806.00804](https://arxiv.org/abs/1806.00804)               | [facebookresearch/nam](https://github.com/facebookresearch/nam) |
|                                     | SCAN         | Unsupervised Image-to-Image Translation with Stacked Cycle-Consistent Adversarial Networks | ECCV 2018             | [1807.08536](https://arxiv.org/abs/1807.08536)               |                                                              |
| dilated conv, improve shape deform. | GANimorph    | Improved Shape Deformation in Unsupervised Image to Image Translation | ECCV 2018             | [1808.04325](https://arxiv.org/abs/1808.04325)               | [brownvc/ganimorph](https://github.com/brownvc/ganimorph/)   |
| instance aware                      | InstaGAN     | Instance-aware image-to-image translation                    | ICLR 2019 (in review) | [openreview](https://openreview.net/pdf?id=ryxwJhC9YX)       |                                                              |



**非监督学习-注意力机制或者模板导向机制**

| Note                 | Model                | Paper                                                        | Conference | paper link                                     | code link                                                    |
| :------------------- | :------------------- | :----------------------------------------------------------- | :--------- | :--------------------------------------------- | :----------------------------------------------------------- |
| mask                 | ContrastGAN          | Generative Semantic Manipulation with Mask-Contrasting GAN   | ECCV 2018  | [1708.00315](https://arxiv.org/abs/1708.00315) |                                                              |
| attention            | DA-GAN               | DA-GAN: Instance-level Image Translation by Deep Attention Generative Adversarial Networks | CVPR 2018  | [1802.06454](https://arxiv.org/abs/1802.06454) |                                                              |
| mask / attention     | Attention-GAN        | Attention-GAN for Object Transﬁguration in Wild Images       |            | [1803.06798](https://arxiv.org/abs/1803.06798) |                                                              |
| attention            | Attention guided GAN | Unsupervised Attention-guided Image to Image Translation     | NIPS 2018  | [1806.02311](https://arxiv.org/abs/1806.02311) | [AlamiMejjati/Unsupervised-Attention-guided-Image-to-Image-Translation](https://github.com/AlamiMejjati/Unsupervised-Attention-guided-Image-to-Image-Translation) |
| attention, one-sided |                      | Show, Attend and Translate: Unsupervised Image Translation with Self-Regularization and Attention |            | [1806.06195](https://arxiv.org/abs/1806.06195) |                                                              |



**非监督学习-多对多（属性）**

| Note                     | Model                        | Paper                                                        | Conference                    | paper link                                     | code link                                                    |
| :----------------------- | :--------------------------- | :----------------------------------------------------------- | :---------------------------- | :--------------------------------------------- | :----------------------------------------------------------- |
|                          | Conditional CycleGAN         | Conditional CycleGAN for Attribute Guided Face Image Generation | ECCV 2018                     | [1705.09966](https://arxiv.org/abs/1705.09966) |                                                              |
|                          | StarGAN                      | StarGAN: Uniﬁed Generative Adversarial Networks for Multi-Domain Image-to-Image Translation | CVPR 2018                     | [1711.09020](https://arxiv.org/abs/1711.09020) | [yunjey/StarGAN](https://github.com/yunjey/StarGAN)          |
|                          | AttGAN                       | AttGAN: Facial Attribute Editing by Only Changing What You Want |                               | [1711.10678](https://arxiv.org/abs/1711.10678) | [LynnHo/AttGAN-Tensorflow](https://github.com/LynnHo/AttGAN-Tensorflow) |
|                          | ComboGAN                     | ComboGAN: Unrestrained Scalability for Image Domain Translation | CVPRW 2018                    | [1712.06909](https://arxiv.org/abs/1712.06909) | [AAnoosheh/ComboGAN](https://github.com/AAnoosheh/ComboGAN)  |
|                          | AugCGAN (Augmented CycleGAN) | Augmented CycleGAN: Learning Many-to-Many Mappings from Unpaired Data | ICML 2018                     | [1802.10151](https://arxiv.org/abs/1802.10151) | [aalmah/augmented_cyclegan](https://github.com/aalmah/augmented_cyclegan) |
| sparsely grouped dataset | SG-GAN                       | Sparsely Grouped Multi-task Generative Adversarial Networks for Facial Attribute Manipulation | MM 2018                       | [1805.07509](https://arxiv.org/abs/1805.07509) | [zhangqianhui/Sparsely-Grouped-GAN](https://github.com/zhangqianhui/Sparsely-Grouped-GAN) |
|                          | GANimation                   | GANimation: Anatomically-aware Facial Animation from a Single Image | ECCV 2018 (honorable mention) | [1807.09251](https://arxiv.org/abs/1807.09251) | [albertpumarola/GANimation](https://github.com/albertpumarola/GANimation) |



**非监督学习-分离（与/或样本导向）**

| Note                                | Model                        | Paper                                                        | Conference | paper link                                                   | code link                                                    |
| :---------------------------------- | :--------------------------- | :----------------------------------------------------------- | :--------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
|                                     | XGAN                         | XGAN: Unsupervised Image-to-Image Translation for Many-to-Many Mappings | ICML 2018  | [1711.05139](https://arxiv.org/abs/1711.05139)               | [dataset](https://google.github.io/cartoonset/)              |
|                                     | ELEGANT                      | ELEGANT: Exchanging Latent Encodings with GAN for Transferring Multiple Face Attributes | ECCV 2018  | [1803.10562](https://arxiv.org/abs/1803.10562)               | [Prinsphield/ELEGANT](https://github.com/Prinsphield/ELEGANT) |
|                                     | MUNIT                        | Multimodal Unsupervised Image-to-Image Translation           | ECCV 2018  | [1804.04732](https://arxiv.org/abs/1804.04732)               | [NVlabs/MUNIT](https://github.com/NVlabs/MUNIT)              |
|                                     | cd-GAN (Conditional DualGAN) | Conditional Image-to-Image Translation                       | CVPR 2018  | [1805.00251](https://arxiv.org/abs/1805.00251)               |                                                              |
|                                     | EG-UNIT                      | Exemplar Guided Unsupervised Image-to-Image Translation      |            | [1805.11145](https://arxiv.org/abs/1805.11145)               |                                                              |
|                                     | DRIT                         | Diverse Image-to-Image Translation via Disentangled Representations | ECCV 2018  | [1808.00948](https://arxiv.org/abs/1808.00948)               | [HsinYingLee/DRIT](https://github.com/HsinYingLee/DRIT)      |
| non-disentangle, face makeup guided | BeautyGAN                    | BeautyGAN: Instance-level Facial Makeup Transfer with Deep Generative Adversarial Network | MM 2018    | [author](https://liusi-group.com/pdf/BeautyGAN-camera-ready.pdf) |                                                              |
|                                     | UFDN                         | A Unified Feature Disentangler for Multi-Domain Image Translation and Manipulation | NIPS 2018  | [1809.01361](https://arxiv.org/abs/1809.01361)               | [Alexander-H-Liu/UFDN](https://github.com/Alexander-H-Liu/UFDN) |