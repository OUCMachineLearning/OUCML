## **为什么要使得AI System具备可解释性呢？**

如在AI医疗领域，如果无法理解及验证AI System做决策的流程机理，那么以默认相信AI判断的方式是不负责任的（Self Driving etc.），更多讨论可以看以下论文。

[Explainable Artificial Intelligence: Understanding, Visualizing and Interpreting Deep Learning Models](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1708.08296) 

![img](https://ws3.sinaimg.cn/large/006tNc79ly1g2w9c6jretj30go0b40u9.jpg)

## 重要的网站[Heatmapping](https://link.zhihu.com/?target=http%3A//Heatmapping.org)

这是一个专门整理Explainable AI的相关想法，尤其是去解释那些State of the art模型，我先展示一个可视化的Demo：[LRP Demos链接](https://link.zhihu.com/?target=https%3A//lrpserver.hhi.fraunhofer.de/handwriting-classification)

![img](https://ws4.sinaimg.cn/large/006tNc79ly1g2w9c7lm2zj30go07daat.jpg)数字辨识及重要的像素热力图显示

猫咪的分类会比较直观一下：

![img](https://ws2.sinaimg.cn/large/006tNc79ly1g2w9c6w879j30go0b40u9.jpg)看样子是学习到了猫咪的轮廓

------

**以下内容来自Heatmapping！**

## 关于LRP的介绍：

- [A Tutorial on Implementing LRP](https://link.zhihu.com/?target=http%3A//heatmapping.org/tutorial)
- [A Quick Introduction to Deep Taylor Decomposition](https://link.zhihu.com/?target=http%3A//heatmapping.org/deeptaylor) 

## 关于LRP的软件：

- [Keras Explanation Toolbox (LRP and other Methods)](https://link.zhihu.com/?target=https%3A//github.com/albermax/innvestigate)
- [GitHub project page for the LRP Toolbox](https://link.zhihu.com/?target=https%3A//github.com/sebastian-lapuschkin/lrp_toolbox)
- [TensorFlow LRP Wrapper](https://link.zhihu.com/?target=https%3A//github.com/VigneshSrinivasan10/interprettensor)
- [LRP Code for LSTM](https://link.zhihu.com/?target=https%3A//github.com/ArrasL/LRP_for_LSTM)

## 关于Deep Model可解释性的talk：

[CVPR18: Tutorial: Part 1: Interpreting and Explaining Deep Models in Computer Vision](https://link.zhihu.com/?target=https%3A//www.youtube.com/watch%3Fv%3DLtbM2phNI7I) 

- EMBC 2019 Tutorial ([Website](https://link.zhihu.com/?target=http%3A//interpretable-ml.org/embc2019tutorial/))
    Explainable ML, Medical Applications
- Northern Lights Deep Learning Workshop Keynote([Website](https://link.zhihu.com/?target=http%3A//nldl2019.org/) | [Slides](https://link.zhihu.com/?target=http%3A//heatmapping.org/slides/2019_NLDL.pdf))
    Explainable ML, Applications
- 2018 Int. Explainable AI Symposium Keynote([Website](https://link.zhihu.com/?target=http%3A//xai.unist.ac.kr/Symposium/2018/) | [Slides](https://link.zhihu.com/?target=http%3A//heatmapping.org/slides/XAI18.pdf))
    Explainable ML, Applications
- ICIP 2018 Tutorial ([Website](https://link.zhihu.com/?target=http%3A//interpretable-ml.org/icip2018tutorial/) | Slides: [1-Intro](https://link.zhihu.com/?target=http%3A//heatmapping.org/slides/2018_ICIP_1.pdf), [2-Methods](https://link.zhihu.com/?target=http%3A//heatmapping.org/slides/2018_ICIP_2.pdf), [3-Evaluation](https://link.zhihu.com/?target=http%3A//heatmapping.org/slides/2018_ICIP_3.pdf), [4-Applications](https://link.zhihu.com/?target=http%3A//heatmapping.org/slides/2018_ICIP_4.pdf))
    Explainable ML, Applications
- MICCAI 2018 Tutorial ([Website](https://link.zhihu.com/?target=http%3A//interpretable-ml.org/miccai2018tutorial/) | [Slides](https://link.zhihu.com/?target=http%3A//heatmapping.org/slides/2018_MICCAI.pdf))
    Explainable ML, Medical Applications
- Talk at Int. Workshop ML & AI 2018 ([Slides](https://link.zhihu.com/?target=http%3A//heatmapping.org/slides/2018_workshopMLAI.pdf))
    Deep Taylor Decomposition, Validating Explanations
- WCCI 2018 Keynote ([Slides](https://link.zhihu.com/?target=http%3A//heatmapping.org/slides/2018_WCCI.pdf))
    Explainable ML, LRP, Applications
- GCPR 2017 Tutorial ([Slides](https://link.zhihu.com/?target=http%3A//heatmapping.org/slides/2017_GCPR.pdf))
- ICASSP 2017 Tutorial (Slides [1-Intro](https://link.zhihu.com/?target=http%3A//heatmapping.org/slides/2017_ICASSP_1.pdf), [2-Methods](https://link.zhihu.com/?target=http%3A//heatmapping.org/slides/2017_ICASSP_2.pdf), [3-Applications](https://link.zhihu.com/?target=http%3A//heatmapping.org/slides/2017_ICASSP_3.pdf))

## Hightlight Papers

S Lapuschkin, S Wäldchen, A Binder, G Montavon, W Samek, KR Müller. [Unmasking Clever Hans Predictors and Assessing What Machines Really Learn](https://link.zhihu.com/?target=http%3A//dx.doi.org/10.1038/s41467-019-08987-4)
Nature Communications, 10:1096, 2019 [[preprint](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1902.10178) | [bibtex](https://link.zhihu.com/?target=http%3A//heatmapping.org/bibtex/lapuschkin-ncomm19.txt)]

## 入门级介绍：

- G Montavon, W Samek, KR Müller. [Methods for Interpreting and Understanding Deep Neural Networks](https://link.zhihu.com/?target=https%3A//doi.org/10.1016/j.dsp.2017.10.011)
    Digital Signal Processing, 73:1-15, 2018 [[preprint](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1706.07979) | [bibtex](https://link.zhihu.com/?target=http%3A//heatmapping.org/bibtex/montavon-dsp18.txt)]
- W Samek, T Wiegand, KR Müller. [Explainable Artificial Intelligence: Understanding, Visualizing and Interpreting Deep Learning Models](https://link.zhihu.com/?target=https%3A//www.itu.int/en/journal/001/Pages/05.aspx)
    ITU Journal: ICT Discoveries - Special Issue 1 - The Impact of AI on Communication Networks and Services, 1(1):39-48, 2018 [[preprint](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1708.08296), [bibtex](https://link.zhihu.com/?target=http%3A//heatmapping.org/bibtex/samek-itu18.txt)]

## 方法相关：

- S Bach, A Binder, G Montavon, F Klauschen, KR Müller, W Samek. [On Pixel-wise Explanations for Non-Linear Classifier Decisions by Layer-wise Relevance Propagation](https://link.zhihu.com/?target=http%3A//journals.plos.org/plosone/article%3Fid%3D10.1371/journal.pone.0130140)
    PLOS ONE, 10(7):e0130140, 2015 [[preprint](https://link.zhihu.com/?target=http%3A//iphome.hhi.de/samek/pdf/BacPLOS15.pdf), [bibtex](https://link.zhihu.com/?target=http%3A//heatmapping.org/bibtex/bach-plos15.txt)]
- G Montavon, S Lapuschkin, A Binder, W Samek, KR Müller. [Explaining NonLinear Classification Decisions with Deep Taylor Decomposition](https://link.zhihu.com/?target=http%3A//dx.doi.org/10.1016/j.patcog.2016.11.008)
    Pattern Recognition, 65:211–222, 2017 [[preprint](https://link.zhihu.com/?target=http%3A//arxiv.org/abs/1512.02479), [bibtex](https://link.zhihu.com/?target=http%3A//heatmapping.org/bibtex/montavon-pr17.txt)]
- L Arras, G Montavon, KR Müller, W Samek. [Explaining Recurrent Neural Network Predictions in Sentiment Analysis](https://link.zhihu.com/?target=http%3A//www.aclweb.org/anthology/W17-5221)
    EMNLP Workshop on Computational Approaches to Subjectivity, Sentiment & Social Media Analysis, 159-168, 2017 [[preprint](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1706.07206), [bibtex](https://link.zhihu.com/?target=http%3A//heatmapping.org/bibtex/arras-emnlp17.txt)]
- A Binder, G Montavon, S Lapuschkin, KR Müller, W Samek. [Layer-wise Relevance Propagation for Neural Networks with Local Renormalization Layers](https://link.zhihu.com/?target=http%3A//dx.doi.org/10.1007/978-3-319-44781-0_8)
    Artificial Neural Networks and Machine Learning – ICANN 2016, Part II, Lecture Notes in Computer Science, Springer-Verlag, 9887:63-71, 2016 [[preprint](https://link.zhihu.com/?target=http%3A//iphome.hhi.de/samek/pdf/BinICANN16.pdf), [bibtex](https://link.zhihu.com/?target=http%3A//heatmapping.org/bibtex/binder-icann16.txt)]
- PJ Kindermans, KT Schütt, M Alber, KR Müller, D Erhan, B Kim, S Dähne. [Learning how to explain neural networks: PatternNet and PatternAttribution](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1705.05598)
    International Conference on Learning Representations (ICLR), 2018
- L Rieger, P Chormai, G Montavon, LK Hansen, KR Müller. [Structuring Neural Networks for More Explainable Predictions](https://link.zhihu.com/?target=https%3A//doi.org/10.1007/978-3-319-98131-4_5) in Explainable and Interpretable Models in Computer Vision and Machine Learning, 115-131, Springer SSCML, 2018
- J Kauffmann, KR Müller, G Montavon. [Towards Explaining Anomalies: A Deep Taylor Decomposition of One-Class Models](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1805.06230)
    arXiv:1805.06230, 2018

## 评估解释 Evaluation of Explanation

- L Arras, A Osman, KR Müller, W Samek. [Evaluating Recurrent Neural Network Explanations](https://link.zhihu.com/?target=http%3A//arxiv.org/abs/1904.11829)
    arXiv:1904.11829, 2019
- W Samek, A Binder, G Montavon, S Bach, KR Müller. [Evaluating the Visualization of What a Deep Neural Network has Learned](https://link.zhihu.com/?target=http%3A//dx.doi.org/10.1109/TNNLS.2016.2599820)
    IEEE Transactions on Neural Networks and Learning Systems, 28(11):2660-2673, 2017 [[preprint](https://link.zhihu.com/?target=http%3A//arxiv.org/abs/1509.06321), [bibtex](https://link.zhihu.com/?target=http%3A//heatmapping.org/bibtex/samek-tnnls17.txt)]

## 关于软件的Papers:

- M Alber, S Lapuschkin, P Seegerer, M Hägele, KT Schütt, G Montavon, W Samek, KR Müller, S Dähne, PJ Kindermans [iNNvestigate neural networks!.](https://link.zhihu.com/?target=http%3A//arxiv.org/abs/1808.04260)
    arXiv:1808.04260, 2018
- S Lapuschkin, A Binder, G Montavon, KR Müller, W Samek [The Layer-wise Relevance Propagation Toolbox for Artificial Neural Networks](https://link.zhihu.com/?target=http%3A//www.jmlr.org/papers/v17/15-618.html)
    Journal of Machine Learning Research, 17(114):1−5, 2016 [[preprint](https://link.zhihu.com/?target=http%3A//iphome.hhi.de/samek/pdf/LapJMLR16.pdf), [bibtex](https://link.zhihu.com/?target=http%3A//heatmapping.org/bibtex/lapuschkin-jmlr16.txt)]

## Application of Science:

- I Sturm, S Bach, W Samek, KR Müller. [Interpretable Deep Neural Networks for Single-Trial EEG Classification](https://link.zhihu.com/?target=http%3A//dx.doi.org/10.1016/j.jneumeth.2016.10.008)
    Journal of Neuroscience Methods, 274:141–145, 2016 [[preprint](https://link.zhihu.com/?target=http%3A//arxiv.org/abs/1604.08201), [bibtex](https://link.zhihu.com/?target=http%3A//heatmapping.org/bibtex/sturm-jnm16.txt)]
- A Binder, M Bockmayr, M Hägele, S Wienert, D Heim, K Hellweg, A Stenzinger, L Parlow, J Budczies, B Goeppert, D Treue, M Kotani, M Ishii, M Dietel, A Hocke, C Denkert, KR Müller, F Klauschen. [Towards computational fluorescence microscopy: Machine learning-based integrated prediction of morphological and molecular tumor profiles](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1805.11178)
    arXiv:1805.11178, 2018
- F Horst, S Lapuschkin, W Samek, KR Müller, WI Schöllhorn. [Explaining the Unique Nature of Individual Gait Patterns with Deep Learning](https://link.zhihu.com/?target=http%3A//dx.doi.org/10.1038/s41598-019-38748-8)
    Scientific Reports, 9:2391, 2019 [[preprint](https://link.zhihu.com/?target=http%3A//arxiv.org/abs/1808.04308), [bibtex](https://link.zhihu.com/?target=http%3A//heatmapping.org/bibtex/horst-srep19.txt)]
- AW Thomas, HR Heekeren, KR Müller, W Samek. [Analyzing Neuroimaging Data Through Recurrent Deep Learning Models](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1810.09945)
    arXiv:1810.09945, 2018

## 文本上的应用：

- L Arras, F Horn, G Montavon, KR Müller, W Samek. ["What is Relevant in a Text Document?": An Interpretable Machine Learning Approach](https://link.zhihu.com/?target=http%3A//dx.doi.org/10.1371/journal.pone.0181142)
    PLOS ONE, 12(8):e0181142, 2017 [[preprint](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1612.07843), [bibtex](https://link.zhihu.com/?target=http%3A//heatmapping.org/bibtex/arras-plos17.txt)]
- L Arras, G Montavon, KR Müller, W Samek. [Explaining Recurrent Neural Network Predictions in Sentiment Analysis](https://link.zhihu.com/?target=http%3A//www.aclweb.org/anthology/W17-5221)
    EMNLP Workshop on Computational Approaches to Subjectivity, Sentiment & Social Media Analysis, 159-168, 2017 [[preprint](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1706.07206), [bibtex](https://link.zhihu.com/?target=http%3A//heatmapping.org/bibtex/arras-emnlp17.txt)]
- L Arras, F Horn, G Montavon, KR Müller, W Samek. [Explaining Predictions of Non-Linear Classifiers in NLP](https://link.zhihu.com/?target=http%3A//www.aclweb.org/anthology/W/W16/W16-1601.pdf)
    ACL Workshop on Representation Learning for NLP, 1-7, 2016 [[preprint](https://link.zhihu.com/?target=http%3A//iphome.hhi.de/samek/pdf/ArrACL16.pdf), [bibtex](https://link.zhihu.com/?target=http%3A//heatmapping.org/bibtex/arras-acl16.txt)]
- F Horn, L Arras, G Montavon, KR Müller, W Samek. [Exploring text datasets by visualizing relevant words](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1707.05261)
    arXiv:1707.05261, 2017

## 图像及脸部识别的应用：

- S Lapuschkin, A Binder, G Montavon, KR Müller, W Samek. [Analyzing Classifiers: Fisher Vectors and Deep Neural Networks](https://link.zhihu.com/?target=http%3A//dx.doi.org/10.1109/CVPR.2016.318)
    Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2912-2920, 2016 [[preprint](https://link.zhihu.com/?target=http%3A//iphome.hhi.de/samek/pdf/LapCVPR16.pdf), [bibtex](https://link.zhihu.com/?target=http%3A//heatmapping.org/bibtex/lapuschkin-cvpr16.txt)]
- S Lapuschkin, A Binder, KR Müller, W Samek. [Understanding and Comparing Deep Neural Networks for Age and Gender Classification](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1708.07689)
    IEEE International Conference on Computer Vision Workshops (ICCVW), 1629-1638, 2017 [[preprint](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1708.07689), [bibtex](https://link.zhihu.com/?target=http%3A//heatmapping.org/bibtex/lapuschkin-iccv17.txt)]
- C Seibold, W Samek, A Hilsmann, P Eisert. [Accurate and Robust Neural Networks for Security Related Applications Exampled by Face Morphing Attacks](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1806.04265)
    arXiv:1806.04265, 2018
- S Bach, A Binder, KR Müller, W Samek. [Controlling Explanatory Heatmap Resolution and Semantics via Decomposition Depth](https://link.zhihu.com/?target=http%3A//dx.doi.org/10.1109/ICIP.2016.7532763)
    Proceedings of the IEEE International Conference on Image Processing (ICIP), 2271-2275, 2016 [[preprint](https://link.zhihu.com/?target=http%3A//iphome.hhi.de/samek/pdf/BacICIP16.pdf), [bibtex](https://link.zhihu.com/?target=http%3A//heatmapping.org/bibtex/bach-icip16.txt)]
- A Binder, S Bach, G Montavon, KR Müller, W Samek. [Layer-wise Relevance Propagation for Deep Neural Network Architectures](https://link.zhihu.com/?target=http%3A//dx.doi.org/10.1007/978-981-10-0557-2_87)
    Proceedings of the 7th International Conference on Information Science and Applications (ICISA), 6679:913-922, Springer Singapore, 2016 [[preprint](https://link.zhihu.com/?target=http%3A//iphome.hhi.de/samek/pdf/BinICISA16.pdf), [bibtex](https://link.zhihu.com/?target=http%3A//heatmapping.org/bibtex/binder-icisa16.txt)]
- F Arbabzadah, G Montavon, KR Müller, W Samek. [Identifying Individual Facial Expressions by Deconstructing a Neural Network](https://link.zhihu.com/?target=http%3A//dx.doi.org/10.1007/978-3-319-45886-1_28)
    Pattern Recognition - 38th German Conference, GCPR 2016, Lecture Notes in Computer Science, 9796:344-354, 2016 [[preprint](https://link.zhihu.com/?target=http%3A//arxiv.org/pdf/1606.07285), [bibtex](https://link.zhihu.com/?target=http%3A//heatmapping.org/bibtex/arbabzadah-gcpr16.txt)]

## 视频的应用：

- C Anders, G Montavon, W Samek, KR Müller. [Understanding Patch-Based Learning by Explaining Predictions](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1806.06926)
    arXiv:1806.06926, 2018
- V Srinivasan, S Lapuschkin, C Hellge, KR Müller, W Samek. [Interpretable human action recognition in compressed domain](https://link.zhihu.com/?target=http%3A//dx.doi.org/10.1109/ICASSP.2017.7952445)
    Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 1692-1696, 2017 [[preprint](https://link.zhihu.com/?target=http%3A//iphome.hhi.de/samek/pdf/SriICASSP17.pdf), [bibtex](https://link.zhihu.com/?target=http%3A//heatmapping.org/bibtex/srinivasan-icassp17.txt)]

## 语音的应用：

- S Becker, M Ackermann, S Lapuschkin, KR Müller, W Samek. [Interpreting and Explaining Deep Neural Networks for Classification of Audio Signals](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1807.03418)
    arXiv:1807.03418, 2018

## 短Paper

- W Samek, G Montavon, A Binder, S Lapuschkin, and KR Müller. [Interpreting the Predictions of Complex ML Models by Layer-wise Relevance Propagation](https://link.zhihu.com/?target=http%3A//arxiv.org/abs/1611.08191)
    NIPS Workshop on Interpretable ML for Complex Systems, 1-5, 2016 [[preprint](https://link.zhihu.com/?target=http%3A//arxiv.org/abs/1611.08191), [bibtex](https://link.zhihu.com/?target=http%3A//heatmapping.org/bibtex/samek-nips16.txt)]
- G Montavon, S Bach, A Binder, W Samek, KR Müller. [Deep Taylor Decomposition of Neural Networks](https://link.zhihu.com/?target=http%3A//icmlviz.github.io/assets/papers/13.pdf)
    ICML Workshop on Visualization for Deep Learning, 1-3, 2016 [[preprint](https://link.zhihu.com/?target=http%3A//iphome.hhi.de/samek/pdf/MonICML16.pdf), [bibtex](https://link.zhihu.com/?target=http%3A//heatmapping.org/bibtex/montavon-icml16.txt)]
- A Binder, W Samek, G Montavon, S Bach, KR Müller. [Analyzing and Validating Neural Networks Predictions](https://link.zhihu.com/?target=http%3A//icmlviz.github.io/assets/papers/18.pdf)
    ICML Workshop on Visualization for Deep Learning, 1-4, 2016 [[preprint](https://link.zhihu.com/?target=http%3A//iphome.hhi.de/samek/pdf/BinICML16.pdf), [bibtex](https://link.zhihu.com/?target=http%3A//heatmapping.org/bibtex/binder-icml16.txt)]

## 最后：BVLC Model Zoo Contributions

- Pascal VOC 2012 Multilabel Model (see [paper](https://link.zhihu.com/?target=http%3A//heatmapping.org/%23cvpr16)): [[caffemodel](https://link.zhihu.com/?target=http%3A//heatmapping.org/files/bvlc_model_zoo/pascal_voc_2012_multilabel/pascalvoc2012_train_simple2_iter_30000.caffemodel)] [[prototxt](https://link.zhihu.com/?target=http%3A//heatmapping.org/files/bvlc_model_zoo/pascal_voc_2012_multilabel/deploy_x30.prototxt)]
- Age and Gender Classification Models (see [paper](https://link.zhihu.com/?target=http%3A//heatmapping.org/%23iccv17)): [[data and models](https://link.zhihu.com/?target=https%3A//github.com/sebastian-lapuschkin/understanding-age-gender-deep-learning-models)]

## **相关资料：**

[1]Samek, Wojciech, Thomas Wiegand, and Klaus-Robert Müller. "Explainable artificial intelligence: Understanding, visualizing and interpreting deep learning models."*arXiv preprint arXiv:1708.08296*(2017).