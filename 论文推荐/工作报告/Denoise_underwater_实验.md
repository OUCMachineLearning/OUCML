# Denoise_underwater 实验

任务:单纯的水下去噪

---

我们的问题:

UNET 的改进太少

|             | PSNR | SSIM |   备注   |  SN  |  DA  |  TODO  |
| :---------: | :--: | :--: | :------: | :--: | :--: | :--: |
|    DNGAN    |      |      | 去海洋雪 |  no  |  no  |    |
|    DNGAN    |      |      | 去海洋雪 | yes  |  no  |    |
|    DNGAN    |      |      | 去海洋雪 |  no  | yes  |   |
|    DNGAN    |      |      | 去海洋雪 | yes  | yes  |   |
| noise2noise | 28.219122 | 0.947879 | 对比实验 |  no  |  no  |  yes  |
|   ID-CGAN   | 18.677912 | 0.782176 |  对比实验  |  no  |  no | yes |
|             |      |      |          |      |      |      |
|             |      |      |          |      |      |      |
|             |      |      |          |      |      |      |

需要补充的实验

- SN 的作用
- DA 的作用
- 训练稳定性曲线变化
- 
- 用表格进一步描述网络的具体结构
- Noise2noise 对比实验—yes
- ID-CGAN对比实验—yes

