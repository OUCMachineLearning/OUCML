# Denoise_underwater 实验

任务:单纯的水下去噪

---

我们的问题:

UNET 的改进太少

|             | PSNR | SSIM |   备注   |  SN  |  DA  |  TODO  | URL |
| :---------: | :--: | :--: | :--: | :--: | :--: | :------: | :------: |
|    DNGAN    |      |      | 去海洋雪 |  no  |  no  |  doing  | sftp://222.195.151.21:3900//home/ouc/cy/work_jiang/DNGAN/DNGAN |
|    DNGAN    | ~~30.167056~~ | ~~0.925217~~ | 去海洋雪 | yes  |  no  |  doing  | sftp://222.195.151.21:3900//home/ouc/cy/work_jiang/DNGAN/DNGAN+sn |
|    DNGAN    |      |      | 去海洋雪 |  no  | yes  | doing | sftp://222.195.151.21:3900//home/ouc/cy/work_jiang/DNGAN/DNGAN+da |
|    DNGAN    | ~~31.263745~~ | ~~0.940520~~ | 去海洋雪 | yes  | yes  | doing | sftp://222.195.151.21:3900//home/ouc/cy/work_jiang/DNGAN/DNGAN+sn+da |
| noise2noise | 28.219122 | 0.947879 | 对比实验 |  no  |  no  |  yes  | sftp://222.195.151.21:3902//home/hx/cy/work_jiang/noise2noise-pytorch |
|   ID-CGAN   | 18.677912 | 0.782176 |  对比实验  |  no  |  no | yes | sftp://222.195.151.21:3902//home/hx/cy/work_jiang/Single-Image-De-Raining-Keras |
|             |      |      |          |      |      |      |      |
|             |      |      |          |      |      |      |      |
|             |      |      |          |      |      |      |      |

需要补充的实验

- SN 的作用
- DA 的作用
- 训练稳定性曲线变化
- 
- 用表格进一步描述网络的具体结构
- Noise2noise 对比实验—yes
- ID-CGAN对比实验—yes

