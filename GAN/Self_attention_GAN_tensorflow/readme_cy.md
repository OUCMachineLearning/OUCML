## celebA
首先,rm dataset/celebA
然后python download.py celebA ......你会发现下载不了哈哈哈哈哈哈哈哈哈
然后,我发下可以直接用kaggle下载
先注册一个kaggle账号,然后install kaggel ,再按github上的教程编辑kaggle api的json
>>kaggle datasets download -d jessicali9530/celeba-dataset 

把/img_align_celeba文件夹解压放到datasets下面

```
hx@hx-b412:~$ python main.py --phase train --dataset img_align_celeba  --gan_type wgan-gp --img 64

```

## cifar10
直接就能跑
python main.py --phase train --dataset cifar10 --gan_type wgan-gp --img 32