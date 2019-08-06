![image-20190522135654280](http://ww3.sinaimg.cn/large/006tNc79ly1g3a1ynuxyzj30zv0u0h1s.jpg)

    import torch
    import torch.nn.functional as F
    from math import exp
    import numpy as np
     
     
    # 计算一维的高斯分布向量
    def gaussian(window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()
     
     
    # 创建高斯核，通过两个一维高斯分布向量进行矩阵乘法得到
    # 可以设定channel参数拓展为3通道
    def create_window(window_size, channel=1):
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
     
     
    # 计算SSIM
    # 直接使用SSIM的公式，但是在计算均值时，不是直接求像素平均值，而是采用归一化的高斯核卷积来代替。
    # 在计算方差和协方差时用到了公式Var(X)=E[X^2]-E[X]^2, cov(X,Y)=E[XY]-E[X]E[Y].
    # 正如前面提到的，上面求期望的操作采用高斯核卷积代替。
    def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
        # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
        if val_range is None:
            if torch.max(img1) > 128:
                max_val = 255
            else:
                max_val = 1
     
            if torch.min(img1) < -0.5:
                min_val = -1
            else:
                min_val = 0
            L = max_val - min_val
        else:
            L = val_range
     
        padd = 0
        (_, channel, height, width) = img1.size()
        if window is None:
            real_size = min(window_size, height, width)
            window = create_window(real_size, channel=channel).to(img1.device)
     
        mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
        mu2 = F.conv2d(img2, window, padding=padd, groups=channel)
     
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
     
        sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2
     
        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2
     
        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = torch.mean(v1 / v2)  # contrast sensitivity
     
        ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
     
        if size_average:
            ret = ssim_map.mean()
        else:
            ret = ssim_map.mean(1).mean(1).mean(1)
     
        if full:
            return ret, cs
        return ret
     
     
     
    # Classes to re-use window
    class SSIM(torch.nn.Module):
        def __init__(self, window_size=11, size_average=True, val_range=None):
            super(SSIM, self).__init__()
            self.window_size = window_size
            self.size_average = size_average
            self.val_range = val_range
     
            # Assume 1 channel for SSIM
            self.channel = 1
            self.window = create_window(window_size)
     
        def forward(self, img1, img2):
            (_, channel, _, _) = img1.size()
     
            if channel == self.channel and self.window.dtype == img1.dtype:
                window = self.window
            else:
                window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
                self.window = window
                self.channel = channel
     
            return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)
我写好的文件夹下名字对其 PSNR/SSIM

```python
'''
calculate the PSNR and SSIM.
same as MATLAB's results
'''
import os
import math
import numpy as np
import cv2
import glob
import sys

def main():
    # Configurations

    # GT - Ground-truth;
    # Gen: Generated / Restored / Recovered images
    
#    folder_GT = 'DMDM40256no_gen'
#    folder_GT = 'testDNDN256no_gen'
    folder_GT = 'testDWNW_199_256no_gen'
    
    folder_Gen = 'testB'

    crop_border = 4
    suffix = '/'  # suffix for Gen images
    test_Y = False  # True: test Y channel only; False: test RGB channels

    PSNR_all = []
    SSIM_all = []
    img_list = sorted(glob.glob(folder_GT + '/*'))

    if test_Y:
        print('Testing Y channel.')
    else:
        print('Testing RGB channels.')
    print(len(img_list))
    
#    fo = open("DN-gt.txt", "w")
    fo = open("DW-gt.txt", "w")
#    fo = open("DM-gt.txt", "w")

    for i, img_path in enumerate(img_list):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
#        print(base_name,img_path,folder_Gen,suffix)
        im_GT = cv2.imread(img_path) 
#        print(type(im_GT))
        im_GT = im_GT / 255
        im_Gen = cv2.imread(os.path.join(folder_Gen, base_name  + '.jpg')) 
#        print(type(im_Gen),os.path.join(folder_Gen, base_name  + '.jpg'))
        im_Gen = im_Gen / 255

        if test_Y and im_GT.shape[2] == 3:  # evaluate on Y channel in YCbCr color space
            im_GT_in = bgr2ycbcr(im_GT)
            im_Gen_in = bgr2ycbcr(im_Gen)
        else:
            im_GT_in = im_GT
            im_Gen_in = im_Gen

        # crop borders
        if im_GT_in.ndim == 3:
            cropped_GT = im_GT_in[crop_border:-crop_border, crop_border:-crop_border, :]
            cropped_Gen = im_Gen_in[crop_border:-crop_border, crop_border:-crop_border, :]
        elif im_GT_in.ndim == 2:
            cropped_GT = im_GT_in[crop_border:-crop_border, crop_border:-crop_border]
            cropped_Gen = im_Gen_in[crop_border:-crop_border, crop_border:-crop_border]
        else:
            raise ValueError('Wrong image dimension: {}. Should be 2 or 3.'.format(im_GT_in.ndim))

        # calculate PSNR and SSIM
        PSNR = calculate_psnr(cropped_GT * 255, cropped_Gen * 255)

        SSIM = calculate_ssim(cropped_GT * 255, cropped_Gen * 255)
#        print('{:3d} - {:25}. \tPSNR: {:.6f} dB, \tSSIM: {:.6f}'.format(i + 1, base_name, PSNR, SSIM))
#        print ("文件名: ", fo.name)
        fo.write('{:3d} - {:25}. \tPSNR: {:.6f} dB, \tSSIM: {:.6f}\n'.format(
                i + 1, base_name, PSNR, SSIM))

#        print(i,"/",len(img_list),i/len(img_list),"%")
        print(i)
        PSNR_all.append(PSNR)
        SSIM_all.append(SSIM)
    
#    print('Average: PSNR: {:.6f} dB, SSIM: {:.6f}'.format(sum(PSNR_all) / len(PSNR_all),sum(SSIM_all) / len(SSIM_all)))
    
    fo.write('Average: PSNR: {:.6f} dB, SSIM: {:.6f}\n'.format(
        sum(PSNR_all) / len(PSNR_all),
        sum(SSIM_all) / len(SSIM_all)))
    # 关闭打开的文件
    fo.close()


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def bgr2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


if __name__ == '__main__':
    main()
ß
```

