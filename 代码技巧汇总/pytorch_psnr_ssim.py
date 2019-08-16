#PSNR
per_image_mse_loss = F.mse_loss (gen_hr,imgs_hr, reduction='none')
per_image_psnr = 10 * torch.log10 (10 / per_image_mse_loss)
tensor_average_psnr = torch.mean (per_image_psnr).item ()

#SSIM
import pytorch_ssim
import torch
from torch.autograd import Variable

img1 = Variable(torch.rand(1, 1, 256, 256))
img2 = Variable(torch.rand(1, 1, 256, 256))

if torch.cuda.is_available():
	img1 = img1.cuda()
	img2 = img2.cuda()

print(pytorch_ssim.ssim(img1, img2))

ssim_loss = pytorch_ssim.SSIM(window_size = 11)

print(ssim_loss(img1, img2))

#MSSSIM
import pytorch_ssim
import torch
from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
m = pytorch_msssim.MSSSIM()

img1 = torch.rand(1, 1, 256, 256)
img2 = torch.rand(1, 1, 256, 256)

print(pytorch_msssim.msssim(img1, img2))
print(m(img1, img2)))
