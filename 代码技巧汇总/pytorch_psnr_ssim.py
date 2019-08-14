#PSNR
per_image_mse_loss = F.mse_loss (gen_hr,imgs_hr, reduction='none')
per_image_psnr = 10 * torch.log10 (10 / per_image_mse_loss)
tensor_average_psnr = torch.mean (per_image_psnr).item ()

#SSIM
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
# X: (N,3,H,W) a batch of RGB images with values ranging from 0 to 255.
# Y: (N,3,H,W)  
ssim_val = ssim( X, Y, data_range=255, size_average=False) # return (N,)
ms_ssim_val = ms_ssim( X, Y, data_range=255, size_average=False ) #(N,)

# or set 'size_average=True' to get a scalar value as loss.
ssim_loss = ssim( X, Y, data_range=255, size_average=True) # return a scalar value
ms_ssim_loss = ms_ssim( X, Y, data_range=255, size_average=True )

# or reuse windows with SSIM & MS_SSIM. 
ssim_module = SSIM(win_size=11, win_sigma=1.5, data_range=255, size_average=True, channel=3)
ms_ssim_module = MS_SSIM(win_size=11, win_sigma=1.5, data_range=255, size_average=True, channel=3)

ssim_loss = ssim_module(X, Y)
ms_ssim_loss = ms_ssim_module(X, Y)
