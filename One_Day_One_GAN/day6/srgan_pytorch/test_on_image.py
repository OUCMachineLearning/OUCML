from models import GeneratorRRDB
from datasets import denormalize, mean, std
import torch
from torch.autograd import Variable
import argparse
import os
from torchvision.utils import save_image
from PIL import Image
from models import *
import torch
from torch.autograd import Variable
import argparse
import os
from torchvision import transforms
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from torchvision.utils import save_image
from PIL import Image
import glob
parser = argparse.ArgumentParser()

parser.add_argument("--image_path", type=str,default='xuan_images_test_NW_gen/', help="Path to image")
parser.add_argument("--save_path", type=str,default='show_result/GAN_DN', help="test to save")

# parser.add_argument("--image_path", type=str,default='../result/NW_v2UNet512no/', help="Path to image")
# parser.add_argument("--save_path", type=str,default='../result/ESR', help="test to save")
parser.add_argument('-c',"--checkpoint_model", type=str, default='saved_models/generator_79.pth', help="Path to checkpoint model")
parser.add_argument("--channels", type=int, default=3, help="Number of image channels")
parser.add_argument("--residual_blocks", type=int, default=9, help="Number of residual blocks in G")
parser.add_argument("--netG", type=str, default='RRDB', help="The network structure of the generator")
parser.add_argument("--concat", type=str, default='no', help="Compare inputs and outputs")
parser.add_argument("--img_weight", type=int, default=256, help="img_weight")

opt = parser.parse_args()
print(opt)
img_weight=opt.img_weight
img_hight=opt.img_weight
save_path=opt.save_path+"_"+opt.netG+"_"+str(opt.img_weight*4)+"_"+opt.concat
# os.makedirs(save_path, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define model and load model checkpoint
if opt.netG=='RRDB':
    generator = GeneratorRRDB (opt.channels,filters=64,num_res_blocks=opt.residual_blocks,num_upsample=2).to (device)
if opt.netG=='UNet' or opt.netG=='NW':
    generator = UNet().to (device)
# generator = GeneratorRRDB(opt.channels, filters=64, num_res_blocks=opt.residual_blocks).to(device)
generator.load_state_dict(torch.load(opt.checkpoint_model))

generator.eval()

transform = transforms.Compose([
    transforms.Resize ((img_weight,img_hight),Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)]
)

# Prepare input
for image_path in glob.glob(opt.image_path+'*.*'):
    print(image_path)
    image_tensor = Variable(transform(Image.open(image_path))).to(device).unsqueeze(0)
    print(image_tensor.shape)

    # Upsample image
    with torch.no_grad():
        sr_image = generator(image_tensor)

    # Save image
    fn = image_path.split("/")[-2]+"/"+image_path.split("/")[-1]

    if opt.concat=="yes":
        if opt.netG=="RRDB":
            # sr_image =sr_image
            image_tensor=nn.functional.interpolate (image_tensor,scale_factor=2)

        print(image_tensor.shape,sr_image.shape,opt.checkpoint_model[-6:-4])
        print(torch.cat ((image_tensor,sr_image),3).shape)

        img_grid = denormalize(torch.cat ((image_tensor,sr_image),3))
        os.makedirs (f"{save_path}_concat",exist_ok=True)
        save_image(img_grid, f"{save_path}_concat/{fn[:-4]}.tif",normalize=False)

    else:
        print(image_tensor.shape,sr_image.shape,opt.checkpoint_model[-6:-4])
        if opt.netG=="RRDB":
            # sr_image =sr_image
            image_tensor=nn.functional.interpolate (image_tensor,scale_factor=4)
        img_grid = denormalize(sr_image)
        img_grid_o = denormalize(image_tensor)
        os.makedirs (f"{save_path}_gen/"+image_path.split("/")[-2],exist_ok=True)
        os.makedirs (f"{save_path}_interpolation/"+ image_path.split("/")[-2],exist_ok=True)

        save_image(img_grid, f"{save_path}_gen/{fn[:-4]}.tif",normalize=False)
        save_image(img_grid_o, f"{save_path}_interpolation/{fn[:-4]}.tif",normalize=False)
print(save_path)