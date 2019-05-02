from models import *
from datasets  import  *
import torch
from torch.autograd import Variable
import argparse
import os
from torchvision.utils import save_image
from PIL import Image
import glob
parser = argparse.ArgumentParser()
parser.add_argument("--image_path", type=str, required=True, help="Path to image")
parser.add_argument("--checkpoint_model", type=str, required=True, help="Path to checkpoint model")
parser.add_argument("--netG", type=str, default='W2M', help="The network structure of the generator")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")

opt = parser.parse_args()
print(opt)

os.makedirs("images/outputs", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_shape = (opt.channels, opt.img_height, opt.img_width)


generator = GeneratorResNet(input_shape, opt.n_residual_blocks).to (device)


generator.load_state_dict(torch.load(opt.checkpoint_model))

generator.eval()

transform = transforms.Compose([
    transforms.Resize((512,512), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

# Prepare input
for image_path in glob.glob(opt.image_path+'*.*'):
    print(image_path)
    image_tensor = Variable(transform(Image.open(image_path))).to(device).unsqueeze(0)
    print(image_tensor.shape)

    # Upsample image
    with torch.no_grad():
        sr_image = generator(image_tensor).cpu()

    # Save image
    fn = image_path.split("/")[-1]
    img_grid = torch.cat((image_tensor.cpu(),sr_image), 3)
    save_image(img_grid, f"images/outputs/sr-{opt.checkpoint_model[-5]}-{fn[:-4]}.tif",nrow=1,normalize=True)
