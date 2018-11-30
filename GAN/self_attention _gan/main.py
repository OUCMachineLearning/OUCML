
from parameter import *
from trainer import Trainer
# from tester import Tester
from data_loader import Data_Loader
from torch.backends import cudnn
from utils import make_folder
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def main(config):
    # For fast training
    cudnn.benchmark = True


    # Data loader
    data_loader = Data_Loader(config.train, config.dataset, config.image_path, config.imsize,
                             config.batch_size, shuf=config.train)

    # Create directories if not exist
    make_folder(config.model_save_path, config.version)
    make_folder(config.sample_path, config.version)
    make_folder(config.log_path, config.version)
    make_folder(config.attn_path, config.version)


    if config.train:
        if config.model=='sagan':
            trainer = Trainer(data_loader.loader(), config)
        elif config.model == 'qgan':
            trainer = qgan_trainer(data_loader.loader(), config)
        trainer.train()
    else:
        tester = Tester(data_loader.loader(), config)
        tester.test()

if __name__ == '__main__':
    config = get_parameters()
    print(config)
    main(config)