# Cutout in Shake-Shake Regularization Networks

In order to add cutout to Xavier Gastaldi's shake-shake regularization code we simply add a cutout function to transforms.lua (lines 16 to 29) and then append the cutout function to the CIFAR-10 and CIFAR-100 pre-processing pipelines (lines 49 and 60 in cifar10.lua and cifar100.lua respectively). 

## Usage  
1. Follow Usage instruction 1 from https://github.com/xgastaldi/shake-shake to install fb.resnet.torch and related libraries.
2. Once installed, navigate to your local fb.resnet.torch/datasets folder.
3. Copy the files from this folder (shake-shake) and paste them into the datasets folder. This should overwrite cifar10.lua, cifar100.lua, and transforms.lua.
4. Continue following remaining instructions from https://github.com/xgastaldi/shake-shake. CIFAR-10 should now train using cutout with a length of 16 and CIFAR-100 will train using cutout with a length of 8.
