# My segmentation codebase(beta version, untested)
Segmentation models(Start with DeeplabV3) based on PyTorch 1.3 with tensorboard
Including modulated(borrowed) mIOU&pixel acc calculation, "poly" learning rate schedule, basic input transformations and visulizations

Currently supported datasets: 
PASCAL VOC 2012(Deeplab 10582 trainaug version)

Currently supported models:
DeeplabV3(ImageNet pretrained ResNet 101) from torchvision
