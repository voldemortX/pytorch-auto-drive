# ERFNet

> [ERFNet: Efficient Residual Factorized ConvNet for Real-Time Semantic Segmentation](https://ieeexplore.ieee.org/abstract/document/8063438/) ITS 2017

## Method Overview

ERFNet proposes a simple encoder-decoder architecture for real-time semantic segmentation. The core design lies in a Non-bottleneck-1D design, which starts with the Non-bottleneck (basic) block for ResNet-34/18, then replace each 3x3 conv by one 1x3 and one 3x1 convs. Although not much celebrated in the semantic segmentation task, it is widely recognized as a strong backbone for lane detection, my bet is the 1D convolution's alignment with thin line objects. However, there is a somewhat ironic problem with ERFNet: intuitively, 1x3 + 3x1 design seems only 6/9 the compute of a 3x3 conv, but with [winograd](https://arxiv.org/pdf/1509.09308.pdf), a 3x3 conv's compute can be reduced to 4/9. So the benefit of this design is only a good-looking FLOPs count *(do correct me if I'm wrong)*. By going deeper, ERFNet shows a higher latency disproportional to its FLOPs count. Despite that, the simple encoder-decoder segmentation network provides good performance. And who knows, maybe even 1x3 convolution is getting optimized somewhere.

<div align=center>
<img src="https://user-images.githubusercontent.com/32259501/158295814-90e84509-f088-41cc-a805-cb2a976665fd.png"/>
</div>

## Results

*Training time estimated with single 2080 Ti.*

*ImageNet pre-training, 3-times average/best.*

### Cityscapes (val)

| backbone | resolution | training time | precision | mIoU (avg) | mIoU | |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ResNet-101 | 512 x 1024 | 5h | mix | 71.99 | 72.47 | [model](https://drive.google.com/file/d/1uzBSboKD-Xt0K6VHd2aF561Cy13q9xRe/view?usp=sharing) \| [shell](../tools/shells/erfnet_cityscapes_512x1024.sh) |

## Profiling

*FPS is best trial-avg among 3 trials on a 2080 Ti.*

| backbone | resolution | FPS | FLOPS(G) | Params(M) |
| :---: | :---: | :---: | :---: | :---: |
| ResNet-101 | 256 x 512 | 91.20 | 15.03 | 2.07 |
| ResNet-101 | 512 x 1024 | 85.51 | 60.11 | 2.07 |
| ResNet-101 | 1024 x 2048 | 21.53 | 240.44 | 2.07 |

## Citation

```
@article{romera2017erfnet,
  title={Erfnet: Efficient residual factorized convnet for real-time semantic segmentation},
  author={Romera, Eduardo and Alvarez, Jos{\'e} M and Bergasa, Luis M and Arroyo, Roberto},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  volume={19},
  number={1},
  pages={263--272},
  year={2017}
}
```
