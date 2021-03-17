# Trained weights: deeplabv2_cityscapes_256x512_fp32_20201227.pt
python main_semseg.py --epochs=60 --lr=0.004 --batch-size=8 --dataset=city --model=deeplabv2 --exp-name=deeplabv2_cityscapes_256x512_fp32
