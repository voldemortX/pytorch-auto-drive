# Trained weights: deeplabv2_cityscapes_256x512_20201225.pt
python main_semseg.py --epochs=60 --lr=0.004 --batch-size=8 --dataset=city --model=deeplabv2 --mixed-precision --exp-name=deeplabv2_cityscapes_256x512
