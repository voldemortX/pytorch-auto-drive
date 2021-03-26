# Trained weights: deeplabv3_cityscapes_512x1024_20210322.pt
python main_semseg.py --epochs=60 --lr=0.002 --batch-size=4 --dataset=city --model=deeplabv3-big --mixed-precision --exp-name=deeplabv3_cityscapes_512x1024
