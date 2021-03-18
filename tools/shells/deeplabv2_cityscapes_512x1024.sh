# Trained weights: deeplabv2_cityscapes_512x1024_20201219.pt
python main_semseg.py --epochs=60 --lr=0.002 --batch-size=4 --dataset=city --model=deeplabv2-big --mixed-precision --exp-name=deeplabv2_cityscapes_512x1024
