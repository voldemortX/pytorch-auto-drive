# Trained weights: deeplabv3_city_256x512_20201226.pt
python main_semseg.py --epochs=60 --lr=0.004 --batch-size=8 --dataset=city --model=deeplabv3 --mixed-precision --exp-name=deeplabv3_city_256x512
