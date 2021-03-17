# Trained weights: deeplabv2_synthia_512x1024_20201225.pt
python main_semseg.py --epochs=20 --lr=0.002 --batch-size=4 --dataset=synthia --model=deeplabv2 --mixed-precision --exp-name=deeplabv2_synthia_512x1024
