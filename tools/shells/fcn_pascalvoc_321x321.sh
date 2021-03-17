# Trained weights: fcn_pascalvoc_321x321_20201111.pt
python main_semseg.py --epochs=30 --lr=0.002 --batch-size=8 --dataset=voc --model=fcn --workers=4 --mixed-precision --exp-name=fcn_pascalvoc_321x321
