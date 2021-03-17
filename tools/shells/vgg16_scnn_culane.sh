# Trained weights: vgg16_scnn_culane_20210309.pt
# Training
python main_landec_as_seg.py --epochs=12 --lr=0.3 --batch-size=20 --dataset=culane --method=scnn --backbone=vgg16 --mixed-precision --exp-name=vgg16_scnn_culane
# Predicting lane points for testing
python main_landec_as_seg.py --state=2 --batch-size=80 --continue-from=vgg16_scnn_culane.pt --dataset=culane --method=scnn --backbone=vgg16 --mixed-precision
# Testing with official scripts
./autotest_culane.sh vgg16_scnn_culane test
