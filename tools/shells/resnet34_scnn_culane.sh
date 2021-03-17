# Trained weights: resnet34_scnn_culane_20210220.pt
# Training
python main_landec_as_seg.py --epochs=12 --lr=0.2 --batch-size=20 --dataset=culane --method=scnn --backbone=resnet34 --mixed-precision --exp-name=resnet34_scnn_culane
# Predicting lane points for testing
python main_landec_as_seg.py --state=2 --batch-size=80 --continue-from=resnet34_scnn_culane.pt --dataset=culane --method=scnn --backbone=resnet34 --mixed-precision
# Testing with official scripts
./autotest_culane.sh resnet34_scnn_culane test
