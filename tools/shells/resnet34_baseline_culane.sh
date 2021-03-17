# Trained weights: resnet34_baseline_culane_20210219.pt
# Training
python main_landec_as_seg.py --epochs=12 --lr=0.2 --batch-size=20 --dataset=culane --method=baseline --backbone=resnet34 --mixed-precision --exp-name=resnet34_baseline_culane
# Predicting lane points for testing
python main_landec_as_seg.py --state=2 --batch-size=80 --continue-from=resnet34_baseline_culane.pt --dataset=culane --method=baseline --backbone=resnet34 --mixed-precision
# Testing with official scripts
./autotest_culane.sh resnet34_baseline_culane test
