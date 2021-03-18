# Trained weights: resnet18_baseline_tusimple_20210215.pt
# Training
python main_landec_as_seg.py --epochs=50 --lr=0.2 --batch-size=20 --dataset=tusimple --method=baseline --backbone=resnet18 --mixed-precision --exp-name=resnet18_baseline_tusimple
# Predicting lane points for testing
python main_landec_as_seg.py --state=2 --batch-size=80 --continue-from=resnet18_baseline_tusimple.pt --dataset=tusimple --method=baseline --backbone=resnet18 --mixed-precision
# Testing with official scripts
./autotest_tusimple.sh resnet18_baseline_tusimple test
