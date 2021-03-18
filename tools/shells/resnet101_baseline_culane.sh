# Trained weights: resnet101_baseline_culane_20210312.pt
# Training, scale lr linearly on 11G GPU (square root scaling does not converge on this dataset)
python main_landec_as_seg.py --epochs=12 --lr=0.08 --batch-size=8 --dataset=culane --method=baseline --backbone=resnet101 --workers=4 --warmup-steps=500 --mixed-precision --exp-name=resnet101_baseline_culane
# Predicting lane points for testing
python main_landec_as_seg.py --state=2 --batch-size=32 --continue-from=resnet101_baseline_culane.pt --dataset=culane --method=baseline --backbone=resnet101 --workers=4 --mixed-precision
# Testing with official scripts
./autotest_culane.sh resnet101_baseline_culane test
