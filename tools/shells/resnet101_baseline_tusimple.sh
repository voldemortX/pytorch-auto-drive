# Trained weights: resnet101_baseline_tusimple_20210217.pt
# Training, scale lr by square root on 11G GPU
python main_landec.py --epochs=50 --lr=0.13 --batch-size=8 --dataset=tusimple --method=baseline --backbone=resnet101 --workers=4 --warmup-steps=500 --mixed-precision --exp-name=resnet101_baseline_tusimple
# Predicting lane points for testing
python main_landec.py --state=2 --batch-size=32 --continue-from=resnet101_baseline_tusimple.pt --dataset=tusimple --method=baseline --backbone=resnet101 --workers=4 --mixed-precision
# Testing with official scripts
./autotest_tusimple.sh resnet101_baseline_tusimple test
