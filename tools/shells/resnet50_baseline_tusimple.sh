# Trained weights: resnet50_baseline_tusimple_20210217.pt
# Training, scale lr by square root on 11G GPU
python main_landec_as_seg.py --epochs=50 --lr=0.13 --batch-size=8 --dataset=tusimple --method=baseline --backbone=resnet50 --workers=4 --warmup-steps=500 --mixed-precision --exp-name=resnet50_baseline_tusimple
# Predicting lane points for testing
python main_landec_as_seg.py --state=2 --batch-size=32 --continue-from=resnet50_baseline_tusimple.pt --dataset=tusimple --method=baseline --backbone=resnet50 --workers=4 --mixed-precision
# Testing with official scripts
./autotest_tusimple.sh resnet50_baseline_tusimple test
