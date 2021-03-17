# Trained weights: vgg16_baseline_tusimple_20210223.pt
# Training
python main_landec_as_seg.py --epochs=50 --lr=0.25 --batch-size=20 --dataset=tusimple --method=baseline --backbone=vgg16 --mixed-precision --exp-name=vgg16_baseline_tusimple
# Predicting lane points for testing
python main_landec_as_seg.py --state=2 --batch-size=80 --continue-from=vgg16_baseline_tusimple.pt --dataset=tusimple --method=baseline --backbone=vgg16 --mixed-precision
# Testing with official scripts
./autotest_tusimple.sh vgg16_baseline_tusimple test
