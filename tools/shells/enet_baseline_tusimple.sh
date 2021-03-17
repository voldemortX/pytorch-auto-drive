# Trained weights: enet_baseline_tusimple_20210312.pt
# Training
python main_landec_as_seg.py --epochs=50 --lr=0.4 --batch-size=20 --dataset=tusimple --method=baseline --backbone=enet --mixed-precision --exp-name=enet_baseline_tusimple
# Predicting lane points for testing
python main_landec_as_seg.py --state=2 --batch-size=80 --continue-from=enet_baseline_tusimple.pt --dataset=tusimple --method=baseline --backbone=enet --mixed-precision
# Testing with official scripts
./autotest_tusimple.sh enet_baseline_tusimple test
