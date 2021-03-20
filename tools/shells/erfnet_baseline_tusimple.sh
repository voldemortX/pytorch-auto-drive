# Trained weights: erfnet_baseline_tusimple_20210201.pt
# Training
python main_landec.py --epochs=50 --lr=0.2 --batch-size=20 --dataset=tusimple --method=baseline --backbone=erfnet --mixed-precision --exp-name=erfnet_baseline_tusimple
# Predicting lane points for testing
python main_landec.py --state=2 --batch-size=80 --continue-from=erfnet_baseline_tusimple.pt --dataset=tusimple --method=baseline --backbone=erfnet --mixed-precision
# Testing with official scripts
./autotest_tusimple.sh erfnet_baseline_tusimple test
