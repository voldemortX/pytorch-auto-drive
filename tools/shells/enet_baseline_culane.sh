# Trained weights: enet_baseline_culane_20210312.pt
# Training
python main_landec.py --epochs=12 --lr=0.5 --batch-size=20 --dataset=culane --method=baseline --backbone=enet --mixed-precision --exp-name=enet_baseline_culane
# Predicting lane points for testing
python main_landec.py --state=2 --batch-size=80 --continue-from=enet_baseline_culane.pt --dataset=culane --method=baseline --backbone=enet --mixed-precision
# Testing with official scripts
./autotest_culane.sh enet_baseline_culane test
