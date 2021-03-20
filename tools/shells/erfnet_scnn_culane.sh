# Trained weights: erfnet_scnn_culane_20210206.pt
# Training
python main_landec.py --epochs=12 --lr=0.2 --batch-size=20 --dataset=culane --method=scnn --backbone=erfnet --mixed-precision --exp-name=erfnet_scnn_culane
# Predicting lane points for testing
python main_landec.py --state=2 --batch-size=80 --continue-from=erfnet_scnn_culane.pt --dataset=culane --method=scnn --backbone=erfnet --mixed-precision
# Testing with official scripts
./autotest_culane.sh erfnet_scnn_culane test
