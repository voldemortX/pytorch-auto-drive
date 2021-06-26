# Trained weights: erfnet_scnn_llamas_20210625.pt
# Training
python main_landec.py --epochs=10 --lr=0.5 --batch-size=20 --dataset=llamas --method=scnn --backbone=erfnet --mixed-precision --exp-name=erfnet_scnn_llamas
# Predicting lane points for testing
python main_landec.py --state=3 --batch-size=80 --continue-from=erfnet_scnn_llamas.pt --dataset=llamas --method=scnn --backbone=erfnet --mixed-precision
# Testing with official scripts
./autotest_llamas.sh erfnet_scnn_llamas val