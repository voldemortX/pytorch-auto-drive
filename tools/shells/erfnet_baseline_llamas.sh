# Trained weights: erfnet_baseline_llamas_20210625.pt
# Training
python main_landec.py --epochs=10 --lr=0.5 --batch-size=20 --dataset=llamas --method=baseline --backbone=erfnet --mixed-precision --exp-name=erfnet_baseline_llamas
# Predicting lane points for testing
python main_landec.py --state=3 --batch-size=80 --continue-from=erfnet_baseline_llamas.pt --dataset=llamas --method=baseline --backbone=erfnet --mixed-precision
# Testing with official scripts
./autotest_llamas.sh erfnet_baseline_llamas val