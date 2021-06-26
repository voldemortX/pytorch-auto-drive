# Trained weights: vgg16_baseline_llamas_20210625.pt
# Training
python main_landec.py --epochs=18 --lr=0.3 --batch-size=20 --dataset=llamas --method=baseline --backbone=vgg16 --mixed-precision --exp-name=vgg16_baseline_llamas
# Predicting lane points for testing
python main_landec.py --state=3 --batch-size=80 --continue-from=vgg16_baseline_llamas.pt --dataset=llamas --method=baseline --backbone=vgg16 --mixed-precision
# Testing with official scripts
./autotest_llamas.sh vgg16_baseline_llamas val
