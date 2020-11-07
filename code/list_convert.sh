#!/bin/bash
echo Converting TuSimple and CULane lists...
cp -r ../../../dataset/culane/list ../../../dataset/culane/lists
python tusimple_list_convertor.py
python culane_list_convertor.py
echo Done.
