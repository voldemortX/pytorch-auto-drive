#!/bin/bash
echo Converting TuSimple and CULane lists...
cp -r ../../../dataset/culane/list ../../../dataset/culane/lists
python list_convertor.py
echo Done.
