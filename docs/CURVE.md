## Generate Bézier labels

Generation script:

```
    python ./tools/curve_fitting_tools/gen_bezier_annotations.py 
            --dataset=<dataset name> 
            --image-set=<image set name: train\test\val> 
            --order=<the order of Bézier curves>
```

**Notes:**

1. We generate Bézier control points from original key points without normalization. 
   The normalized control points can be obtained by using `--norm` in the generated script.
   
2. The test set of LLAMAS dataset is unavailable, thus we cannot obtain the LLAMAS test set 's Bézier labels.

3. In CurveLanes dataset, some lanes were marked by sparse key points (for instance, 2 and 3 points) , therefore, before obtain Bézier labels we interpolate lanes.  

## Bézier label format
All labels are saved in a json file, named `<image-set>_<order>.json`.

```
   {"raw_file":filename1, "Bezier_control_points": [[...],[...], ..., [...]}
   {"raw_file":filename2, "Bezier_control_points": [[...],[...], ..., [...]}
   ...
   {"raw_file":filenamen, "Bezier_control_points": [[...],[...], ..., [...]}
```

## Upper-bound test script

This script is used to obtain prediction results from fitted curves.

```
   python ./tools/curve_fitting_tools/upperbound.py 
          --dataset=<dataset name>
          --state=<1: test set/2: val test>
          --fit-function=<bezier/poly>
          --num-points=<the number of generating key points>
          --order=<the order of generating curves>
```

We still need to run `autotest_<culane\llamas\tusimple>.sh` to get F1/Accuracy.

LPD metric script:

```
   python ./tools/curve_fitting_tools/lpd_mertic.py 
          --pred=<.json with the predictions>
          --gt=<.json with the gt>
          --gt-type='tusimple'
```

The `lpd_metric.py` is used to get lpd metric, which was employed in [PolyLaneNet](https://arxiv.org/abs/2004.10924). 

We copy this test script from this [repo](https://github.com/lucastabelini/PolyLaneNet), you can find more information in this [issue](https://github.com/lucastabelini/PolyLaneNet/issues/50).


**Notes:**

1. The upper-bound test on TuSimple dataset does not require `--num-points`.

2. The lpd metric only supports TuSimple dataset.

3. Bézier curves are simply fitted with least-squares, which is not optimal.

## Upper-bounds on test set (except LLAMAS uses val)

**100 sample points for the CULane eval.**

### CULane F1
| Order | Bézier | polynomial |
| :---: | :---: | :---: |
| 1st | 99.6024 | 99.6177 |
| 2nd | 99.9733 | 99.9685 |
| 3rd | 99.9962 | 99.9971 |
| 4th | 99.9962 | 99.9990 |
| 5th | 99.9847 | 99.9990 |

### TuSimple Accuracy
| Order | Bézier | polynomial |
| :---: | :---: | :---: |
| 1st | 96.4738 | 97.9629 |
| 2nd | 98.4588 | 99.0760 |
| 3rd | 99.5239 | 99.7463 |
| 4th | 99.8120 | 99.9498 |
| 5th | 99.9106 | 99.9883 |

### LLAMAS F1
| Order | Bézier | polynomial |
| :---: | :---: | :---: |
| 1st | 98.8978 | 99.1409 |
| 2nd | 99.4408 | 99.5178 |
| 3rd | 99.7191 | 99.6259 |
| 4th | 99.7987 | 99.6961 |
| 5th | 99.8501 | 99.7501 |

### TuSimple LPD
| Order | Bézier | polynomial |
| :---: | :---: | :---: |
| 1st | 0.956382 | 1.415590 |
| 2nd | 0.652477 | 0.944469 |
| 3rd | 0.471154 | 0.557482 |
| 4th | 0.314884 | 0.329481 |
| 5th | 0.238662 | 0.208554 |

*LPD metric: lower is better.*
