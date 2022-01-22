## Generate Bézier labels

Generation script:

```
    python ./tools/curve_fitting_tools/gen_Bézier_annotations.py 
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
   {"raw_file":filename1, "Bézier_control_points": [[...],[...], ..., [...]}
   {"raw_file":filename2, "Bézier_control_points": [[...],[...], ..., [...]}
   ...
   {"raw_file":filenamen, "Bézier_control_points": [[...],[...], ..., [...]}
```

## Upper-bound test script

This script is used to obtain prediction results from fitted curves.

```
   python ./tools/curve_fitting_tools/upperbound.py 
          --dataset=<dataset name>
          --state=<1: test set/2: val test>
          --fit-function=<Bézier/poly>
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

### CULane F1
| Order | Bézier | polynomial |
| :---: | :---: | :---: |
| 1st | 99.60242549 | 99.61768015 |
| 2nd | 99.97330435 | 99.96853727 |
| 3rd | 99.99618634 | 99.99713975 |
| 4th | 99.99618634 | 99.99904658 |
| 5th | 99.98474534 | 99.99904658 |

### TuSimple Accuracy
| Order | Bézier | polynomial |
| :---: | :---: | :---: |
| 1st | 0.964899781 | 0.979629356 |
| 2nd | 0.984646195 | 0.990759581 |
| 3rd | 0.99526131 | 0.997463498 |
| 4th | 0.998124636 | 0.999498263 |
| 5th | 0.999098691 | 0.999883392 |

### LLAMAS F1
| Order | Bézier | polynomial |
| :---: | :---: | :---: |
| 1st | 0.9889 | 0.9914 |
| 2nd | 0.9944 | 0.9952 |
| 3rd | 0.9972 | 0.9963 |
| 4th | 0.9980 | 0.9970 |
| 5th | 0.9985 | 0.9975 |

### TuSimple LPD
| Order | Bézier | polynomial |
| :---: | :---: | :---: |
| 1st | 0.956382 | 1.415590 |
| 2nd | 0.652477 | 0.944469 |
| 3rd | 0.471154 | 0.557482 |
| 4th | 0.314884 | 0.329481 |
| 5th | 0.238662 | 0.208554 |

*LPD metric: lower is better.*
