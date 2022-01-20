from ._utils import TUSIMPLE_ROOT

dataset = dict(
    name='TuSimpleAsBezier',
    image_set='train',  # Only set for training. Testing will override this value by --state.
    root=TUSIMPLE_ROOT,
    order=3,
    aux_segmentation=True
)
