from ._roots import TUSIMPLE_ROOT

dataset = dict(
    name='TuSimpleAsSegmentation',
    image_set='train',  # Only set for training. Testing will override this value by --state.
    root=TUSIMPLE_ROOT
)
