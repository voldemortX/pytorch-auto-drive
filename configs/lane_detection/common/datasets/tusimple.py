from ._utils import TUSIMPLE_ROOT

dataset = dict(
    name='TuSimple',
    image_set='train',  # Only set for training. Testing will override this value by --state.
    root=TUSIMPLE_ROOT,
    padding_mask=False,
    process_points=True
)
