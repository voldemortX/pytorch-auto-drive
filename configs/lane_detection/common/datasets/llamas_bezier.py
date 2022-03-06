from ._utils import LLAMAS_ROOT

dataset = dict(
    name='LLAMAS_AsBezier',
    image_set='train',  # Only set for training. Testing will override this value by --state.
    root=LLAMAS_ROOT,
    order=3,
    aux_segmentation=True
)
