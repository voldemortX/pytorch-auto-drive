from ._utils import CULANE_ROOT

dataset = dict(
    name='CULaneAsSegmentation',
    image_set='train',  # Only set for training. Testing will override this value by --state.
    root=CULANE_ROOT
)
