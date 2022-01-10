from ._utils import SYNTHIA_ROOT

dataset = dict(
    name='SYNTHIA_Segmentation',
    image_set='train',  # Only set for training. Testing will override this value by --state.
    root=SYNTHIA_ROOT
)
