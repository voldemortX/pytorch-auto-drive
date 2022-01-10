from ._utils import GTAV_ROOT

dataset = dict(
    name='GTAV_Segmentation',
    image_set='train',  # Only set for training. Testing will override this value by --state.
    root=GTAV_ROOT
)
