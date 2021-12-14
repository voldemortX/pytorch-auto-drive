from ._utils import PASCAL_ROOT

dataset = dict(
    name='PASCAL_VOC_Segmentation',
    image_set='trainaug',  # Only set for training. Testing will override this value by --state.
    root=PASCAL_ROOT
)
