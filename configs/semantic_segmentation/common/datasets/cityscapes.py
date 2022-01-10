from ._utils import CITYSCAPES_ROOT

dataset = dict(
    name='CityscapesSegmentation',
    image_set='train',  # Only set for training. Testing will override this value by --state.
    root=CITYSCAPES_ROOT
)
