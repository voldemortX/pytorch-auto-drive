from ._utils import LLAMAS_ROOT

dataset = dict(
    name='LLAMAS_AsSegmentation',
    image_set='train',  # Only set for training. Testing will override this value by --state.
    root=LLAMAS_ROOT
)
