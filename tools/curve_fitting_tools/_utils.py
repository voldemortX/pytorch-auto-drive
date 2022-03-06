from importmagician import import_from
with import_from('./'):
    from configs.lane_detection.common.datasets._utils import TUSIMPLE_ROOT, CULANE_ROOT, LLAMAS_ROOT


root_map = {
    'tusimple': TUSIMPLE_ROOT,
    'culane': CULANE_ROOT,
    'llamas': LLAMAS_ROOT
}

size_map = {
    'tusimple': (720, 1280),
    'culane': (590, 1640),
    'llamas': (717, 1276)
}
