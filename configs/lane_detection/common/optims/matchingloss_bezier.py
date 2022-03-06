loss = dict(
    name='HungarianBezierLoss',
    weight=[0.4, 1],
    weight_seg=[0.4, 1],
    curve_weight=1,
    label_weight=0.1,
    seg_weight=0.75,
    alpha=0.8,
    num_sample_points=100,
    bezier_order=3,
    k=9,  # for the local maximum prior
    reduction='mean',
    ignore_index=255
)
