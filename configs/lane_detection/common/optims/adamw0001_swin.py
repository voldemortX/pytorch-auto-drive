optimizer = dict(
    name='torch_optimizer',
    torch_optim_class='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    parameters=[
        dict(
            params='absolute_pos_embed',
            weight_decay=0
        ),
        dict(
            params='relative_position_bias_table',
            weight_decay=0
        ),
        dict(
            params='norm',
            weight_decay=0
        ),
        dict(
            params='__others__'
        )
    ]
)
