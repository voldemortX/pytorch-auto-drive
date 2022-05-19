lr = 0.0006
optimizer = dict(
    name='torch_optimizer',
    torch_optim_class='Adam',
    lr=lr,
    parameters=[
        dict(
            params='conv_offset',
            lr=lr * 0.1  # 1/10 lr for DCNv2 offsets
        ),
        dict(
            params='__others__'
        )
    ]
)
