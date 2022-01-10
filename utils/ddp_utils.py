# Simple wrapper-like utils copied from pytorch/vision
import os
import torch
import torch.distributed as dist


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def reduce_dict(input_dict, average=True, to_item=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
        to_item (bool): whether convert tensor to its item (used for logging)
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v.item() if to_item else v for k, v in zip(names, values)}
    return reduced_dict


def init_distributed_mode(cfg):
    if cfg['state'] == 0 and cfg['world_size'] > 0:  # Restrict ddp to training
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            cfg['rank'] = int(os.environ["RANK"])
            cfg['world_size'] = int(os.environ['WORLD_SIZE'])
            cfg['gpu'] = int(os.environ['LOCAL_RANK'])
        elif 'SLURM_PROCID' in os.environ:
            cfg['rank'] = int(os.environ['SLURM_PROCID'])
            cfg['gpu'] = cfg['rank'] % torch.cuda.device_count()
        elif hasattr(cfg, "rank"):
            pass
        else:
            print('Not using distributed mode')
            cfg['distributed'] = False
            return
    else:
        print('Not using distributed mode')
        cfg['distributed'] = False
        return

    cfg['distributed'] = True
    torch.cuda.set_device(cfg['gpu'])
    cfg['dist_backend'] = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        cfg['rank'], cfg['dist_url']), flush=True)
    torch.distributed.init_process_group(backend=cfg['dist_backend'], init_method=cfg['dist_url'],
                                         world_size=cfg['world_size'], rank=cfg['rank'])
    setup_for_distributed(cfg['rank'] == 0)
