# Setup to temporarily avoid recursive imports


def lane_pruning(existence, existence_conf, max_lane):
    # Prune lanes based on confidence (a max number constrain for lanes in an image)
    # Maybe too slow (but should be faster than topk/sort),
    # consider batch size >> max number of lanes
    while (existence.sum(dim=1) > max_lane).sum() > 0:
        indices = (existence.sum(dim=1, keepdim=True) > max_lane).expand_as(existence) * \
                  (existence_conf == existence_conf.min(dim=1, keepdim=True).values)
        existence[indices] = 0
        existence_conf[indices] = 1.1  # So we can keep using min

    return existence, existence_conf
