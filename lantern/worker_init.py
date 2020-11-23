from lantern import set_seeds


def worker_init(seed, worker_id):
    # TODO: investigate epoch
    set_seeds(
        seed * 2 ** 16
        + worker_id * 2 ** 24
        # + epoch
    )
