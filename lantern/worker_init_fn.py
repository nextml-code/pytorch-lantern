from lantern import set_seeds


def worker_init_fn(seed):
    def worker_init(worker_id):
        set_seeds(seed * 2**16 + worker_id * 2**24)

    return worker_init
