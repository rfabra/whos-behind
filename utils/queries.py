import numpy as np
import random

rnd_seed = 1


def uniform(dataset, size, meta_info):
    """Generates an artificial dataset following a uniform distribution,
    between the minimum and maximum values of each feature.

    Args:
        dataset (list): original dataset, used to extract possible values of each feature
        size (int): number of instances of the new surrogate dataset
        meta_info (list): attribute information from datasets: [(attr_name, [list_of_values])...]
    Returns:
        ds (list): unlabelled surrogate dataset
    """

    global rnd_seed
    rng = np.random.RandomState(rnd_seed)
    rnd_seed += 1
    ds_size = len(dataset)
    n_feats = len(meta_info)

    # Set of possible values per feature
    feat_vals = [list(set([dataset[d][f] for d in range(ds_size) if dataset[d][f] != None])) for f in range(n_feats)]
    # New surrogate dataset
    ds = [dict() for _ in range(size)]
    # Generate `size` instances
    for i in range(size):
        for f, (f_name, f_vals) in enumerate(meta_info):
            # Numerical
            if type(f_vals) is str and (f_vals.lower() == 'numerical' or f_vals.lower() == 'real'
                                        or f_vals.lower() == 'continuous' or f_vals.lower() == 'numeric'):
                vals = feat_vals[f]
                if len(vals) > 0:
                    min_val = np.min(vals)
                    max_val = np.max(vals)
                    ds[i][f_name] = rng.uniform(min_val, max_val)
                else:
                    ds[i][f_name] = None
            # Integer
            elif type(f_vals) is str and f_vals.lower() == 'integer':
                vals = feat_vals[f]
                if len(vals) > 0:
                    min_val = np.min(vals)
                    max_val = np.max(vals)
                    ds[i][f_name] = rng.randint(min_val, max_val)
                else:
                    ds[i][f_name] = None
            # Discrete
            elif type(f_vals) is list:
                if len(f_vals) > 0:
                    ds[i][f_name] = random.choice(f_vals)
                else:
                    ds[i][f_name] = None
            else:
                print("Something wrong at reading feature: {}, {}".format(f_name, f_vals))

    return ds
