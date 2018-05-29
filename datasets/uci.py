from os import listdir
from os.path import isfile, join

def load_raw_datasets(path = 'datasets/UCI/', min_idx = None, max_idx = None):
    """Load UCI datasets from specified path.
    Args:
        path (str): path containing .arff files.
        min_idx (int): minimum index to select the datasets.
        max_idx (int): maximum index to select the datasets.
    Returns:
        dss (list): loaded .arff datasets."""

    import arff
    # Get .arff file names
    ds_files = sorted([path + f for f in listdir(path) if isfile(join(path, f)) and f[-4:] == 'arff'])
    dss = []
    # The class (label) is sometimes specified with other than 'class'
    class_alias = ['class', 'symboling', 'survival_status', 'num', 'surgical_lesion', 'decision', 'contraceptive_method_used',
                   'class_attribute']

    # If no indexes specified, load all datasets
    if min_idx == None: min_idx = 0
    if max_idx == None: max_idx = len(ds_files)
    # Start dataset load
    for i in range(min_idx, max_idx):
        dsf = ds_files[i]
        try:
            print(str(i) + " " + dsf)
            # Load datasets
            dataset = arff.load(open(dsf, 'r'))
            # Extract meta-information
            meta_info = [attr for attr in dataset['attributes'] if attr[0].lower() not in class_alias]

            # Separate data and labels
            data = dataset['data']
            labels = [d[-1] for d in data]
            data = [d[:-1] for d in data]

            dss.append((data, labels, meta_info, dsf.split('/')[-1]))
        except Exception as e:
            print('--------------------------------')
            print(str(i) + " " + dsf)
            print(str(e))
            print('--------------------------------')

    return dss
