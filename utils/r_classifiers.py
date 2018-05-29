
def model(method_name, train, tr_labels, test, tr_file, tr_l_file, ts_file, pred_file):
    """ Learns a model from provided data and stores its predictions.
    Args:
        method_name (str): name of the method.
        train (list): training instances.
        tr_labels (list): list of labels for training instances.
        tr_file (str): file name to save training instances.
        tr_l_file (str): file name to save training labels.
        ts_file (str): file name to save test instances.
        pred_file (str): file to save the predictions of the model over test data."""

    from subprocess import call
    import numpy as np

    # Save numpy arrays
    tr_file =tr_file
    tr_l_file = tr_l_file
    ts_file =  ts_file
    pred_file =  pred_file
    pred_file.replace('.npy', '.csv')

    np.save(tr_file, train)
    np.savetxt(tr_l_file, tr_labels, fmt="%s")
    np.save(ts_file, test)

    # Call R script to learn the model and predict test labels
    print("Command: Rscript R_model.R {} {} {} {} {}".format(method_name, tr_file,tr_l_file, ts_file, pred_file))
    call(['Rscript', 'R_model.R', method_name, tr_file, tr_l_file, ts_file, pred_file])

    # Save predictions
    f = open(pred_file)
    preds = f.readlines()
    preds = list(map(lambda t: t.strip('\n"'), preds))
    f.close()

    return np.array(preds)

def prepare_file(ds_name, o_clf_name, s_clf_name, suffix):
    """Returns file names for model predictions functionality.
    Args:
        ds_name: name of the dataset.
        o_clf_name: name of the oracle model.
        s_clf_name: name of the surrogate model.
        suffix: other data to appear at the end of the file name.
    Returns:
        Name of the file, according to specified information."""

    if s_clf_name == None:
        return '{}_{}_{}.npy'.format(ds_name, o_clf_name, suffix)
    else:
        return '{}_{}_{}_{}.npy'.format(ds_name, o_clf_name, s_clf_name, suffix)
