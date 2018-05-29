
def csv_confusion_matrix(conf_matrix, clf_names):
    """Returns a string containing the provided confusion matrix in CSV format.
    Args:
        conf_matrix (list): confusion matrix.
        clf_names (list): model names.
    Returns:
        out (str): string containing conf_matrix as CSV table. """

    # Table header
    out = 'Family,' + ','.join(clf_names) + '\n'
    # Confusion matrix data
    for i in range(len(clf_names)):

        out += clf_names[i] + ','
        for j in range(len(clf_names)):
            out += str(conf_matrix[i][j]) + ','
        out = out[:-1] + '\n'

    # Replace model names by family names
    out = out.replace("RDA", "DA")
    out = out.replace("RF", "EN")
    out = out.replace("C5.0", "DT")
    out = out.replace("SVM_GAUSSIAN", "SVM")
    out = out.replace("MLP", "NNET")
    out = out.replace("NAIVE BAYES", "NB")
    out = out.replace("KNN", "NN")
    out = out.replace("GLMNET", "GLM")
    out = out.replace("SIMPLS", "PLSR")
    out = out.replace("PMR", "LMR")

    return out

def csv_report(report):
    """Returns a CSV file containing the report with precision, recall, and f-score.
    Args:
        report (str): report as provided by the classification_report() python utility from scikit-learn.
    Returns:
        str:  A CSV file containing
    """
    # Add 'Family" to the header
    report = 'Family ' + report

    # Replace model names by family names
    report = report.replace("RDA", "DA")
    report = report.replace("RF", "EN")
    report = report.replace("C5.0", "DT")
    report = report.replace("SVM_GAUSSIAN", "SVM")
    report = report.replace("MLP", "NNET")
    report = report.replace("NAIVE BAYES", "NB")
    report = report.replace("KNN", "NN")
    report = report.replace("GLMNET", "GLM")
    report = report.replace("SIMPLS", "PLSR")
    report = report.replace("PRM", "LMR")

    import re
    # Split report by lines
    report = report.split('\n')
    # and clean data
    # replace any sequence of whitespaces by single whitespace
    report = list(map(lambda t: re.sub(r"""\s+""", ' ', t), report))
    # Remove spaces at left
    report = list(map(lambda t: t.lstrip(' '), report))
    # Replace spaces with comma
    report = list(map(lambda t: t.replace(' ', ','), report))
    report = list(map(lambda t: t.replace(' ', ','), report))
    report = list(map(lambda t: t, report))
    # Remove empty strings
    report = list(filter(lambda t: t != '', report))
    # Fix 'avg/total' line
    report[-1] = report[-1].replace(',/,', '/')


    #Fix model names (replacing ',' by ' ')
    report[1:-1] = list(map(lambda t: re.sub(r"""([a-zA-Z]+),([a-zA-Z]+)""", r'\1 \2', t), report[1:-1]))


    return '\n'.join(report) + '\n'
