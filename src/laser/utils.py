def _name(metric):
    if metric == "xsimpp":
        return "xsim(++)"
    if metric == "cosine_distance":
        return "cosdist"
    return metric

def _display_name(metric):
    if metric == "xsim":
        return "xSIM"
    if metric == "xsimpp":
        return "xSIM++"
    if metric == "cosine_distance":
        return "Cosine Distance"
    return metric.upper()

def _file(metric):
    return metric + "_matrix.csv"

