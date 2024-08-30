def _name(metric):
    if metric == "xsimpp":
        return "xsim(++)"
    if metric == "cosine_distance":
        return "cosdist"
    return metric

def _file(metric):
    return metric + "_matrix.csv"

