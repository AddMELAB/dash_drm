# first line: 18
def get_pca_contrast(data, ds=1, cps=3):
    """Computes a basic PCA contrast with 3 first components as RGB"""
    data = data[::ds, ::ds]
    rx, ry, s0, s1 = data.shape
    data = data.reshape((rx * ry, s0*s1))
    data = MinMaxScaler().fit_transform(data)
    data = PCA(cps).fit_transform(data)
    data = MinMaxScaler().fit_transform(data)
    data = data.reshape((rx, ry, cps))
    data = data * 255
    return data
