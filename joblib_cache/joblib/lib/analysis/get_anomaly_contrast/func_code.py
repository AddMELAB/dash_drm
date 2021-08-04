# first line: 31
def get_anomaly_contrast(data, ds=1):
    """Computes a basic Z-score image (1 dimensional)"""
    zs = data[::ds, ::ds]
    rx, ry, s0, s1 = zs.shape
    zs = zs.reshape((rx * ry, s0 * s1))
    zs = PCA(2).fit_transform(zs)
    zs = StandardScaler().fit_transform(zs)
    zs = np.abs(zs)
    zs = np.sqrt(np.sum(np.square(zs), axis=1))
    zs = MinMaxScaler().fit_transform(zs.reshape(-1,1))
    zs = zs.reshape((rx, ry)) * 255
    return zs
