# first line: 69
def get_grain_segmentation(data, ds=1):
    zs = data[::ds, ::ds]
    rx, ry, s0, s1 = zs.shape

    # This is just a convention
    dataset = {
        'data': zs.reshape((rx * ry, s0 * s1)),
        'spatial_resol': (rx, ry),
        'angular_resol': (s0, s1),
    }

    dataset = _run_lrc_mrm(
        dataset,  # Dataset, formatted as above
        30,  # NMF components
        min(2000, rx*ry),  # Sampling size
    )

    # Reshape the maps
    segmentation = dataset.get('segmentation').reshape((rx, ry))
    gbs = dataset.get('boundaries').reshape((rx, ry))

    # Rescale segmentation
    segmentation = segmentation - segmentation.min()
    segmentation = segmentation / segmentation.max()
    segmentation = plt.cm.jet(segmentation)
    segmentation[gbs] = [1, 1, 1, 0]
    segmentation = segmentation * 255

    return segmentation
