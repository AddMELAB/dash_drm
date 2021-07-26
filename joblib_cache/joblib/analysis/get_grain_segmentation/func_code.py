# first line: 62
def get_grain_segmentation(data, ds=1):
    zs = data[::ds, ::ds]
    print('\n> Loaded DRM data: ', zs.shape)
    rx, ry, s0, s1 = zs.shape

    # This is just a convention
    dataset = {
        'data': zs.reshape((rx * ry, s0 * s1)),
        'spatial_resol': (rx, ry),
        'angular_resol': (s0, s1),
    }

    print('\n> Started segmentation. Fitting NMF model...')
    dataset = _run_lrc_mrm(
        dataset,  # Dataset, formatted as above
        50,  # NMF components
        2000,  # Sampling size
    )
    print('\n> Finished segmentation!')

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
