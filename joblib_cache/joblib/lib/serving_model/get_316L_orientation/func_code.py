# first line: 33
def get_316L_orientation(data, ds=1):
    """
    Z-map orientation prediction in Inconel 718 (Tensorflow implementation).
    The model predicts the full orientation; feel free to modify the app to
    display X and Y maps as well or Euler maps.
    I would advise tf version 2.1.0; I've had issues with more recent versions.
    The data MUST BE of shape (x, y, 13, 36)!
    """
    data = data[::ds, ::ds]
    rx, ry, s0, s1 = data.shape

    if (s0 != 13) | (s1 != 36):
        print('316L grain orientation prediction expects (x, y, 13, 36) as dataset shape!')
        z_map = np.zeros((rx, ry))
    else:
        train_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        train_path = os.path.join(train_path, 'trained_model_316L/')
        model = CustomModel(train_path, s0=s0, s1=s1)
        data = data.reshape((rx * ry, s0, s1))
        dataset = tf.data.Dataset.from_tensor_slices(data).batch(50)
        predictions = model.predict(dataset)
        x_map, y_map, z_map = get_maps(predictions, rx, ry)
    return y_map
