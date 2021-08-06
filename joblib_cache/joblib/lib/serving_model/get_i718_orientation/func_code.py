# first line: 8
def get_i718_orientation(data, ds=1):
    """
    Z-map orientation prediction in Inconel 718 (Tensorflow implementation).
    The model predicts the full orientation; feel free to modify the app to
    display X and Y maps as well or Euler maps.
    I would advise tf version 2.1.0; I've had issues with more recent versions.
    The data MUST BE of shape (x, y, 6, 72)!
    """
    data = data[::ds, ::ds]
    rx, ry, s0, s1 = data.shape

    if (s0 != 6) | (s1 != 72):
        print('I718 grain orientation prediction expects (x, y, 6, 72) as dataset shape!')
        z_map = np.zeros((rx, ry))
    else:
        train_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        train_path = os.path.join(train_path, 'trained_model_i718/')
        model = CustomModel(train_path, s0=s0, s1=s1)
        data = data.reshape((rx * ry, s0, s1))
        dataset = tf.data.Dataset.from_tensor_slices(data).batch(50)
        predictions = model.predict(dataset)
        z_map = get_zmap(predictions, rx, ry)
    return z_map
