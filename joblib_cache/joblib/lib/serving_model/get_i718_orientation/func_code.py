# first line: 7
def get_i718_orientation(data, ds=1):
    data = data[::ds, ::ds]
    rx, ry, s0, s1 = data.shape

    train_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    train_path = os.path.join(train_path, 'trained_model/')
    # train_path = os.path.abspath()

    print('train path: ', train_path)
    model = CustomModel(train_path, s0=s0, s1=s1)
    print('CREATED MODEL')
    data = data.reshape((rx * ry, s0, s1))
    print('Feeding in: ', data.shape)
    dataset = tf.data.Dataset.from_tensor_slices(data).batch(50)
    print('Dataset: ', dataset)
    predictions = model.predict(dataset)
    print('GOT PREDS: ', predictions.shape)
    z_map = get_zmap(predictions, rx, ry)
    print('GOT ZMAP: ', z_map.shape)
    return z_map
