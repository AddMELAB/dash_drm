# first line: 7
def get_prediction(data, ds=1):
    data = data[::ds, ::ds]
    rx, ry, s0, s1 = data.shape
    train_path = 'C:/Users/mallo/Documents/GitHub/dash_tests/trained_model/'
    print('train path: ', train_path)
    model = CustomModel(train_path, s0=s0, s1=s1)
    print('SUCCESSFULLY CREATED MODEL')
    data = data.reshape((rx * ry, s0, s1))
    print('Feeding in: ', data.shape)
    dataset = tf.data.Dataset.from_tensor_slices(data).batch(50)
    print('Dataset: ', dataset)
    predictions = model.predict(dataset)
    print('GOT PREDS: ', predictions.shape)
    z_map = get_zmap(predictions, rx, ry)
    print('GOT ZMAP: ', z_map.shape)
    return z_map
