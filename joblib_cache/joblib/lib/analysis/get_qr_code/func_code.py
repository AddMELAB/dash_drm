# first line: 66
def get_qr_code(data, ds):
    data = data[::ds, ::ds]
    rx, ry, s0, s1 = data.shape

    model = pickle.load(open('./saved_models/ann_model_skl_0.sav', 'rb'))
    
    preds = model.predict(data.reshape((rx*ry, s0*s1))).reshape((rx, ry))

    preds_color = np.empty((rx, ry, 4))
    preds_color[preds == 0] = [255, 37, 21, 255]
    preds_color[preds == 1] = [30, 253, 8, 255]
    preds_color[preds == 2] = [38, 44, 247, 255]

    return preds_color
