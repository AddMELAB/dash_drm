import os
import numpy as np
import tensorflow as tf
import warnings

warnings.filterwarnings("ignore")

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
        x_map, y_map, z_map = get_maps(predictions, rx, ry)
    return z_map


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


class CustomModel():
    def __init__(self, checkpoint, s0, s1, kernelNeurons1=64, kernelNeurons2=64, denseNeurons1=256, denseNeurons2=128):
        self.s0, self.s1 = s0, s1
        self.kernelNeurons1 = kernelNeurons1
        self.kernelNeurons2 = kernelNeurons2
        self.denseNeurons1 = denseNeurons1
        self.denseNeurons2 = denseNeurons2

        # @tf.function
        def preprocess(x):
            x = tf.multiply(x, 1 / 255.0)
            x = tf.expand_dims(x, axis=-1)
            return x

        inputs = tf.keras.Input(shape=(self.s0, self.s1))

        # Preprocessing
        lambds = tf.keras.layers.Lambda(lambda x: preprocess(x), output_shape=(self.s0, self.s1, 2))(inputs)

        # Define the output
        init = tf.keras.initializers.VarianceScaling(0.5)
        conv2d0 = tf.keras.layers.Conv2D(self.kernelNeurons1, (3, 3), strides=(1, 1), padding="same",
                                         activation='relu')(lambds)
        mpool1 = tf.keras.layers.MaxPooling2D((2, 2))(conv2d0)
        conv2d1 = tf.keras.layers.Conv2D(self.kernelNeurons2, (3, 3), strides=(1, 1), padding="same",
                                         activation='relu')(mpool1)
        mpool2 = tf.keras.layers.MaxPooling2D((2, 2))(conv2d1)
        globav = tf.keras.layers.Flatten()(mpool2)
        dense0 = tf.keras.layers.Dense(self.denseNeurons1 * 3, activation='relu', kernel_initializer=init)(globav)
        dense01 = tf.keras.layers.Dense(self.denseNeurons2 * 3, activation='relu', kernel_initializer=init)(dense0)
        output = tf.keras.layers.Dense(3, activation=None)(dense01)

        # Build
        self.model = tf.keras.Model(inputs=inputs, outputs=output)

        # # Reload weights
        self.model.load_weights(checkpoint).expect_partial()

        # Compile
        self.model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam(0.01),
        )
        print(self.model.summary())

    def predict(self, dataset):
        predictions = self.model.predict(dataset)
        return predictions


def get_maps(preds, rx, ry):
    xmap_pred, ymap_pred, zmap_pred = visualize(preds, rx, ry)
    xmap_pred = xmap_pred.numpy().reshape((rx, ry, 3))
    ymap_pred = ymap_pred.numpy().reshape((rx, ry, 3))
    zmap_pred = zmap_pred.numpy().reshape((rx, ry, 3))
    return xmap_pred, ymap_pred, zmap_pred

# @tf.function
def visualize(eulers, rx, ry, reshape=True):
    rot_mat = eulers_to_rot_mat(eulers)
    rot_mat_inv = tf.linalg.inv(rot_mat)
    output = tf.concat((symmetrize(rot_mat_inv[:,:,0]),
                        symmetrize(rot_mat_inv[:,:,1]),
                        symmetrize(rot_mat_inv[:,:,2])), axis=1)
    xmap = colorize(output[:,0:3])
    ymap = colorize(output[:,3:6])
    zmap = colorize(output[:,6:9])
    return xmap, ymap, zmap

def symmetrize(indeces):
    indeces = tf.abs(indeces)
    norms = tf.square(indeces[:,0])+tf.square(indeces[:,1])+tf.square(indeces[:,2])
    norms = tf.expand_dims(norms, 1)
    indeces = indeces/norms
    indeces = tf.sort(indeces, axis=1, direction='ASCENDING')
    return indeces

def colorize(indeces):
    a = tf.abs(tf.subtract(indeces[:, 2], indeces[:, 1]))
    b = tf.abs(tf.subtract(indeces[:, 1], indeces[:, 0]))
    c = indeces[:, 0]
    rgb = tf.concat((tf.expand_dims(a, -1),
                     tf.expand_dims(b, -1),
                     tf.expand_dims(c, -1)), axis=1)
    # Normalization
    maxes = tf.reduce_max(rgb, axis=1)
    a = a/maxes
    b = b/maxes
    c = c/maxes
    rgb = tf.concat((tf.expand_dims(a, -1),
                     tf.expand_dims(b, -1),
                     tf.expand_dims(c, -1)), axis=1)
    return rgb

def eulers_to_rot_mat(eulers):
    i1 = eulers[:,0]
    i2 = eulers[:,1]
    i3 = eulers[:,2]
    i1c = tf.cos(i1)
    i1s = tf.sin(i1)
    i2c = tf.cos(i2)
    i2s = tf.sin(i2)
    i3c = tf.cos(i3)
    i3s = tf.sin(i3)
    x00 = i1c*i2c*i3c-i1s*i3s
    x01 = -i3c*i1s-i1c*i2c*i3s
    x02 = i1c*i2s
    x10 = i1c*i3s+i2c*i3c*i1s
    x11 = i1c*i3c-i2c*i1s*i3s
    x12 = i1s*i2s
    x20 = -i3c*i2s
    x21 = i2s*i3s
    x22 = i2c
    c0 = tf.stack((x00,x01,x02), axis=1)
    c1 = tf.stack((x10,x11,x12), axis=1)
    c2 = tf.stack((x20,x21,x22), axis=1)
    rot_mat = tf.stack((c0,c1,c2), axis=1)
    return rot_mat