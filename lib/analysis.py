import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import NMF
from skimage.future.graph import RAG
import heapq
from skimage.segmentation import find_boundaries
from skimage.morphology import skeletonize
import pickle
import warnings

warnings.filterwarnings("ignore")


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


def get_316L_classification(data, ds):
    """
    ANN classification of 100/110/111 textures in 316L (trained on QR code data).
    The data MUST BE of shape (x, y, 13, 36). If it's not, no error will be raised
    but a message will be printed.
    """
    data = data[::ds, ::ds]
    rx, ry, s0, s1 = data.shape

    if (s0 != 13) | (s1 != 36):
        print('316L Classification expects (x, y, 13, 36) as dataset shape!')
        preds = np.zeros((rx, ry))
    else:
        model = pickle.load(open('./saved_models/ann_model_skl_0.sav', 'rb'))
        preds = model.predict(data.reshape((rx*ry, s0*s1))).reshape((rx, ry))

    preds_color = np.empty((rx, ry, 4))
    preds_color[preds == 0] = [255, 37, 21, 255]
    preds_color[preds == 1] = [30, 253, 8, 255]
    preds_color[preds == 2] = [38, 44, 247, 255]

    return preds_color


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


class NMFDataCompressor():
    def __init__(self, n_components):
        '''Instantiates the compressor model with set number of components'''
        self.compressor = NMF(n_components, max_iter=2_000)

    def fit(self, dataset, sample_size):
        '''
        - Extracts a random sample from the dataset
        - Fits the compressor model
        '''
        # Get data array from the dataset
        data = dataset.get('data')

        # Extract a random sample
        idx = np.arange(data.shape[0])
        np.random.shuffle(idx)
        data_extract = data[idx[:sample_size]]

        # Fit the compressor
        self.compressor.fit(data_extract)

    def transform(self, data):
        '''Returns a compressed feature vector representation of the data'''
        return self.compressor.transform(data)


def _run_lrc_mrm(dataset, cps, sample_size):
    '''
    Runs the LRC-MRM pipeline.

    Args:
        dataset: formatted DRM dataset with info on angular and pixel resolution
        cps: number of NMF components
        sample_size: sampling size for the calculation
    '''
    compressor = NMFDataCompressor(cps)
    compressor.fit(dataset, sample_size)
    compressed_dataset = compressor.transform(dataset['data'])
    dataset['data'] = compressed_dataset
    dataset = _fit_lrc_model(
        dataset,
        model=LogisticRegression(penalty='none', max_iter=2000),
        training_set_size=sample_size,
        test_set_size=sample_size,
    )
    dataset = _lrc_mrm_segmentation(dataset)
    return dataset


def _lrc_mrm_segmentation(dataset):
    '''
    Implementation of the multi-region merging segmentation controlled by
    a trained classifier model. Design of the function was originally inspired
    by the merge_hierarchical function of Skimage:
    (https://github.com/scikit-image/scikit-image/blob/master/skimage/)
    '''
    # Collect spatial resolution and data from dataset
    rx, ry = dataset.get('spatial_resol')
    data = dataset.get('data')
    model = dataset.get('lrc_model')

    # Define merging decision function
    mdf = lambda vect: model.predict_proba(np.atleast_2d(vect))[0, 1]

    # Initialize region agacency graph (RAG)
    rag, edge_heap, segments = _initialize_graph(rx, ry, data, model)

    # Start the region-merging algorithm
    while (len(edge_heap) > 0) and (edge_heap[0][0] < 0.5):

        # Pop the smallest edge from the heap if weight < 0.5
        smallest_weight, n1, n2, valid = heapq.heappop(edge_heap)

        # Check that the edge is valid
        if valid:

            # Make sure that n1 is the smallest regiom
            if (rag.nodes[n1]['count'] > rag.nodes[n2]['count']):
                n1, n2 = n2, n1

            # Update properties of n2
            rag.nodes[n2]['labels'] = (rag.nodes[n1]['labels']
                                       + rag.nodes[n2]['labels'])
            rag.nodes[n2]['count'] = (rag.nodes[n1]['count']
                                      + rag.nodes[n2]['count'])

            # Get new neighbors of n2
            n1_nbrs = set(rag.neighbors(n1))
            n2_nbrs = set(rag.neighbors(n2))
            new_neighbors = (n1_nbrs | n2_nbrs) - n2_nbrs - {n1, n2}

            # Disable edges of n1 in the heap list
            for nbr in rag.neighbors(n1):
                edge = rag[n1][nbr]
                edge['heap item'][3] = False

            # Remove n1 from the graph (edges are still in the heap list)
            rag.remove_node(n1)

            # Update new edges of n2
            for nbr in new_neighbors:
                rag.add_edge(n2, nbr)
                edge = rag[n2][nbr]
                master_n2 = rag.nodes[n2]['master']
                master_nbr = rag.nodes[nbr]['master']
                weight = mdf(_vector_similarity(master_n2, master_nbr))
                heap_item = [weight, n2, nbr, (weight < 0.5)]
                edge['heap item'] = heap_item
                # Push edges to the heap
                heapq.heappush(edge_heap, heap_item)

    # Compute grain segmentation map
    label_map = np.arange(segments.max() + 1)
    for ix, (n, d) in enumerate(rag.nodes(data=True)):
        for lab in d['labels']:
            label_map[lab] = ix
    segmentation = label_map[segments]

    # Compute grain boundary map
    gbs = skeletonize(find_boundaries(segmentation, mode='inner'))

    # Return updated dataset
    dataset['segmentation'] = segmentation.ravel()
    dataset['boundaries'] = gbs.ravel()

    return dataset


def _fit_lrc_model(dataset, model, training_set_size, test_set_size):
    '''Fits a model, computes precision, recall and accuracy'''
    training_set = _get_sample_set(dataset, training_set_size)
    model.fit(training_set['x'], training_set['y'])
    dataset['lrc_model'] = model
    return dataset


def _initialize_graph(rx, ry, data, model):
    '''
    Initializes the Region Adjacency Graph (RAG).
    '''
    # Define merging decision function
    mdf = lambda vect: model.predict_proba(np.atleast_2d(vect))[0, 1]

    # Initialize RAG
    xmap, ymap = get_xymaps(rx, ry)
    segments = np.arange(rx * ry).reshape((rx, ry))
    rag = RAG(segments)

    # Initialize nodes
    data_reshaped = data.reshape((rx, ry, data.shape[1]))
    for n in rag:
        rag.nodes[n].update({'labels': [n]})
    for index in np.ndindex(segments.shape):
        current = segments[index]
        rag.nodes[current]['count'] = 1
        rag.nodes[current]['master'] = data_reshaped[index]
        rag.nodes[current]['xpos'] = xmap[current]
        rag.nodes[current]['ypos'] = ymap[current]

    # Initialize edges
    edge_heap = []
    for n1, n2, d in rag.edges(data=True):
        master_x = rag.nodes[n1]['master']
        master_y = rag.nodes[n2]['master']
        weight = mdf(_vector_similarity(master_x, master_y))
        # Push the edge into the heap
        heap_item = [weight, n1, n2, (weight < 0.5)]
        d['heap item'] = heap_item
        heapq.heappush(edge_heap, heap_item)

    return rag, edge_heap, segments


def _get_sample_set(dataset, sample_size):
    '''
    Randomly extracts a training or test set from the dataset.
    - Class 0: pairs of adjacent voxels
    - Class 1: pairs of non-adjacent voxels
    '''
    # Collect data from the dataset
    rx, ry = dataset.get('spatial_resol')
    data = dataset.get('data')
    xmap, ymap = get_xymaps(rx, ry)

    # Extract adjacent sample
    Xclose, yclose = _get_adjacent_sample(
        rx, ry, data, sample_size, xmap, ymap)

    # Extract non-adjacent sample
    Xfar, yfar = _get_non_adjacent_sample(data, sample_size, xmap, ymap)

    # Stack both samples
    X = np.vstack((Xclose, Xfar))
    y = np.hstack((yclose, yfar))

    # Shuffle extracted set
    idx = np.arange(0, X.shape[0])
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]

    sample_set = {'x': X, 'y': y}

    return sample_set


def _get_adjacent_sample(rx, ry, data, sample_size, xmap, ymap):
    '''
    Samples Sbar, the distribution of adjacent pixel feature vectors.
    '''
    # Get set of random data
    X0, idx = shuffler(data, sample_size)

    # Modify location by 1 pixel
    modifiedX = xmap[idx] + (np.random.randint(0, 2, sample_size) * 2 - 1)
    modifiedX = np.clip(modifiedX.astype('int'), 0, rx - 1)
    modifiedY = ymap[idx] + (np.random.randint(0, 2, sample_size) * 2 - 1)
    modifiedY = np.clip(modifiedY.astype('int'), 0, ry - 1)

    # Find corresponding signal
    X1 = np.empty_like(X0)
    c = 0
    i = np.arange(rx * ry)
    for xc, yc in zip(modifiedX, modifiedY):
        u = np.zeros((rx, ry))
        u[xc, yc] = 1
        num = i[(u.ravel() == 1)]
        X1[c] = data[num]
        c += 1

    # Compute distance vectors
    Xclose = _vector_similarity(X1, X0)

    # Label as 0
    yclose = np.zeros(Xclose.shape[0], dtype=np.uint8)

    return Xclose, yclose


def _get_non_adjacent_sample(data, sample_size, xmap, ymap):
    '''
    Samples Dbar, the distribution of non-adjacent pixels.
    '''
    # Get random set of data and location of selected pixel pairs, twice
    X0, idx = shuffler(data, sample_size)
    locX_0, locY_0 = xmap[idx], ymap[idx]
    X1, idx = shuffler(data, sample_size)
    locX_1, locY_1 = xmap[idx], ymap[idx]

    # Compute distance vectors
    Xfar = _vector_similarity(X1, X0)

    # Filter out adjacent examples
    adjacent_filter = np.abs(locX_0 - locX_1) + np.abs(locY_0 - locY_1) < 2
    Xfar = Xfar[~adjacent_filter]

    # Label as 1
    yfar = np.ones(Xfar.shape[0], dtype=np.uint8)

    return Xfar, yfar


def _vector_similarity(a, b):
    '''Returns distance vector of two input feature vectors'''
    return np.square(np.subtract(a, b))


def get_xymaps(rx, ry):
    '''
    Helper function.
    '''
    xmap = np.empty((rx*ry))
    for k in range(rx):
        xmap[k*ry:(k+1)*ry] = np.array([k]*ry)
    xmap = xmap.reshape((rx, ry))
    xmap = xmap.ravel()
    ymap = np.empty((rx*ry))
    for k in range(rx):
        ymap[k*ry:(k+1)*ry] = np.arange(ry)
    ymap = ymap.reshape((rx, ry))
    ymap = ymap.ravel()
    return xmap, ymap


def shuffler(data, sample_size):
    '''
    Helper function.
    '''
    idx = np.arange(data.shape[0])
    np.random.shuffle(idx)
    sample = data[idx[:sample_size]]
    indeces = idx[:sample_size]
    return sample, indeces