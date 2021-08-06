![alt text](https://raw.githubusercontent.com/MalloryWittwer/dash_drm/main/static/app_overview.jpg)

This is a web app made with [Plotly Dash](https://plotly.com/dash/) designed for DRM analysis. The app can be deployed locally and viewed in the browser. As an alternative, a simpler version of the app can be deployed on Heroku from the [heroku_branch](https://github.com/AddMELAB/dash_drm/tree/heroku_branch) branch. For details on how to deploy on Heroku, see [their documentation](https://devcenter.heroku.com/articles/getting-started-with-python). The Heroku version does not implement full orientation prediction for I718 (or 316L) because Tensorflow is too heavy for Heroku's 500 Mb free deployment plan. Also, it can only support uploading small DRM datasets for the same reason. And finally, sometimes I noticed the Plotly graphs didn't show up properly online.

## Installation steps

- Install Python >= 3.6
- Clone the repository
- (Recommended) Create a new environment
- Install dependencies via `pip install -r requirements.txt`
- Run `app.py` and view the app in your browser (localhost)!

## Main functionalities

Start by drag-and dropping a DRM dataset. The expected format of the dataset is **a 4D matrix stored in NPY format** (.npy file) where the axes represent (resolution_x, resolution_y, theta, phi). To save datasets as .npy if you are unfamiliar with it, consider using the Save Matrix option in the [Tkinter GUI](https://github.com/AddMELAB/pydrm_GUI). Once loaded, to browse the DRM dataset, use the sliders in the "DRM Daset" tab of the display pannel.

There are currently a few different analysis supported by the app:

1. **PCA contrast:** Projects the dataset on its first three PCA components and shows these components as RGB channels. As the variance is maximized, this usually produces an image with high color contrast, which can be handy for visualization.
2. **Z-score analysis:** Computes a single channel image corresponding to the distance of each pixel to the global mean. This is can be useful to spot outlier pixels and can sometimes highlight defects in the microstructure.
3. **316L texture classification:** 3-class texture classification (111, 110, 100) in SS.316L using sklearn's MLP classifier.
4. **Grain segmentation:** Implements the LRC-MRM segmentation algorithm for autonomous grain identification (see [our publication](https://doi.org/10.1016/j.matchar.2021.110978) for more details).
5. **I718 orientation prediction (Z map):** Requires tensorflow 2.1 (only on localhost). Full orientation prediction using a CNN.Only the IPF Z map is displayed but that can be easily changed in the code (the code predicts the full orientaiton, not just the Z map). Tensorflow is too heavy (400 Mb) to be hosted on Heroku. Need a future fix!
6. **Stainless steel 316L orientation prediction (Y map)**: same concept.

To downscale the DRM dataset spatially before performing the analysis (to speed up calculations), use the "Downscale" slider. A value of 1 corresponds to the native resolution, while 8 will select one out of eight pixels in both the X and Y axes (resulting in 64X data reduction).
