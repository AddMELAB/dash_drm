![alt text](https://raw.githubusercontent.com/MalloryWittwer/dash_drm/main/static/app_overview.jpg)

This is a web app made with [Plotly Dash](https://plotly.com/dash/) designed for DRM analysis. The app can be deployed locally and viewed in a browser. Alternatively, a simpler version of the app can be deployed on Heroku from the [heroku_branch](https://github.com/AddMELAB/dash_drm/tree/heroku_branch) branch. For details on how to deploy on Heroku, see [their documentation](https://devcenter.heroku.com/articles/getting-started-with-python). The Heroku version does not implement full orientation prediction for I718 (or 316L) because Tensorflow is too heavy for Heroku's 500 Mb free deployment plan. Also, it can only support uploading small DRM datasets for the same reason. And finally, sometimes I noticed the Plotly graphs didn't show up properly online. So, overall, I'd recommend sticking to the main branch and running the app locally.

## Installation steps

- Install Python >= 3.6
- Clone the repository
- (Recommended) Create a new environment
- Install dependencies via `pip install -r requirements.txt`
- Run `app.py` and view the app in your browser (localhost)!

## Main functionalities

Start by drag-and dropping a DRM dataset. The expected format of the dataset is **a 4D matrix stored in NPY format** (.npy file) where the axes represent (resolution_x, resolution_y, theta, phi). To save datasets in this format, consider using the Save Matrix option in our [Tkinter GUI](https://github.com/AddMELAB/pydrm_GUI). Once loaded, to browse the DRM dataset, use the sliders in the "DRM Daset" tab of the display pannel.

There are currently a few different analysis supported by the app:

1. **PCA contrast:** Mostly a toy analysis. Projects the dataset on its first three PCA components and shows these components as RGB channels. As the variance is maximized, this usually produces an image with vivid colors, which can be handy for a quick visualization.
2. **Z-score analysis:** Computes a single channel image corresponding to the distance of each pixel to the global mean. This is can be useful to spot outlier pixels and can sometimes highlight defects in the microstructure.
3. **316L texture classification:** 3-class texture classification (111, 110, 100) in SS.316L using sklearn's MLP classifier. It leverages the trained model in the `saved_models` folder for inference.
4. **Grain segmentation:** Implements the LRC-MRM segmentation algorithm for autonomous grain identification (see [our publication](https://doi.org/10.1016/j.matchar.2021.110978) for more details). Be aware that the algorithm can be slow for large datasets!
5. **I718 orientation prediction (Z map):** Tested with Tensorflow 2.1 (CPU version). Full orientation prediction using a CNN. Only the IPF Z map is displayed but that could be easily changed in the code (the code predicts the full orientaiton, not just the Z map). The CNN model is reloaded from the `trained_model_i718` folder.
6. **Stainless steel 316L orientation prediction (Y map)**: same concept but for stainless steel.

To downscale the DRM dataset spatially before performing any of the analysis above (to speed up calculations), use the "Downscale" slider. A value of 1 corresponds to the native resolution, while 8 will select one out of eight pixels in both the X and Y axes (resulting in 64X data reduction).

Finally, note that the code is set up so that it should be quite straight forward to add new analysis to the dropdown menu. For this, you will need to encapsulate a function that takes as input the drm dataset as 4D npy matrix and (optional) the downscale factor, and somehow returns an image in the range 0-255 that can be plotted using [imshow](https://plotly.com/python/imshow/) from Plotly. Import that function in `app.py`, cache it if necessary (if it takes more than a few seconds to run) and add an entry for that function to the dropdown menu.
