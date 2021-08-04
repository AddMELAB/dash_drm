![alt text](https://raw.githubusercontent.com/MalloryWittwer/dash_drm/main/static/app_overview.jpg)

Dash-DRM is a web app made with [Plotly Dash](https://plotly.com/dash/) designed for DRM analysis. The app can be deployed locally and viewed in the browser. As an alternative, a simpler version of the app is deployed on Heroku at [https://drm-dash-app.herokuapp.com/](https://drm-dash-app.herokuapp.com/). The Heroku version does not implement orientation prediction for I718 because Tensorflow is too heavy for Heroku's 500 Mb free deployment plan. Also, it can only support uploading small DRM datasets for the same reason.

## Installation steps

- Install Python >= 3.6
- Clone the repository
- (Recommended) Create a new environment
- Install dependencies via `pip install -r requirements.txt`
- Run `app.py` and view the app on your localhost!

## Main functionalities

Start by drag-and dropping a DRM dataset. The expected format of the dataset is **a 4D matrix stored in NPY format** where the axes represent (resolution_x, resolution_y, theta, phi). Once loaded, to browse the DRM dataset, use the sliders in the "DRM Daset" tab of the display pannel.

There are currently four different analysis supported by the app:

1. **PCA contrast:** Projects the dataset on its first three PCA components and shows these components as RGB channels. As the variance is maximized, this usually produces an image with high color contrast, which can be handy for visualization.
2. **Z-score analysis:** Computes a single channel image corresponding to the distance of each pixel to the global mean. This is can be useful to spot outlier pixels and can sometimes highlight defects in the microstructure.
3. **I718 orientation prediction:** Requires tensorflow 2.1 (only on localhost). Full orientation prediction in SS.316L using a CNN (see our publication for more details).Only the Z-map is displayed. Tensorflow is too heavy (400 Mb) to be hosted on Heroku. Need a future fix!
4. **316L texture classification:** 3-class texture classification (111, 110, 100) in SS.316L using sklearn's MLP classifier.
5. **Grain segmentation:** Implements the LRC-MRM segmentation algorithm for autonomous grain identification (see [our publication](https://doi.org/10.1016/j.matchar.2021.110978) for more details).

To downscale the DRM dataset spatially before performing the analysis (to speed up calculations), use the "Downscale" slider. A value of 1 corresponds to the native resolution, while 8 will select one out of eight pixels in both the X and Y axes (resulting in 64X data reduction).

For any inquiries, please contact addme.lab@ntu.edu.sg.
