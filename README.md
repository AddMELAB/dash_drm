Dash-DRM is a web app made with [Plotly Dash](https://plotly.com/dash/) designed for DRM analysis. You can visit [https://drm-dash-app.herokuapp.com/](https://drm-dash-app.herokuapp.com/) where the app is deployed on Heroku.

## Installation steps

- Install Python >= 3.6
- Clone the repository
- (Recommended) Create a new environment
- Install dependencies via `pip install -r requirements.txt`
- Run `app.py` and view the app on your localhost!

## Main functionalities

To browse the DRM dataset, use the sliders in the "DRM Daset" tab of the display pannel.
There are currently four different analysis supported by the app:

1. **PCA contrast:** Projects the dataset on its first three PCA components and shows these components as RGB channels. As the variance is maximized, this usually produces an image with high color contrast, which can be handy for visualization.
2. **Z-score analysis:** Computes a single channel image corresponding to the distance of each pixel to the global mean. This is can be useful to spot outlier pixels and can sometimes highlight defects in the microstructure.
3. **Texture prediction:** /!\ Only available on localhost with tensorflow > 2.1 installed /!\. Full orientation prediction in SS.316L using a CNN (see our publication for more details). Tensorflow is too heavy (400 Mb) to be hosted on Heroku. Need a future fix!
4. **Texture classification:** 3-class texture classification (111, 110, 100) in SS.316L using sklearn's MLP classifier.
5. Grain segmentation: Implements the LRC-MRM segmentation algorithm for autonomous grain identification. See our publication for more details.
