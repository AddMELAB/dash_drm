import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
from numpy.core.numeric import binary_repr
from numpy.lib.polynomial import _binary_op_dispatcher
import plotly.express as px
from joblib import Memory
import base64
import io
import numpy as np

from lib.analysis import (  # Feel free to add your analysis function here!
    get_pca_contrast, 
    get_anomaly_contrast, 
    get_grain_segmentation, 
    get_316L_classification,
)
# from lib.serving_model import get_i718_orientation

# Cached functions (recommended for heavy computations):
memory = Memory("./joblib_cache", bytes_limit=3000000000, verbose=3)
get_pca_contrast_cached = memory.cache(get_pca_contrast)
get_anomaly_contrast_cached = memory.cache(get_anomaly_contrast)
get_qr_code_cached = memory.cache(get_316L_classification)
get_grain_segmentation_cached = memory.cache(get_grain_segmentation)
# get_i718_orientation_cached = memory.cache(get_i718_orientation)

# Global variables:
n_clicks_trakcer = None
prediction = np.zeros((100, 100, 3))
string = np.zeros((100, 100, 10, 36))

# List of available tasks:
list_of_tasks = {
    0 : ("PCA contrast", get_pca_contrast_cached),
    1 : ("Z-score analysis", get_anomaly_contrast_cached),
    2 : ("316L ([100]/[110]/[111]) classification", get_qr_code_cached),
    3 : ("Grain segmentation", get_grain_segmentation_cached),
    # 4 : ("Inconel 718 orientation", get_i718_orientation_cached)
 }

# Initialize app
app = dash.Dash(__name__)

server = app.server  # Used in Heroku deployment only

# LAYOUT
app.layout = html.Div([
    # Header
    html.H1("DRM data visualization dashboard", id="header"),

    # Upload
    html.Div([
        # Parameters board
        dcc.Upload(id='upload-image', 
                    children=html.Div(['Drag and Drop a DRM dataset', html.Br(), 'Currently loaded: None']), 
                    multiple=False),
        dcc.Dropdown(id="select-dropdown",
                        options = [{"label": lab, "value": val} for val, (lab, _) in list_of_tasks.items()],
                        multi=False, value=0),
        html.Div([
            html.Label("Downscale", htmlFor="downscale-slider", id="label-downscaler"),
            dcc.Slider(id='downscale-slider', min=1, max=8, marks={i: '{}'.format(i) for i in range(1, 9)}, value=1),
        ], id="downscale-container"),
        html.Button("Start", id='button-predict'),
    ], id='form-board'),

    # Analysis tabs board
    html.Div([
        dcc.Tabs([
            dcc.Tab(label="Analysis output", children=[
                html.Div([
                    html.Div(id='image-upload', children=[]),
                ], id="tab-1-content"),
            ], id="tab-1-container"),
            dcc.Tab(label="DRM dataset", children=[
                html.Div([
                    html.Div(id='dataset-image-div', children=[]),
                ], id="tab-2-content"),
            ], id="tab-2-container"),
        ], id="tab-master"),
    ], id="analysis-board"),

    # Sliders board
    html.Div([
        html.Div([
            html.Label("Theta", htmlFor="phi-slider", id="label-phi-slider"),
            dcc.Slider(id='phi-slider', min=0, max=10, updatemode='drag', value=0),
        ], id="phi-container"),
        html.Div([
            html.Label("Phi", htmlFor="theta-slider", id="label-theta-slider"),
            dcc.Slider(id='theta-slider', min=0, max=10, updatemode='drag', value=0),
        ], id="theta-container"),
    ], id="sliders-board"),

], id="layout")


def parse_contents(content):
    string = base64.b64decode(content)
    string = io.BytesIO(string)
    string = np.load(string)
    return string


def create_imshow_graph(fig):
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    return fig


# Callback to upload dataset via drag-and-drop
@app.callback(
    [Output('upload-image', 'children'),
    Output('button-predict', 'children'),
    Output('phi-slider', 'max'),
    Output('theta-slider', 'max')],
    [Input('upload-image', 'contents')],
)
def update_output(content):
    global string
    if content is not None:
        string = parse_contents(content.split(',')[1])
        return [
            html.Div(['Drag and Drop a DRM dataset', html.Br(), f'Currently loaded: {string.shape}']), 
            html.P("RUN"), 
            string.shape[3], string.shape[2]
        ]
        
    else:
        return [
            html.Div(['Drag and Drop a DRM dataset', html.Br(), 'Currently loaded: None']), 
            html.P("NO FILES"), 
            10, 10
        ]
        

# Callback to start "predict"
@app.callback(
    Output('image-upload', 'children'),
    [Input('button-predict', 'n_clicks'),  # Get clicks from button
    Input('select-dropdown', 'value'),  # Get index of task
    Input('downscale-slider', 'value')],  # Get downscaling factor
)
def start_predict(n_clicks, task_index, downscale_factor):
    global n_clicks_trakcer, prediction, list_of_tasks, string

    if n_clicks is not None:  # This drives me nuts
        if n_clicks != n_clicks_trakcer:  # If n_clicks has changed
            n_clicks_trakcer = n_clicks
            task = list_of_tasks[task_index][1]
            print('starting task: ', task)
            prediction = task(string, ds=downscale_factor)
            print('got prediction: ', prediction.shape)

    graph = dcc.Graph(
        id="predict-graph",
        figure=create_imshow_graph(
            px.imshow(prediction),
        ),
    )

    return graph

# Callback for updating drm dataset visualization
@app.callback(
    Output('dataset-image-div', 'children'),
    [Input('phi-slider', 'value'),
    Input('theta-slider', 'value')],
)
def change_dataset_display(phi_value, theta_value):
    global string

    image_to_display = string[:, :, theta_value, phi_value]

    graph = dcc.Graph(
        id="dataset-graph",
        figure=create_imshow_graph(
            px.imshow(image_to_display, color_continuous_scale="gray"),
        ),
    )

    return graph

if __name__ == '__main__':
    app.run_server(
        # debug=True,
        # dev_tools_hot_reload=False,  # I'd trigger this only when modifying CSS.
    )
