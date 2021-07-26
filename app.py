import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
from numpy.lib.type_check import imag
from analysis import get_pca_contrast, get_anomaly_contrast, get_grain_segmentation
from serving_model import get_orientation
from utils import *
import plotly.express as px
from joblib import Memory

# Cache heavy calculations:
memory = Memory("./joblib_cache", bytes_limit=3000000000, verbose=3)
get_orientation_cached = memory.cache(get_orientation)
grain_segmentation_cached = memory.cache(get_grain_segmentation)

# Global variables:
n_clicks_trakcer = None
prediction = np.zeros((100, 100))
string = np.zeros((100, 100, 10, 36))

# Initialize app
app = dash.Dash(__name__)

# LAYOUT
app.layout = html.Div([
    # Header
    html.H1("DRM data visualization dashboard", id="header"),
    # Main section
    html.Div([
        # Upload
        html.Div([
            # Parameters board
            dcc.Upload(id='upload-image', 
                        children=html.Div(['Drag and Drop a DRM dataset', html.Br(), 'Currently loaded: None']), 
                        multiple=False),
            dcc.Dropdown(id="select-dropdown",
                            options=[{"label": "PCA contrast", "value": 0},
                                    {"label": "Z-score analysis", "value": 1},
                                    {"label": "Texture prediction", "value": 2},
                                    {"label": "Grain segmentation", "value": 3}],
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
                dcc.Tab(label="Predicted Texture", children=[
                    html.Div([
                        html.Div(id='image-upload', children=[]),
                    ], id="tab-1-content"),
                ], id="tab-1-container"),
                dcc.Tab(label="DRM Dataset", children=[
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
                html.Label("Phi", htmlFor="theta-slider", id="label-theta-slider"),
            ], id="label-container"),
            html.Div([
                dcc.Slider(id='phi-slider', min=0, max=10, updatemode='drag', value=0),
                dcc.Slider(id='theta-slider', min=0, max=10, updatemode='drag', value=0),
            ], id="slider-container"),
        ], id="sliders-board"),
        
    ], id="main"),

], id="layout")


def predict(task_index, ds):
    if task_index == 0:
        prediction = get_pca_contrast(string, ds=ds)
    elif task_index == 1:
        prediction = get_anomaly_contrast(string, ds=ds)
    elif task_index == 2:
        prediction = get_orientation_cached(string, ds=ds)
    else:
        prediction = grain_segmentation_cached(string, ds=ds)
    return prediction


def create_imshow_graph(array):
    fig = px.imshow(array)
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    return fig


# Callback to upload dataset via drag-and-drop
@app.callback(
    Output('upload-image', 'children'),
    Output('button-predict', 'children'),
    Output('phi-slider', 'max'),
    Output('theta-slider', 'max'),
    Input('upload-image', 'contents'),
)
def update_output(content):
    global string
    if content is not None:
        string = parse_contents(content.split(',')[1])
        return [
            html.Div(['Drag and Drop a DRM dataset', html.Br(), # Output 1
            f'Currently loaded: {string.shape}']), html.P("RUN"), # Output 2
            string.shape[3], string.shape[2],  # Output 3
        ]
    else:
        return [
            html.Div(['Drag and Drop a DRM dataset', html.Br(), 'Currently loaded: None']), 
            html.P("NO FILES"),
            10, 10,
        ]


# Callback to start "predict"
@app.callback(
    Output('image-upload', 'children'),
    Input('button-predict', 'n_clicks'),  # Get clicks from button
    Input('select-dropdown', 'value'),  # Get index of task
    Input('downscale-slider', 'value'),  # Get downscaling factor
)
def start_predict(n_clicks, task_index, downscale_factor):
    global n_clicks_trakcer
    global prediction
    if n_clicks is not None:  # This drives me nuts
        if n_clicks != n_clicks_trakcer:  # If n_clicks has changed
            n_clicks_trakcer = n_clicks
            prediction = predict(task_index, downscale_factor)  # array 255

    fig = px.imshow(prediction)
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0, pad=0))

    # fig = create_imshow_graph(prediction)
    graph = dcc.Graph(id="my-graph", figure=fig)

    return graph

# Callback for updating drm dataset visualization
@app.callback(
    Output('dataset-image-div', 'children'),
    Input('phi-slider', 'value'),
    Input('theta-slider', 'value'),
)
def change_dataset_display(phi_value, theta_value):
    try:
        image_to_display = string[:, :, theta_value, phi_value]
    except:
        image_to_display = np.ones((100, 100))

    fig = px.imshow(image_to_display, binary_string=True)
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0, pad=0))

    # fig = create_imshow_graph(image_to_display)
    graph = dcc.Graph(id="graph-dataset", figure=fig)

    return graph

if __name__ == '__main__':
    app.run_server(debug=True, port=8002,
                #    dev_tools_hot_reload=False,
                   )
