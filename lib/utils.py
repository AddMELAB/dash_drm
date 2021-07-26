import numpy as np
import base64
import io

def parse_contents(content):
    string = base64.b64decode(content)
    string = io.BytesIO(string)
    string = np.load(string)
    return string


    # graph = prediction_to_graph(prediction)

    # display_size = 500
    # try:
    #     rx, ry, _ = prediction.shape
    # except:
    #     rx, ry = prediction.shape
    # if rx <= ry:
    #     y_display = display_size
    #     x_display = int(display_size / ry * rx)
    # else:
    #     x_display = display_size
    #     y_display = int(display_size / rx * ry)
    # prediction = Image.fromarray(prediction.astype(np.uint8)).resize((y_display, x_display))
    # prediction = np.array(prediction)