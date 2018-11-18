import io
import time
import falcon
import os
from pymemcache.client import Client
import json

from .image_processing import get_average_color_of_frame, rgb_to_threat

try:
    import picamera
except ImportError:
    print('Please install the Pi Camera module.')


def get_frame(format='png'):
    frame = io.BytesIO()

    while True:
        # Loop until this thread has access to the camera
        try:
            with picamera.PiCamera() as camera:
                camera.resolution = (1920,1080)
                camera.brightness = 50
                camera.saturation = 0
                camera.capture(frame, format=format)
                break
        except picamera.exc.PiCameraMMALError:
            # Raised if another thread/program is accessing the camera
            time.sleep(0.5)

    frame.seek(0)
    return frame


class ImageSensingValue:
    def on_get(self, req, resp):
        """Read frame from Pi Camera and determine its threat value and return it."""
        frame = get_frame()
        avg_color = get_average_color_of_frame(frame)
        threat_level = rgb_to_threat(avg_color)

        resp.body = json.dumps({'value': threat_level, })
        resp.status = falcon.HTTP_200


class ImageSensingFrame:
    def on_get(self, req, resp):
        """Read frame from Pi Camera and return it."""
        frame = get_frame()
        resp.data = bytes(frame.read())
        resp.content_type = falcon.MEDIA_PNG
        resp.status = falcon.HTTP_200
        

class ThreatAtPosition:
    def on_get(self, req, resp):
        # Go to position
        frame = get_frame()
        avg_color = get_average_color_of_frame(file)
        threat_level = rgb_to_threat(avg_color)

        memcached_client = Client(('localhost', 11211))
        state = json.loads(memcached_client.get('state'))
        pos = (state['x'], state['y'])

        resp.body = json.dumps({'position': pos, 'threat_value': threat_level, })
        resp.status = falcon.HTTP_200

