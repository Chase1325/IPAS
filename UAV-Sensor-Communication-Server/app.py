import falcon

from commands import image_sensing, localization

import falcon_jsonify


api = falcon.API()

api.add_route('/image_sensing_value/', image_sensing.ImageSensingValue())
api.add_route('/image_sensing_frame/', image_sensing.ImageSensingFrame())
api.add_route('/threat_at_position/', image_sensing.ThreatAtPosition())
api.add_route('/localization/', localization.Localization())
