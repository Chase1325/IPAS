import numpy as np


class IPAS2PozyxDummy:
    def __init__(self, *args, **kwargs):
        pass

    def convert(self, ipas_coordinates):
        return ipas_coordinates


class IPAS2Pozyx:
    def __init__(self, pozyx_tag_locations, ipas_field_dimensions):
        '''
        ipas_field_dimensions = object with atributes 'width' and 'height'
        pozyx_tag_locations = list of two (x,y) tuples placed in the corners closest to y-axis
        '''
        self.y_scale = (pozyx_tag_locations[1][1] - pozyx_tag_locations[0][1])/ipas_field_dimensions.height
        self.y_offset = pozyx_tag_locations[0][1]
        self.x_offset = pozyx_tag_locations[0][0]
        self.x_scale = y_scale*1.6*ipas_field_dimensions.width/ipas_field_dimensions.height
        self.rotation = np.arctan((pozyx_tag_locations[1][0] - pozyx_tag_locations[0][0])/y_scale)

    def convert(self, ipas_coordinates):
        return (ipas_coordinates.x*self.x_scale-self.x_offset, ipas_coordinates.y*self.y_scale-self.y_offset)
        '''
        still needs to implement rotation
        '''


