import socket, errno
import json
from pymavlink import mavutil
from numpy import sin, cos, pi
from time import time
from datetime import datetime

# Define sin and cos so that they can take degrees
def sind(angle):
    return sin(angle * pi/180)

def cosd(angle):
    return cos(angle * pi/180)


class Pozyx2GPSConverter:
    IGNORE_FLAGS = (mavutil.mavlink.GPS_INPUT_IGNORE_FLAG_HDOP |
                    mavutil.mavlink.GPS_INPUT_IGNORE_FLAG_VDOP |
                    mavutil.mavlink.GPS_INPUT_IGNORE_FLAG_VEL_HORIZ |
                    mavutil.mavlink.GPS_INPUT_IGNORE_FLAG_VEL_VERT |
                    mavutil.mavlink.GPS_INPUT_IGNORE_FLAG_SPEED_ACCURACY |
                    mavutil.mavlink.GPS_INPUT_IGNORE_FLAG_HORIZONTAL_ACCURACY |
                    mavutil.mavlink.GPS_INPUT_IGNORE_FLAG_VERTICAL_ACCURACY)

    def __init__(self, yaw_alignment=0, lat_0=42.273283958, 
                 long_0=-71.8083867, elevation=174):
        """
        Input the coordinates of the origin of the Pozyx system as
        decimal latitude, longitude, and elevation above sea level.

        Defaults are for that of Alden Hall.

        :param yaw_alignment: Angle Pozyx x-axis is rotated from 
                                Magnetic North in degrees
        :param lat_0: Latitude of Pozyx origin
        :param long_0: Longitude of Pozyx origin
        :param elevation: Elevation above sea-level in meters
        """

        self.yaw = yaw_alignment
        self.lat_0 = lat_0
        self.long_0 = long_0
        self.elevation = elevation #in m above sea level

        a = 6378137  # Earth polar radius in m
        e2 = 0.00669437999014

        # Calculate length of 1 deg longitude and 1 deg 
        #  latitude in m at origin location
        self.deg_lat = pi*a*(1-e2)/(180*(1-e2*sind(self.lat_0)**2)**1.5)
        self.deg_long = pi*a*cosd(self.lat_0)/(180*(1-e2*sind(self.lat_0)**2)**.5)
        
        # JSON Packet that will be sent to MavProxy
        self.data = {
            'time_usec' : 0,                        # (uint64_t) Timestamp (micros since boot or Unix epoch)
            'gps_id' : 0,                           # (uint8_t) ID of the GPS for multiple GPS inputs
            'time_week_ms' : 0,                     # (uint32_t) GPS time (milliseconds from start of GPS week)
            'time_week' : 0,                        # (uint16_t) GPS week number
            'fix_type' : 3,                         # (uint8_t) 0-1: no fix, 2: 2D fix, 3: 3D fix. 4: 3D with DGPS. 5: 3D with RTK
            'lat' : 0,                              # (int32_t) Latitude (WGS84), in degrees * 1E7
            'lon' : 0,                              # (int32_t) Longitude (WGS84), in degrees * 1E7
            'alt' : 0,                              # (float) Altitude (AMSL, not WGS84), in m (positive for up)
            # Fake number of satellites
            'satellites_visible' : 6,               # (uint8_t) Number of satellites visible.
            'ignore_flags' : self.IGNORE_FLAGS,     # (uint16_t) Flags indicating which fields to ignore (see GPS_INPUT_IGNORE_FLAGS enum). All other fields must be provided.
            # Ignored Parameters, may specify in later versions
            'hdop' : 0,                             # (float) GPS HDOP horizontal dilution of position in m
            'vdop' : 0,                             # (float) GPS VDOP vertical dilution of position in m
            'vn' : 0,                               # (float) GPS velocity in m/s in NORTH direction in earth-fixed NED frame
            've' : 0,                               # (float) GPS velocity in m/s in EAST direction in earth-fixed NED frame
            'vd' : 0,                               # (float) GPS velocity in m/s in DOWN direction in earth-fixed NED frame
            'speed_accuracy' : 0,                   # (float) GPS speed accuracy in m/s
            'horiz_accuracy' : 0,                   # (float) GPS horizontal accuracy in m
            'vert_accuracy' : 0,                    # (float) GPS vertical accuracy in m
        }
        
        # Create UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # MavProxy will be listening here:
        self.recving_socket_addr = ('127.0.0.1', 25100)
        # Store GPS epoch for usage later
        self.gps_epoch = datetime.strptime("1980-01-06 00:00:00", "%Y-%m-%d %H:%M:%S")

    def pozyx2gps(self, pozyx_data):
        """
        Convert Pozyx coordinatest to GPS coordinates
        https://en.wikipedia.org/wiki/Latitude

        :param pozyx_data: Iterable with current Pozyx location 
                            in mm in form of (x,y,z)
        """

        # Pozyx coordinates in m
        x = pozyx_data[1] / 1000
        y = pozyx_data[0] / 1000
        z = pozyx_data[2] / 1000

        lat = self.lat_0 + x/self.deg_lat*cosd(self.yaw) + y/self.deg_long*sind(self.yaw)
        lon = self.long_0 - x/self.deg_lat*sind(self.yaw) + y/self.deg_long*cosd(self.yaw)
        h = self.elevation + z

        return (lat, lon, h)

    def send(self, pozyx_data):
        """
        Generate GPS packet and send over UDP to MavProxy

        :param pozyx_data: Iterable with current Pozyx location
                            in mm in form of (x,y,z)
        """

        lat, lon, h = self.pozyx2gps(pozyx_data)
        self.data['lat'] = lat * 10**7
        self.data['lon'] = lon * 10**7
        self.data['alt'] = h

        # Current time
        self.data['time_usec'] = time() * 10**6

        # GPS week and week milliseconds
        elapsed_since_gps_epoch = float((datetime.now() - self.gps_epoch).total_seconds())
        seconds_per_week = 3600 * 24 * 7
        self.data['time_week'] = elapsed_since_gps_epoch // seconds_per_week
        self.data['time_week_ms'] = (elapsed_since_gps_epoch % seconds_per_week) * 1000

        # Send UDP Packet
        self.sock.sendto(json.dumps(self.data).encode(), self.recving_socket_addr)

