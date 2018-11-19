from pypozyx import *
from pypozyx.structures.generic import Data, SingleRegister
from pypozyx.definitions.constants import *
from pypozyx.definitions.registers import *
from pypozyx.definitions.bitmasks import POZYX_INT_MASK_IMU
from pymemcache.client.base import Client
from serial import Serial
from argparse import ArgumentParser
from datetime import datetime
import ctypes
import signal
import time
import sys
import os
import RPi.GPIO as GPIO
import numpy as np
import json
import socket
import struct
from copy import deepcopy
from multiprocessing import Process, Pipe
from threading import Thread, Lock

from pozyx2gps import Pozyx2GPSConverter

#RUN THIS FILE AS THE MAIN FOR DRONE



DEBUG = False

LOG_DIR = '/home/pi/localization-logs'
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)
LOG_FILE = f"{LOG_DIR}/position-log-{ datetime.now().strftime('%d-%m-%Y_%H.%M.%S') }"


def log(*args, **kwargs):
    print(*args, **kwargs)
    with open(LOG_FILE, 'a+') as f:
        print(*args, file=f, **kwargs)


class PozyxInterface:
    def __init__(self, anchors, dimension=POZYX_3D, algorithm=POZYX_POS_ALG_UWB_ONLY):

        self.serial_port = None
        self.find_pozyx()
        if self.serial_port is None:
            log("Error: Pozyx not connected!")
            sys.exit(1)

        self.pozyx = PozyxSerial(self.serial_port)
        self.anchors = anchors
        self.dimension = dimension
        self.algorithm = algorithm

        self.setup()

    def find_pozyx(self):
        for port in get_serial_ports():
            if "pozyx" in port.description.lower():
                self.serial_port = port.device

    def setup(self):
        self.set_anchors_manual()

    def get_position(self):
        pos = Coordinates()
        if self.pozyx.doPositioning(pos, self.dimension, 1000, self.algorithm) != POZYX_SUCCESS:
            self.print_error_code("positioning")
        error = PositionError()
        if self.pozyx.getPositionError(error) != POZYX_SUCCESS:
            log("Failed to get positioning error.")
        return pos, error

    def set_anchors_manual(self):
        status = self.pozyx.clearDevices()
        for a in self.anchors:
            status &= self.pozyx.addDevice(a)
        if len(self.anchors) > 4:
            status &= self.pozyx.setSelectionOfAnchors(POZYX_ANCHOR_SEL_AUTO, len(self.anchors))
        return status

    def set_anchors_auto(self):
        pass

    def print_position(self, pos):
        """Prints the Pozyx's position"""
        log(f"POS: x(mm):{pos.x} y(mm):{pos.y} z(mm):{pos.z}")

    def print_error_code(self, operation):
        """Prints the Pozyx's error"""
        error_code = SingleRegister()

        self.pozyx.getErrorCode(error_code)
        log("ERROR %s, local error code %s" % (operation, str(error_code)))

    def print_configuration_result(self):
        """Prints the anchor configuration result in a human-readable way."""
        list_size = SingleRegister()

        status = self.pozyx.getDeviceListSize(list_size)
        log("List size: {0}".format(list_size[0]))
        if list_size[0] != len(self.anchors):
            self.printPublishErrorCode("configuration")
            return
        device_list = DeviceList(list_size=list_size[0])
        status = self.pozyx.getDeviceIds(device_list)
        log("Calibration result:")
        log("Anchors found: {0}".format(list_size[0]))
        log("Anchor IDs: ", device_list)

        for i in range(list_size[0]):
            anchor_coordinates = Coordinates()
            status = self.pozyx.getDeviceCoordinates(device_list[i], anchor_coordinates)
            log("ANCHOR,0x%0.4x, %s" % (device_list[i], str(anchor_coordinates)))

    def print_anchor_configuration(self):
        """Prints the anchor configuration"""
        for anchor in self.anchors:
            log("ANCHOR,0x%0.4x,%s" % (anchor.network_id, str(anchor.coordinates)))

    def get_range(self, id):
        range = DeviceRange()
        if self.pozyx.doRanging(id, range) != POZYX_SUCCESS:
            log(f"Failed to range to device {hex(id)}")
        return range


class SerialInterface:
    class BeaconConfigMsg(ctypes.Union):
        class BeaconConfigStruct(ctypes.Structure):
            _pack_ = 1
            _fields_ = [
                ("beacon_id", ctypes.c_uint8),
                ("beacon_count", ctypes.c_uint8),
                ("x", ctypes.c_int32),
                ("y", ctypes.c_int32),
                ("z", ctypes.c_int32),
            ]

        _fields_ = [
            ("info", BeaconConfigStruct),
            ("buf", ctypes.c_uint8 * 14),
        ]


    class BeaconDistanceMsg(ctypes.Union):
        class BeaconDistanceStruct(ctypes.Structure):
            _pack_ = 1
            _fields_ = [
                ("beacon_id", ctypes.c_uint8),
                ("distance", ctypes.c_uint32),
            ]

        _fields_ = [
            ("info", BeaconDistanceStruct),
            ("buf", ctypes.c_uint8 * 5),
        ]



    class VehiclePositionMsg(ctypes.Union):
        class VehiclePositionStruct(ctypes.Structure):
            _pack_ = 1
            _fields_ = [
                ("x", ctypes.c_int32),
                ("y", ctypes.c_int32),
                ("z", ctypes.c_int32),
                ("position_error", ctypes.c_int16),
            ]

        _fields_ = [
            ("info", VehiclePositionStruct),
            ("buf", ctypes.c_uint8 * 14),
        ]

    MSG_HEADER = ctypes.c_uint8(0x01)
    MSGID_BEACON_CONFIG = ctypes.c_uint8(0x02)
    MSGID_BEACON_DIST = ctypes.c_uint8(0x03)
    MSGID_POSITION = ctypes.c_uint8(0x04)

    def __init__(self, dev):
        self.serial = Serial(dev, 9600)

    def send_msg(self, msg_id, data_len, data):
        # Sanity check
        if data_len < 1:
            return

        # Message is data length + 1 for checksum
        msg_len = ctypes.c_uint8(data_len + 1)

        # Calc checksum and place in last element of array
        checksum = ctypes.c_uint8(0)
        checksum.value ^= msg_id.value
        checksum.value ^= msg_len.value
        for byte in data:
            checksum.value ^= byte

        # Send msg
        num_sent = 0
        num_sent += self.serial.write(self.MSG_HEADER)
        num_sent += self.serial.write(msg_id)
        num_sent += self.serial.write(msg_len)
        num_sent += self.serial.write(data)
        num_sent += self.serial.write(checksum)
        #self.serial.flush()

    def send_beacon_distance(self, i, distance):
        msg = self.BeaconDistanceMsg()
        msg.info.beacon_id = i
        msg.info.distance = distance
        self.send_msg(self.MSGID_BEACON_DIST, len(msg.buf), msg.buf)

    def send_beacon_config(self, anchors):
        msg = self.BeaconConfigMsg()
        msg.info.beacon_count = len(anchors)
        for i in range(msg.info.beacon_count):
            msg.info.beacon_id = i
            msg.info.x = anchors[i].pos.x
            msg.info.y = anchors[i].pos.y
            msg.info.z = anchors[i].pos.z
            self.send_msg(self.MSGID_BEACON_CONFIG, len(msg.buf), msg.buf)

    def send_vehicle_position(self, position, pos_error):
        # Sanity check position
        if position.x == 0 or position.y == 0:
            return

        msg = self.VehiclePositionMsg()

        msg.info.x = position.x
        msg.info.y = position.y
        msg.info.z = position.z

        msg.info.position_error = int(pos_error.xy)
        self.send_msg(self.MSGID_POSITION, len(msg.buf), msg.buf)

    def read(self, *args, **kwargs):
        return self.serial.read(*args, **kwargs)


def send_beacon_config(anchors, serial):
    serial.send_beacon_config(anchors)


def send_position(position, pos_error, serial):
    serial.send_vehicle_position(position, pos_error)


class PositionState:
    alpha = 0.2
    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0
        self.initialized = False

    def update_pos(self, x, y, z):
        if self.initialized:
            self.x = int((1-self.alpha)*self.x + self.alpha*x)
            self.y = int((1-self.alpha)*self.y + self.alpha*y)
            self.z = int((1-self.alpha)*self.z + self.alpha*z)
        else:
            self.x = int(x)
            self.y = int(y)
            self.z = int(z)
            self.initialized = True


def position_generator(pozyx, IS_UAV):
    # Use a generator expression to maintain state of position
    smooth_pos = PositionState()
    beta = 0.25
    pos_deviation = None
    alt_deviation = 266
    low_alt_count = 0
    alt = 266  # Min value from altimeter
    while True:
        # Get position from pozyx
        raw_pos, pos_error = pozyx.get_position()
        raw_pos_v = np.array([raw_pos.x, raw_pos.y, raw_pos.z])
        smooth_pos_v = np.array([smooth_pos.x, smooth_pos.y, smooth_pos.z])
        pos_dev_i = np.linalg.norm(raw_pos_v - smooth_pos_v)
        if pos_deviation is None:
            pos_deviation = pos_dev_i
        new_pos_dev = (1-beta)*pos_deviation + beta*pos_dev_i
        if raw_pos.x == 0 and raw_pos.y == 0 and raw_pos.z == 0:
            continue
        elif pos_dev_i > 5*pos_deviation:
            # Got shite value
            pos_deviation = new_pos_dev
            continue
        else:
            pos_deviation = new_pos_dev

        rngs = []
        for i in range(len(pozyx.anchors)):
            rngs.append(pozyx.get_range(pozyx.anchors[i].network_id))

        # TODO, pos_error from std dev
        yield raw_pos, pos_error, rngs

        smooth_pos.update_pos(raw_pos.x, raw_pos.y, alt)
        #yield smooth_pos, pos_error


def send_ranges(serial, rngs):
    for i, rng in enumerate(rngs):
        serial.send_beacon_distance(i, rng.distance)


def locked_property(func):
    def wrapped_prop(self, *args, **kwargs):
        self.lock.acquire()
        value = func(self, *args, **kwargs)
        self.lock.release()
        return value
    return wrapped_prop


class PositionTimingClient(Thread):
    def __init__(self, conn, ip, port):
        super(PositionTimingClient, self).__init__()
        self.lock = Lock()

        self.conn = conn
        self.ip = ip
        self.port = port
        self._dead = False
        self._can_position = False

        self.stop = False

    def run(self):
        while not self.stop:
            try:
                data = self.conn.recv(1)
                while data and data[-1] != ord('\n'):
                    data += self.conn.recv(1)
                if not data:
                    break
            except:
                break

            try:
                text = data.decode().strip()
                if DEBUG: print(f'Received {text}')
                if text.lower()[:2] == "go":
                    if DEBUG: print('Can Position!')
                    self.can_position = True
                elif text.lower() == 'ping':
                    if DEBUG: print('Received ping, responding pong.')
                    self.conn.sendall('pong\n'.encode())
                    self.can_position = False
                else:
                    if DEBUG: print('Cannot Position')
                    self.can_position = False
            except Exception:
                self.can_position = False
        self.is_dead = True

    def close(self):
        self.conn.close()

    @property
    @locked_property
    def is_dead(self):
        return self._dead

    @is_dead.setter
    @locked_property
    def is_dead(self, value):
        self._dead = value

    @property
    @locked_property
    def can_position(self):
        can = self._can_position
        # Whenever read, reset and wait for next message
        self._can_position = False
        return can

    @can_position.setter
    @locked_property
    def can_position(self, value):
        self._can_position = value


class PositionTimingServer(Thread):
    def __init__(self, ip, port):
        super(PositionTimingServer, self).__init__()
        self.lock = Lock()

        self.ip = ip
        self.port = port

        self.stop = False

        self.server = None
        self.client = None

    def run(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((self.ip, self.port))
        self.server.listen()

        while not self.stop:
            conn, addr = self.server.accept()
            self.lock.acquire()
            if self.client:
                self.client.close()
            self.client = PositionTimingClient(conn, *addr)
            print(f'New client!: {addr[0]}')
            self.client.start()
            self.lock.release()

        self.join()

    @property
    @locked_property
    def has_client(self):
        if self.client:
            dead = self.client.is_dead
        else:
            dead = True
        return not dead

    @property
    @locked_property
    def can_position(self):
        client_can_pos = not self.client or self.client.can_position
        if DEBUG: print(f'!self.client:{not self.client} | client_can_pos:{client_can_pos} | client_dead:{self.client.is_dead if self.client else False}')
        return not self.client or client_can_pos or self.client.is_dead

    @property
    @locked_property
    def client_ip(self):
        if self.client:
            return self.client.ip
        else:
            return None

    @property
    @locked_property
    def client_ip_num(self):
        if self.client:
            return int(self.client.ip.split('.')[-1])
        else:
            return None


class PozyxPositionProcess(Process):
    ip_base = '192.168.0.{}'
    ip_start = 10
    n_hosts = 5
    server_port = 7470

    def __init__(self, comms, anchors, IS_UAV, postimeout=0.1):
        super(PozyxPositionProcess, self).__init__()

        self.postimeout = postimeout

        self.comms = comms
        self.my_ip = socket.gethostbyname(socket.gethostname())
        self.my_ip_num = int(self.my_ip.split('.')[-1])
        self.server = None
        self.client = None

        self.pos = None
        self.pos_error = None
        self.rngs = None

        self.IS_UAV = IS_UAV

        self.anchors = anchors
        self.pozyx = PozyxInterface(self.anchors)

        self.pozyx.print_configuration_result()

        self.pozyx_pos_generator = position_generator(self.pozyx, self.IS_UAV)

        self.sensor_data = SensorData()
        self.tmp_sensor_data = SensorData()
        self.sensor_calibration_status = SingleRegister()

        self.stop = False

    def ping_server(self):
        # Ping the server. If we get pong, it is still alive. No response means
        #  we don't have a server
        try:
            self.server.sendall('ping\n'.encode())
        except socket.error:
            # Failed to send ping. Server is dead.
            if DEBUG: print('Failed to ping server.')
            return False
        else:
            try:
                data = self.server.recv(1)
                while data and data[-1] != ord('\n'):
                    data += self.server.recv(1)
            except:
                return False

            if not data:
                # Didn't recieve a response, server is dead
                return False
            else:
                text = data.decode().strip()
                if text.lower() == 'pong':
                    # Server is alive
                    return True

    def run(self):
        self.client = PositionTimingServer(self.my_ip, self.server_port)
        self.client.start()

        # Give a 5 second buffer for everyone to come online if we are
        #  starting multiple devices
        time.sleep(5)

        self.find_next_host()
        # If we are looking for hosts but don't want to block, thread it
        #  in this variable
        t = None
        loops_wo_positioning = 0

        while not self.stop:
            if self.client.has_client and not self.server:
                # If we have a client but no server, we need to stop and get a server
                if t is not None: t.join()
                self.find_next_host()
            elif not self.client.has_client and self.server:
                # If we don't have a client but we have a server, wait for client
                if DEBUG: print('Waiting for client')
                while not self.client.has_client and self.server:
                    if self.ping_server():
                        # Server is alive, wait for new client
                        time.sleep(0.25)
                    else:
                        # Server is dead, don't waste time here
                        self.server = None
                        break
                # Force a reset
                loops_wo_positioning = 999

            can_position = self.client.can_position

            if loops_wo_positioning > 50 and not can_position:
                # Break a stalemate condition
                if self.client.has_client and self.server:
                    # If my client has a higher IP than me, it means I am the lowest IP
                    # Therefore I can start the positioning
                    if self.client.client_ip_num > self.my_ip_num:
                        can_position = True

            if DEBUG: print(f'Can position: {can_position} Client: {self.client.client_ip} Server: {self.server}')
            if can_position or not self.server:
                # Get the position
                self.pos, self.pos_error, self.rngs = next(self.pozyx_pos_generator)

                if self.server:
                    try:
                        if DEBUG: print('Sending go')
                        self.server.sendall('Go!?\n'.encode())
                    except socket.error as e:
                        # Failed to send to next client
                        if DEBUG: print('Failed to send go')
                        loops_wo_server = 999

                loops_wo_positioning = 0
            else:
                # On off loops, ping our server
                if not self.ping_server():
                    # Server is dead, reset it
                    self.server = None

                loops_wo_positioning += 1

            # Get new sensor data if availible and good
            if self.pozyx.pozyx.checkForFlag(POZYX_INT_MASK_IMU, 0.01) == POZYX_SUCCESS:
                status = self.pozyx.pozyx.getAllSensorData(self.tmp_sensor_data)
                status &= self.pozyx.pozyx.getCalibrationStatus(self.sensor_calibration_status)
                if status == POZYX_SUCCESS:
                    self.sensor_data = self.tmp_sensor_data


            # Check to see if main process wants the position
            if self.comms.poll():
                data = self.comms.recv()
                if data == 'stop':
                    self.stop = True
                elif data:
                    self.comms.send((self.pos, self.pos_error, self.rngs, deepcopy(self.sensor_data)))
            time.sleep(self.postimeout)

        if self.server:
            self.server.shutdown(socket.SHUT_RDWR)
        if self.client:
            self.client.stop = True
            self.client.join()

    def find_next_host(self):
        max_host = self.ip_start + self.n_hosts
        current_host = self.my_ip_num

        while True:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.my_ip, self.server_port + 1))
            server.settimeout(0.5)

            current_host += 1

            if current_host > max_host:
                current_host = self.ip_start

            if current_host == self.my_ip_num:
                self.server = None
                break

            try:
                ip = self.ip_base.format(current_host)
                server.connect((ip, self.server_port))
            except socket.error as e:
                continue
            else:
                self.server = server
                if ip == self.client.client_ip: print('PHASE LOCKED')
                break


class RangeFinderProcess(Process):
    def __init__(self, comms, mavproxy_addr='127.0.0.1:14550', serial_dev='/dev/serial0'):
        super(RangeFinderProcess, self).__init__()

        self.comms = comms

        self.mavproxy_addr = mavproxy_addr
        self.serial_dev = serial_dev

        self.end = False

        self.last_update = ""
        self.alt = 0
        self.update_hz = 0
        self.last_time = time.time()

    @staticmethod
    def get_range_finder_reading(connection):
        b = connection.read(1)
        while b != b'R':
            b = connection.read(1)
        b = connection.read(1)
        num = b''
        while b != b'\r':
            num += b
            b = connection.read(1)
        if connection.in_waiting > 10:
            connection.flush()
        return int(num.decode()) // 10

    def run(self):
        GPIO.setmode(GPIO.BCM)
        out_pin = [12, ]
        GPIO.setup(out_pin, GPIO.OUT, initial=GPIO.LOW)
        p = GPIO.PWM(out_pin[0], 9)
        p.start(30)

        while True:
            try:
                vehicle = dk.connect(self.mavproxy_addr, wait_ready=True, heartbeat_timeout=60)
            except dk.APIException as e:
                print(f'Range finder could not connect to vehicle: {e}')
                print(f'Sleeping for 10 seconds and trying again.')
                time.sleep(10)
            else:
                break

        connection = Serial(self.serial_dev)
        connection.flush()

        while not self.end:
            # Send range beacon data
            raw_range = self.get_range_finder_reading(connection)
            now = time.time()
            self.update_hz = 1 / (now-self.last_time)
            self.last_time = now
            if 30 < raw_range < 495:
                msg = vehicle.message_factory.distance_sensor_encode(
                    0,	        # time since system boot, not used
                    30,	        # Min distance cm
                    500,        # Max distance cm
                    raw_range,	# Current distance, must be int
                    1,		# 0=Laser 1=Ultrasound 2=Infared 3=Radar 4=Unknown
                    0,		# Onboard ID, not used
                    # Must be set to a MAV_SENSOR_ROTATION_PITCH_270 for mavlink
                    #  rangefinder, represents downward facing orientation
                    mavutil.mavlink.MAV_SENSOR_ROTATION_PITCH_270,
                    0		#Covariance, Not used
                )
                vehicle.send_mavlink(msg)
                vehicle.flush()
                self.last_update = f"SRange {raw_range} DRange {vehicle.rangefinder.distance} " \
                                   f"DAlt {vehicle.location.global_relative_frame.alt}"
                self.alt = raw_range
            else:
                self.last_update = f"Range out of bounds. Ignored ({raw_range}cm). " \
                                   f"DAlt {vehicle.location.global_relative_frame.alt}"
            self.last_update += f" Update Rate: {self.update_hz:.4f}Hz"

            if self.comms.poll() and self.comms.recv():
                self.comms.send((self.last_update, self.alt))
        p.stop()
        vehicle.close()
        connection.close()
        self.comms.close()

    def cleanup(self):
        self.end = True
        self.join()


if __name__ == "__main__":
    parser = ArgumentParser(description='Process Pozyx data.')
    parser.add_argument('-x', help='X length of Pozyx field', type=int)
    parser.add_argument('-y', help='Y length of Pozyx field', type=int)
    parser.add_argument('--vehicle', help='Specify vehicle type, Ground or UAV')
    parser.add_argument('--debug', help='Print debug statements', action='store_true')
    parser.add_argument('--postimeout', help='Time between positioning attempts', type=float, default=0.05)
    parser.add_argument('-i', '--id', help='Specify ID for output file.')
    args = parser.parse_args()

    if args.id:
        LOG_FILE += '-' + args.id
    LOG_FILE += '.dat'

    DEBUG = args.debug

    # Default to True if ground isn't specified
    IS_UAV = args.vehicle.lower() != 'ground' if args.vehicle else True

    # https://raspberrypi.stackexchange.com/questions/12966/what-is-the-difference-between-board-and-bcm-for-gpio-pin-numbering
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)

    x = args.x if args.x else 14600
    y = args.y if args.y else 11100

    ANCHORS = [
        DeviceCoordinates(0x6110, 1, Coordinates(0, 0, 1455)),
        DeviceCoordinates(0x6115, 1, Coordinates(0, y, 2587)),
        DeviceCoordinates(0x6117, 1, Coordinates(x, 0, 2024)),
        DeviceCoordinates(0x611E, 1, Coordinates(x, y, 1057)),
    ]

    pixracer = SerialInterface("/dev/serial0")
    if IS_UAV:
        import dronekit as dk
        from pymavlink import mavutil
        rf_comm1, rf_comm2 = Pipe()
        range_finder_process = RangeFinderProcess(rf_comm2)
        range_finder_process.start()

    gps_converter = Pozyx2GPSConverter()

    memcached_client = Client(('localhost', 11211))

    pozyx_waki, pozyx_taki = Pipe()  # :D
    pozyx_pos_process = PozyxPositionProcess(pozyx_taki, deepcopy(ANCHORS), IS_UAV, args.postimeout)
    pozyx_pos_process.start()

    def cleanup(signal, frame):
        if IS_UAV:
            range_finder_process.cleanup()
            rf_comm1.close()
        pozyx_waki.send('stop')
        sys.exit(0)
    signal.signal(signal.SIGINT, cleanup)

    state = {}

    while True:
        # Tell the process maintaining the Pozyx position that we want it
        pozyx_waki.send(True)
        # Get the position
        pos, pos_error, rngs, sensor_data = pozyx_waki.recv()

        if not pos:
            if DEBUG: print('Position not ready yet. Waiting.')
            time.sleep(0.1)
            continue

        status = ""
        if state and (pos.x, pos.y, pos.z) != (state['x'], state['y'], state['z']):
            status += f"POS: x(mm):{pos.x} y(mm):{pos.y} z(mm):{pos.z}"

        state['x'] = pos.x
        state['y'] = pos.y
        state['z'] = pos.z

        if IS_UAV:
            # Tell the process maintaining the rangefinder distance that we want it
            rf_comm1.send(True)
            # Get the range finder distance
            rf_last_update, rf_alt = rf_comm1.recv()
            status += rf_last_update
            state['alt'] = rf_alt

            pos.z = rf_alt

            send_beacon_config(ANCHORS, pixracer)
            send_ranges(pixracer, rngs)
            send_position(pos, pos_error, pixracer)
            gps_converter.send((pos.x, pos.y, pos.z))

        if status:
            log(status)

        state['yaw'] = sensor_data.euler_angles.heading
        state['pitch'] = sensor_data.euler_angles.pitch
        state['roll'] = sensor_data.euler_angles.roll
        state['d2x'] = sensor_data.linear_acceleration.x
        state['d2y'] = sensor_data.linear_acceleration.y
        state['d2z'] = sensor_data.linear_acceleration.z
        state['qw'] = sensor_data.quaternion.w
        state['qx'] = sensor_data.quaternion.x
        state['qy'] = sensor_data.quaternion.y
        state['qz'] = sensor_data.quaternion.z

        # Write data out to the memcache so that other programs can read it
        memcached_client.set('state', json.dumps(state))
