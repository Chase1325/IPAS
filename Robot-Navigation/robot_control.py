import RPi.GPIO as GPIO

from math import atan2, pi
import time
import json
import threading
import signal
import sys
import numpy as np
import random
import traceback
import socket

from funcs import normalize_angle
from state import State
from motor import Motor



class Robot:
    def __init__(self, right_motor, left_motor, state, pozyx_to_cor=40):
        self.rmotor = right_motor
        self.lmotor = left_motor
        self.state = state
        # Distance between pozyx and the center of rotation, mm
        self.pozyx_to_cor = pozyx_to_cor

    def brake(self):
        self.rmotor.brake()
        self.lmotor.brake()

    def forward(self):
        self.rmotor.forward()
        self.lmotor.forward()

    def reverse(self):
        self.rmotor.reverse()
        self.lmotor.reverse()

    def drive(self, speed, correction_factor=1):
        self.rmotor.drive(speed)
        self.lmotor.drive(speed*correction_factor)

    def turn(self, speed=0.5, ccw=False, correction_factor=1):
        if not ccw:
            self.rmotor.forward()
            self.lmotor.reverse()
        else:
            self.rmotor.reverse()
            self.lmotor.forward()

    def waypoint_as_np_array(func):
        # Simple decorator for methods which have waypoint
        #  as their first argument. This makes sure waypoint
        #  is a numpy array before proceeding
        def func_wrapper(self, waypoint, *args, **kwargs):
            if not isinstance(waypoint, np.ndarray):
                waypoint = np.array(waypoint)
            return func(self, waypoint, *args, **kwargs)
        return func_wrapper

    @waypoint_as_np_array
    def waypoint_heading(self, waypoint):
        location = self.state.location
        heading = self.state.heading
        # Direction to center of rotation. Assumes pozyx
        #  is mounted forward of the center of the wheels
        cor_heading = normalize_angle(heading + pi)
        # Current location of the center of rotation
        cor_location = location + self.pozyx_to_cor * np.array([np.cos(cor_heading), np.sin(cor_heading)])
        offset = cor_location - waypoint
        target_heading = atan2(offset[1], offset[0])
        return normalize_angle(target_heading - pi)

    def rotate(self, omega, t):
        t_start = time.time()
        while (time.time()-t_start)<t:
            self.lmotor.drive(-omega)
            self.rmotor.drive(omega)
            # TODO: Sleep to make this loop less busy, or callback
        self.lmotor.brake()
        self.rmotor.brake()

    def rotate2heading(self, target_heading, MOE=pi/180):
        heading_error = normalize_angle(target_heading - self.state.heading)
        kp = 11
        kd = 0.6
        ki = 0.12
        error_sum = 0
        last_error = 0
        while abs(heading_error) > MOE:  # Thanks @gus_K for making sure the logic is sound :P
            p = kp*heading_error
            i = ki*error_sum
            d = kd*(heading_error - last_error)
            PID_output = p + i + d
            self.rmotor.drive(PID_output)
            self.lmotor.drive(-PID_output)
            error_sum += heading_error
            last_error = heading_error
            time.sleep(0.01)
            heading_error = normalize_angle(target_heading - self.state.heading)
            print(heading_error, PID_output)
        self.brake()
        return heading_error

    @waypoint_as_np_array
    def nav2waypoint(self, waypoint, MOE_heading=pi/180, MOE_location=100, speed=50, correction=1):
        #align robot with next waypoint
        #print(self.state)
        self.rotate2heading(self.waypoint_heading(waypoint), MOE_heading)
        location = self.state.location
        dist = np.linalg.norm(waypoint - location)
        heading_error = normalize_angle(self.waypoint_heading(waypoint) - self.state.heading)
        #print(self.state)
       #print("Heading Error: ",heading_error,", Position Error: ",dist)
        while abs(heading_error) < MOE_heading*30 and dist > MOE_location:
            self.rmotor.drive(speed)
            self.lmotor.drive(speed*correction)
            time.sleep(0.01)
            location = self.state.location
            dist = np.linalg.norm(waypoint - location)
            heading_error = normalize_angle(self.waypoint_heading(waypoint) - self.state.heading)
        if dist > MOE_location:
            return self.nav2waypoint(waypoint, MOE_heading=MOE_heading, MOE_location=MOE_location, speed=speed, correction=correction)
        else:
            self.brake()
            return location

    @waypoint_as_np_array
    def nav2waypoint2(self, waypoint, MOE_heading=pi/180, MOE_location=100, speed=25, correction=1):
        self.rotate2heading(self.waypoint_heading(waypoint), MOE_heading)
        location = self.state.location
        dist = np.linalg.norm(waypoint - location)
        heading_error = normalize_angle(self.waypoint_heading(waypoint) - self.state.heading)
        kp = 25
        kd = 1.5
        ki = .05
        kp_dist = 0.0015
        error_sum = 0
        last_error = 0
        while dist > MOE_location:
            PID_output = kp*heading_error+kd*(heading_error-last_error)+ki*error_sum
            ldrive = (kp_dist*dist*speed-PID_output)*correction
            rdrive = kp_dist*dist*speed+PID_output
            driveRatio = ldrive/rdrive
            if(rdrive > 100):
                rdrive = 100
                ldrive = rdrive*driveRatio
            elif(ldrive > 100):
                ldrive = 100
                rdrive = ldrive/driveRatio
            self.rmotor.drive(rdrive)
            self.lmotor.drive(ldrive)
            error_sum += heading_error
            last_error = heading_error
            time.sleep(0.1)
            heading_error = normalize_angle(self.waypoint_heading(waypoint) - self.state.heading)
            location = self.state.location
            dist = np.linalg.norm(waypoint - location)
            print("Heading Error: ",heading_error," Distance: ",dist)
        return location

    def calibrate_correction_factor(self, start_cf=1):
        cf = start_cf
        self.forward()
        dheading = 999
        while dheading > 0.0001:
            heading = self.state.heading
            self.drive(25,cf)
            time.sleep(2)
            new_heading = self.state.heading
            dheading = abs((new_heading - heading)/heading)
            print(heading, new_heading, dheading)
            dcf = min(0.1 * dheading, 0.1)
            if heading > new_heading:
                print("Pulling to the right, decreasing left motor cf.")
                cf -= dcf
            else:
                print("Pulling to the left, increasing left motor cf.")
                cf += dcf
            print(f"cf:{cf}")
            heading = new_heading
        return cf

    def dance(self):
        moves = [
            (1, 75),
            (75, 1),
            (75, 75),
            (-75, -75),
            (-75, 75),
            (75, -75),
        ]
        move = random.choice(moves)
        old_move = move
        while True:
            while move == old_move:
                move = random.choice(moves)
            speed = random.uniform(0.5, 1)
            self.lmotor.drive(speed*move[0])
            self.rmotor.drive(speed*move[1])
            time.sleep(random.random()*5+1)
            old_move = move


if __name__ == "__main__":
    try:
        GPIO.setmode(GPIO.BOARD)
        pwma = 37
        a1 = 40
        a2 = 38
        ae = 33
        pwmb = 11
        b1 = 13
        b2 = 15
        be = 31
        ma = Motor(pwma, a1, a2, ae)
        mb = Motor(pwmb, b1, b2, be)
        state = State(2.0682151636132806, lencoder=ma.encoder, rencoder=mb.encoder, wheel_width=150.5)
        state.start()
        print("Starting Robot Control")

        def cleanup_and_exit(*args):
            state.join()
            GPIO.cleanup()
            sys.exit(0)

        signal.signal(signal.SIGINT, cleanup_and_exit)

        # Wait for thread to acquire heading
        while state.heading is None:
            time.sleep(0.1)

        r = Robot(ma, mb, state)
        waypoint = (2400, 4900)
        if len(sys.argv) == 2 and sys.argv[1] == 'server':
            print("Listening on 6579")
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(('localhost', 6579))
            while True:
                conn, addr = self.server.accept()
                try:
                    data = self.conn.recv(1)
                    while data and data[-1] != ord('\n'):
                        data += self.conn.recv(1)
                    if not data:
                        break
                except socket.error:
                    break

                try:
                    text = data.decode().strip()
                except UnicodeDecodeError:
                    break

                print(f'Received: {text} from {addr}')

                if ':' in text:
                    try:
                        command, value = text.split(':')
                    except ValueError:
                        break
                else:
                    command = text

                if command.lower() == 'goto':
                    value = value.strip('()')
                    try:
                        x, y = (int(i) for i in value.split(','))
                    except ValueError:
                        break
                    r.nav2waypoint((x,y), correction=1)
                    print(f'Navigated to ({x}, {y})')
                elif command.lower() == 'pos':
                    pos = r.state.location
                    conn.sendall(f'{pos}\n')
                conn.sendall('Done\n')
        elif len(sys.argv) == 2 and sys.argv[1] == 'shell':
            import code; code.interact(local=locals())
        elif len(sys.argv) == 2 and sys.argv[1] == 'status':
            print(r.state.heading)
            print(r.waypoint_heading(waypoint))
        elif len(sys.argv) >= 2 and sys.argv[1] == 'cf':
            r.calibrate_correction_factor(float(sys.argv[2]) if len(sys.argv) > 2 else 1)
        elif len(sys.argv) == 2 and sys.argv[1] == 'dance':
            r.dance()
        elif len(sys.argv) == 2 and sys.argv[1] == 'stay':
            input()
        elif len(sys.argv) == 1:
            #r.rotate2heading(r.waypoint_heading(waypoint))
            #waypoint_heading = r.waypoint_heading(waypoint)
            #r.rotate2heading(waypoint_heading)
            #r.rotate2heading(0,pi/180)
            print(r.nav2waypoint(waypoint, correction=1))
            time.sleep(4)
            print(r.nav2waypoint((2200, 6000), correction=1))
            time.sleep(4)
            print(r.nav2waypoint((2900, 5500), correction=1))
            #r.rotate(30,100)
            #r.nav2waypoint2(waypoint)
        elif len(sys.argv) == 2 and sys.argv[1] == 'adv':
            print(r.nav2waypoint2(waypoint, correction=1))
            time.sleep(4)
            print(r.nav2waypoint2((2200, 6000), correction = 1))
            time.sleep(4)
            print(r.nav2waypoint2((2900, 5500), correction = 1))
    except Exception as e:
        print(traceback.format_exc())
        print(e)
    cleanup_and_exit(state)
