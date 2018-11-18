import numpy as np
import threading
import time
import json
from pymemcache.client.base import Client
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag
from pyquaternion import Quaternion

from funcs import normalize_angle


class PositionState:
    alpha = 0.3
    beta = 0.25

    def __init__(self):
        self.pos = np.array([0, 0])
        self.dev = 0
        self.initialized = False

    def update_pos(self, pos):
        new_pos = np.array(pos)
        if self.initialized:
            new_dev = np.linalg.norm(new_pos - self.pos)
            old_dev = self.dev
            if new_dev < 3*old_dev:
                # Got a good value 
                # If this condition fails, drop the position but still update the deviation
                self.pos = (1-self.alpha)*self.pos + self.alpha*new_pos
            self.dev = (1-self.beta)*old_dev + self.beta*new_dev
        else:
            self.pos = new_pos
            self.dev = np.linalg.norm(new_pos)
            self.initialized = True

    @property
    def x(self):
        return self.pos[0]

    @property
    def y(self):
        return self.pos[1]

 
class State(threading.Thread):
    def __init__(self, pozyx_offset, lencoder, rencoder, wheel_width=150.5):
        super(State, self).__init__()
        self.lencoder = lencoder
        self.rencoder = rencoder
        self.wheel_width = wheel_width
        self.pozyx_offset = pozyx_offset

        self._state = np.zeros(8)
        # https://github.com/rlabbe/filterpy
        self.KF = KalmanFilter(dim_x=8, dim_z=8)
        self.KF.x = np.array([0,]*8).T
        # Will be defined each iteration
        # self.KF.F = 
        # self.KF.Q = 
        self.KF.H = np.eye(8)  # No measurements need to be transformed
        self.KF.P = np.eye(8) * 500 
        R_vel = 10*(self.rencoder.step+self.lencoder.step)
        R_accel = 100 # 2*10
        self.KF.R = np.square(np.diag([150/2, R_vel, R_accel, 30/2, R_vel, R_accel, np.pi/180, R_vel/(np.pi*self.wheel_width)]))
        
        self.lock = threading.Lock()
        self.daemon = True
        self.stop = False
        self.memcached_client = Client(('localhost', 11211))

        self._heading = None
        self._location = None

    @staticmethod
    def state_transition_matrix(dt):
        F = np.eye(8)
        F[0,1] = dt
        F[0,2] = 0.5*dt**2
        F[1,2] = dt
        F[3,4] = dt
        F[3,5] = 0.5*dt**2
        F[4,5] = dt
        F[6,7] = dt
        return F

    def locked_property(func):
        # Simple decorator to handle locking and releasing
        #  before sharing variables
        def wrapped_prop(self, *args, **kwargs):
            self.lock.acquire()
            value = func(self, *args, **kwargs)
            self.lock.release()
            return value
        return wrapped_prop

    def run(self):
        last_state_read = time.time() 
        c = 0
        while not self.stop:
            state = self.memcached_client.get('state')
            if state:
                state = json.loads(state)
                heading = state['yaw']*np.pi/180
                # Pozyx increases CW, so make it CCW
                heading = 2*np.pi - heading
                # Subject value Pozyx reads for heading when it should read 0
                heading -= self.pozyx_offset
                heading = normalize_angle(heading)

                if self.pozyx_offset == 0:
                    print(heading)


                x, y = state['x'], state['y']
                now = time.time()
                dt = now - last_state_read
                dx, dy, dheading = self._compute_change_from_encoders(dt, self.heading if self.heading else heading)
                # Body fixed linear acceleration
                lin_accel = np.array([state[f'd2{i}'] for i in 'xyz'])
                # Pozyx fixed accelerations
                # Start by getting quaternion
                quart = Quaternion(*(state[f'q{i}'] for i in 'wxyz'))
                # Then rotate by its inverse
                pozyx_accel = quart.inverse.rotate(lin_accel)
                # Y from Pozyx is vertical, so grab the Z value instead
                d2x, d2y = pozyx_accel[0], pozyx_accel[2]

                z = np.array([x, dx, d2x, y, dy, d2y, heading, dheading])
                self.KF.F = self.state_transition_matrix(dt)
                q = Q_discrete_white_noise(dim=3, dt=dt, var=0.001)
                self.KF.Q = block_diag(q, q, Q_discrete_white_noise(dim=2, dt=dt, var=0.001))
                self.KF.predict()
                self.KF.update(z)

                self.lock.acquire()
                self._state = self.KF.x
                self.lock.release()

                self.heading = self._state[6]
                self.location = (self._state[0], self._state[3])
                last_state_read = now
                #if c % 5 == 0: 
                    #print(f'{dt:.6f}: {x} {dx:6f} {d2x:6f} {y} {dy:6f} {d2y:6f} {heading:.6f} {dheading}')
                    #print(self)
                c += 1
            time.sleep(0.1)

    def _compute_change_from_encoders(self, dt, heading):
        # http://www8.cs.umu.se/kurser/5DV122/HT13/material/Hellstrom-ForwardKinematics.pdf
        #  Page 7
        # Set initial conditions to 0, x=0, y=0
        #lticks = self.lencoder.get_ticks()
        #lstep = self.lencoder.step
        lvel = self.lencoder.velocity
        #rticks = self.rencoder.get_ticks()
        #rstep = self.rencoder.step
        rvel = self.rencoder.velocity
        #wdt = (rticks*rstep-lticks*lstep) / self.wheel_width
        wdt = (rvel-lvel) * dt / self.wheel_width
        try:
            #R = self.wheel_width / 2 * (rticks + lticks) / (rticks - lticks)
            R = self.wheel_width / 2 * (rvel + lvel) / (rvel - lvel)
        except ZeroDivisionError:
            R = 0
        ICC = np.array([-R*np.sin(heading), R*np.cos(heading), 0]).T
        A = np.array([
            [np.cos(wdt), -np.sin(wdt), 0], 
            [np.sin(wdt), np.cos(wdt), 0], 
            [0, 0, 1], 
        ])
        res = A @ ICC + (ICC + np.array([0, 0, wdt]).T)
        res /= dt
        return res[0], res[1], res[2]

    def join(self, *args, **kwargs):
        self.stop = True
        super(State, self).join(*args, **kwargs)

    @property
    @locked_property
    def heading(self):
        return self._heading

    @heading.setter
    @locked_property
    def heading(self, value):
        self._heading = value

    @property
    @locked_property
    def location(self):
        return self._location

    @location.setter
    @locked_property
    def location(self, value):
        self._location = value

    def __str__(self):
        return f'State: Heading:{self.heading} Location:{self.location}'

