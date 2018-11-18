import RPi.GPIO as GPIO
import time
import numpy as np


class Encoder:
    def __init__(self, pin, motor, wheel_radius=65/2, resolution=10, measure_bias=0.3):
        self.wheel_radius = wheel_radius
        self.pin = pin
        self.resolution = resolution
        self.measure_bias = measure_bias
        self.step = 2*np.pi*self.wheel_radius / resolution
        self.motor = motor

        # Typical max drive speed for motor, mm/s
        self.max_drive_speed = 200 

        self.tick_time = time.time()
        # Number of ticks since last
        self.ticks_since_last = 0
        self.velocity = 0

        GPIO.setup(self.pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
        GPIO.add_event_detect(self.pin, GPIO.RISING, callback=self.callback)

    def callback(self, channel):
        now = time.time()
        dt = now - self.tick_time
        try:
            timeout = self.step / (self.motor.speed/100 * self.max_drive_speed) / 3
        except ZeroDivisionError:
            timeout = self.step / (self.max_drive_speed) / 3
        if (dt > timeout):
            self.tick_time = now
            sign = 1
            if self.motor.is_forward:
                self.ticks_since_last += 1
            else:
                self.ticks_since_last -= 1
                sign = -1
        
            self.velocity = (1 - self.measure_bias)*self.velocity + sign*self.measure_bias*max(17, self.step / dt)
            #print(self.velocity, (1 - self.measure_bias)*self.velocity, sign*self.measure_bias*(self.step/dt), dt)

    def get_ticks(self):
        # Returns the number of ticks since the last time this was called
        ticks = self.ticks_since_last
        self.ticks_since_last = 0
        return ticks


    def reset(self):
        self.tick_time = time.time()
        self.velocity = 0
