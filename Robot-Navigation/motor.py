import RPi.GPIO as GPIO

from encoder import Encoder


class Motor:
    def __init__(self, pwm_pin, sig1_pin, sig2_pin, encoder_pin, **kwargs):
        self.pwm_pin = pwm_pin
        self.sig1_pin = sig1_pin
        self.sig2_pin = sig2_pin

        GPIO.setup(self.pwm_pin, GPIO.OUT)
        self.pwm = GPIO.PWM(self.pwm_pin, 10000)
        self.pwm.start(0)
        GPIO.setup(self.sig1_pin, GPIO.OUT)
        GPIO.setup(self.sig2_pin, GPIO.OUT)

        self.is_forward = True
        self.speed = 0

        e_prefix = 'encoder_'
        self.encoder = Encoder(
            encoder_pin, 
            self,
            # Any keyword arguments passed into this class with 
            # 'encoder_' prefixing them will be passed to Encoder
            # with the prefix removed. Ex. encoder_timeout=1 will be
            # sent passed to the encoder as timeout=1
            **{key[len(e_prefix):]:value for key, value in kwargs.items() if key.startswith(e_prefix)}
        )

    def brake(self):
        self.pwm.ChangeDutyCycle(0)
        self.speed = 0
        GPIO.output(self.sig1_pin, GPIO.LOW)
        GPIO.output(self.sig2_pin, GPIO.LOW)

    def forward(self):
        GPIO.output(self.sig1_pin, GPIO.HIGH)
        GPIO.output(self.sig2_pin, GPIO.LOW)
        self.is_forward = True

    def reverse(self):
        GPIO.output(self.sig1_pin, GPIO.LOW)
        GPIO.output(self.sig2_pin, GPIO.HIGH)
        self.is_forward = False

    def drive(self, speed):
        if speed > 0:
            self.forward()
        else:
            self.reverse()
        self.speed = speed
        self.pwm.ChangeDutyCycle(min(abs(speed), 100))


