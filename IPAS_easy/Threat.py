"""Threat

Define a Threat which is associated with an Environment and used by Search.

Associates a cost for a location in the Environment. Our common threat is a
summation of Gaussians

threat = sum( 1/sqrt(2*pi*sigma_i^2)*exp(-1/2*sigma_i^2*(x - mu_i)^2)"""
import math
import numpy as np


class Threat(object):
    """Base class for a Threat"""

    def __init__(self):
        pass


class ThreatField(object):
    """Base class for a threat field. Can be a linear combination of Threats

    A ThreatField subclass should implement a method to add threats and a
    method to calculate the threat value. This should allow for flexibility in
    different threat basis functions."""

    def __init__(self, threats=None):
        self.threats = threats
        self.n_threats = len(threats)

    def __getitem__(self, item):
        return self

    def add_threat(self, threat):
        return NotImplementedError

    def threat_value(self, location):
        return NotImplementedError

    def __str__(self):
        return "ThreatField: n_threats = {0}".format(self.n_threats)


class GaussThreat(Threat):
    """A single threat in an Environment with properties of a Gaussian distribution.

    location: mean of Gaussian, typically (mean_x, mean_y)
    shape: the variance/std-dev of Gaussian, typically (sigma_x, sigma_y)
    intensity: external coefficient to scale the Gaussian

    Create a GaussThreat using:
    threat1 = GaussThreat(location=(2, 2), shape=(0.5, 0.5), intensity=5)"""

    def __init__(self, location, shape, intensity):
        self.location = location
        self.shape = shape
        self.intensity = intensity

    def __str__(self):
        return "GaussThreat: loc = {0}, shape = {1}, intensity = {2}".format(self.location, self.shape, self.intensity)


class GaussThreatField(ThreatField):
    """A ThreatField as a linear combination of Gaussians.

    You can create several threats and then initialize the field:
    threat1 = GaussThreat(location=(2, 2), shape=(0.5, 0.5), intensity=5)
    threat2 = GaussThreat(location=(8, 8), shape=(1.0, 1.0), intensity=5)
    threat3 = GaussThreat(location=(8, 2), shape=(1.5, 1.5), intensity=5)
    threats = [threat1, threat2, threat3]
    threat_field = GaussThreatField(threats=threats, offset=2)

    OR/AND add threats to an already constructed field
    threat4 = GaussThreat(location=(2, 8), shape=(0.5, 0.5), intensity=5)
    threat_field.add_threat(threat4)

    the threat_value functions should return the cumulative intensity at a given location"""

    def __init__(self, threats=None, offset=0):
        super(GaussThreatField, self).__init__(threats)
        self.offset = offset

    def threat_value(self, x, y):
        """Given a location, returns the threat value of the field

        this_threat_value = threat_field.threat_value(x_loc, y_loc)"""
        threat_val = self.offset
        for threat in self.threats:
            threat_val = threat_val + (threat.intensity * (1 / (2 * threat.shape[0] * threat.shape[1])) *
                                       np.exp((-1 / 2) * (
                                           ((x - threat.location[0]) ** 2) / (threat.shape[0] ** 2) +
                                           ((y - threat.location[1]) ** 2) / (threat.shape[1] ** 2))))

        return threat_val

    def add_threat(self, threat):
        """Add a new threat to the field

        threat4 = GaussThreat(location=(2, 8), shape=(0.5, 0.5), intensity=5)
        threat_field.add_threat(threat4)
        """
        self.threats.append(threat)
        self.n_threats = self.n_threats + 1

