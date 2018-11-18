"""Test out Threat and threat field classes"""

from Threat import GaussThreat, GaussThreatField
from Environment import XYEnvironment
import math
import matplotlib.pyplot as plt
from Visualize import draw_threat_field, draw_threat_field_2D


def main():
    env = XYEnvironment(x_size=10, y_size=10, x_pts=20, y_pts=20)

    threat1 = GaussThreat(location=(2, 2), shape=(0.5, 0.5), intensity=5)
    threat2 = GaussThreat(location=(8, 8), shape=(1.0, 1.0), intensity=5)
    threat3 = GaussThreat(location=(8, 2), shape=(1.5, 1.5), intensity=5)
    threats = [threat1, threat2, threat3]
    threat_field = GaussThreatField(threats=threats, offset=2)

    env.add_threat_field(threat_field)
    print("Env.threat_field: ", env.threat_field)

    threat4 = GaussThreat(location=(2, 8), shape=(0.5, 0.5), intensity=5)

    threat_field.add_threat(threat4)

    print("Env.threat_field: ", env.threat_field)

    sensor_loc = (3, 3)

    print(threat_field.threats)
    print(threat_field.threats[0])

    for idx, threat in enumerate(threat_field.threats):
        print("Threat ", idx, ", Location ", threat_field.threats[idx].location)

    for threat in range(len(threat_field.threats)):
        print(threat_field.threats[threat])

    threat_val = threat_field.threat_value(x=sensor_loc[0], y=sensor_loc[1])

    print("Threat at sensor location: ", sensor_loc, " = ", threat_val)

    ax1 = draw_threat_field(env=env, threat_field=threat_field)
    # ax1.view_init(azim=0, elev=90)

    ax2 = draw_threat_field_2D(env=env, threat_field=threat_field)
    plt.show()

if __name__ == "__main__":
    main()
