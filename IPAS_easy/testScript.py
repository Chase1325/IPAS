import sim_ipas as ipas 
import Environment as Env
import Threat
import numpy as np
import random as r
from argparse import ArgumentParser

from Visualize import draw_threat_field, draw_threat_field_2D, draw_path
import matplotlib.pyplot as plt
from IPAS2Pozyx import IPAS2Pozyx, IPAS2PozyxDummy
from Search import AstarPath


def main():
    parser = ArgumentParser(description='Run IPAS Demonstration')
    parser.add_argument('-axis', help='Show axis on threat field', action='store_true')
    parser.add_argument('-threats', help='Number of threats', type=int, default=10)
    parser.add_argument('-default', help='Default field? Overrides -threats', action='store_true')
    parser.add_argument('-x', help='X size of field', type=int, default=10)
    parser.add_argument('-y', help='Y size of field', type=int, default=10)
    parser.add_argument('-optimal', help='Show optimal path', action='store_true')
    parser.add_argument('-simulate', help='Simulate IPAS', action='store_true')
    parser.add_argument('-lastfield', help='Use last field generated', action='store_true')
    parser.add_argument('-demo', help='Demonstrate IPAS', action='store_true')
    parser.add_argument('-sensors', help='Last digit of sensor IPs', nargs='+', type=int)


    args = parser.parse_args()

    env = Env.XYEnvironment(args.x, args.y, args.x+1, args.y+1)

    threats = []
    if args.default:
        threats = [
            Threat.GaussThreat(location=(0, 5), shape=(1.0, 1.0), intensity=10),
            Threat.GaussThreat(location=(2, 2), shape=(0.5, 0.5), intensity=5),
            Threat.GaussThreat(location=(8, 8), shape=(1.0, 1.0), intensity=50),
            Threat.GaussThreat(location=(8, 1), shape=(1.5, 1.5), intensity=10),
            Threat.GaussThreat(location=(2, 8), shape=(0.5, 0.5), intensity=5),
            Threat.GaussThreat(location=(4, 4), shape=(0.5, 0.5), intensity=5),
            Threat.GaussThreat(location=(6, 6), shape=(0.5, 0.5), intensity=5),
            Threat.GaussThreat(location=(1, 9), shape=(1, 1), intensity=20),
            Threat.GaussThreat(location=(3, 6), shape=(1, 1), intensity=50),
            Threat.GaussThreat(location=(10,5), shape=(1, 1), intensity=20),
        ]
    elif args.lastfield:
        with open('/tmp/last_threat_field.csv', 'r') as file:
            for line in file:
                values = line.split(',')
                x = int(values[0])
                y = int(values[1])
                shape = float(values[2])
                intensity = int(values[3])
                threats.append(
                    Threat.GaussThreat(
                        location=(x, y), 
                        shape=(shape, shape), 
                        intensity=intensity,
                    )
                )
                print(threats[-1])
    else:
        with open('/tmp/last_threat_field.csv', 'w') as outfile:
            for i in range(args.threats): 
                min_dim = min(args.x, args.y)
                x = r.randint(0, args.x)
                y = r.randint(0, args.y)
                shape = r.uniform(0.5, 1.5) * min_dim / 10
                intensity = r.randint(min_dim//2, 5*min_dim)
                threats.append(
                    Threat.GaussThreat(
                        location=(x, y), 
                        shape=(shape, shape), 
                        intensity=intensity,
                    )
                )
                print(threats[i])
                print(f"{x},{y},{shape},{intensity}", file=outfile)
            # Force away from outside edge
            threats.append(Threat.GaussThreat(location=(1, args.y - 1), shape=(1, 1), intensity=40))
            print(f"1,{args.y-1},1,40", file=outfile)

    threat_field = Threat.GaussThreatField(threats=threats, offset=0)
    env.add_threat_field(threat_field)

    fig = plt.figure(figsize=(args.x, args.y), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    if not args.axis:
        ax.axis('off')

    draw_threat_field_2D(env, threat_field, ax, colorbar=False)

    with open('/home/roger/share/my_fig.png', 'wb') as outfile:
        fig.canvas.print_png(outfile)

    path = None

    if args.simulate or args.demo:
        input('Press enter to start')

        sensors = [f'192.168.0.1{i}' for i in args.sensors]
        #ipas2pozyx = IPAS2Pozyx()
        ipas2pozyx = IPAS2PozyxDummy()
        Ipas = ipas.IPAS(
            env, 
            sensors, 
            sensor_noise=1, 
            position_converter=ipas2pozyx,
            wait_to_continue=True,
        )

        if args.simulate:
            path = Ipas.simulate()
        elif args.demo:
            path = Ipas.demonstrate()

    #print(Ipas.get_H((5,5)))

    #draw_threat_field(env=env, threat_field=threat_field)

    if path:
        # Plot the threat field as 2D heat map and add path on top
        draw_path(ax, path)

    if args.optimal: 
        draw_path(ax, AstarPath(env, (0,0), (args.x, args.y)), color='yellow')

    with open('/home/roger/share/my_fig_path.png', 'wb') as outfile: 
        plt.gcf().canvas.print_png(outfile)

    plt.show()


if __name__ == "__main__":
    print("\n\n")
    main()
