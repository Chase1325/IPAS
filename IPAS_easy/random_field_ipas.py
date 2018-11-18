import random as r
import matplotlib.pyplot as plt

from IPAS_easy import Environment as Env
from IPAS_easy import Threat
from IPAS_easy.Visualize import draw_threat_field_2D


### Generate a Random Threatfield

if __name__ == '__main__':
    x_max = 100
    y_max = 100
    x_pts = x_max
    y_pts = y_max
    env = Env.XYEnvironment(x_size=x_max, y_size=y_max, n_grid_x=x_pts, n_grid_y=y_pts)

    n_threats = 20
    threats = []
    for i in range(20):
        shape = r.randint(int(x_max/2),int(x_max*3/2))/10
        threats.append(
            Threat.GaussThreat(
                location=(r.randint(0,x_max), r.randint(0,y_max)),
                shape=(shape, shape),
                intensity=r.randint(0,50)
            )
        )
        print(threats[-1])

    threat_field = Threat.GaussThreatField(threats=threats, offset=0)
    env.add_threat_field(threat_field)
    ax2 = draw_threat_field_2D(env=env, threat_field=threat_field, colorbar=False)

    plt.axis('off')
    fig1 = plt.gcf()
    plt.show()
    plt.draw()
    fig1.tight_layout()
    fig1.savefig('C:\RESEARCH\Code\IPAS\Figs\my_fig.png', dpi=400, bbox_inches='tight', pad_inches=0)
