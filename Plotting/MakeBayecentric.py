import numpy as np
import matplotlib.pyplot as plt
import Preprocessing.Load_data as ld


# Load in the Bayesian coordinates for a given case input
def get_bayesian_coords(case, path):
    coords = {}
    RANS, LES = ld.load_case_data(case, path, 'no')
    rans_x = RANS['bay_x'].to_numpy()
    print(rans_x.shape)
    rans_y = RANS['bay_y'].to_numpy()
    les_x = LES['bay_x'].to_numpy()
    les_y = LES['bay_y'].to_numpy()
    coords['les'] = (les_x, les_y)
    coords['rans'] = (rans_x, rans_y)
    print(f'Coords.keys = {coords.keys()}')

    return coords


def plot_BaycentricTri(coord):
    ## ------------------------------------------------------------------------  ##
    # plot the boundaries of the triangles
    plt.figure()
    plt.axis('square')
    plt.plot([0, 1], [0, 0], 'r')
    plt.plot([0, 1 / 2], [0, np.sqrt(3) / 2], 'b')
    plt.plot([1, 1 / 2], [0, np.sqrt(3) / 2], 'g')
    plt.title('Barycentric Tri.')
    plt.xlim([-0.1, 1.12])
    plt.ylim([-0.06, 0.93])
    # plt.text(.53,.87,'x_{3c}')
    # plt.text(-.085,0,'x_{2c}')
    # plt.text(1.02,0,'x_{1c}')
    # plt.gca('XColor', 'none','YColor','none')
    ## ------------------------------------------------------------------------  ##
    # compute plottting coordinates
    # l3 = -l1 - l2
    # x = l1 -l2 + 3/2*l3 + 1/2
    # y = np.sqrt(3)/2*(3*l3+1)
    # plt.scatter(x,y)
    color_map = {'les': 'red','rans': 'blue'}
    # Plot each type with a different color and label
    for key in coord:
        x_vals, y_vals = coord[key]
        plt.scatter(x_vals, y_vals,
                    s=20,
                    c=color_map.get(key, 'black'),  # fallback color is 'black'
                    label=key)
    plt.legend()
    plt.show()
    #plt.savefig('foo.png')

def plot_all(trials):
    path = '../data'
    for trial in trials:
        coords = get_bayesian_coords(trial, path)
        plot_BaycentricTri(coords)
    return

