#TODO: Add documentation

import numpy as np


def func_rosenbrock(pop, a=1, b=100):
    # http://en.wikipedia.org/wiki/Rosenbrock_function
    x = pop[:, 0]
    y = pop[:, 1]
    return (a - x)**2 + b * (y - x**2)**2


def print_func(func, **kwargs):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from matplotlib import cm

    if func == 'rosenbrock':
        # Initial parammeters
        params = dict(range_=[-1.5, 1.5], step_=0.05)
        # Params in kwargs
        for param in params.keys():
            if param in kwargs:
                params[param] = kwargs[param]
        # Fill grid
        x = np.arange(params['range_'][0], params['range_'][1], params['step_'])
        y = np.arange(params['range_'][0], params['range_'][1], params['step_'])
        x, y = np.meshgrid(x, y)
        pop = np.vstack((x.flatten(), y.flatten())).transpose()
        z = func_rosenbrock(pop)
        z = z.reshape(x.shape)
        # Plot
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.jet, linewidth=0, antialiased=False)
        plt.show()