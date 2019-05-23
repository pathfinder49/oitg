"""Fit sinc**2 function to data

Assumes at least half an oscillation is visible within data.
The lowest fitted frequency is determined by the Niquist limit, assuming
equal point spacing (though points may still be spaced irregularly)."""
import numpy as np
import numpy.fft
from scipy.signal import lombscargle
from FitBase import FitBase


def parameter_initialiser(x, y, p):
    x_range = np.max(x) - np.min(x)
    delta_x = x_range / len(x)
    min_w = np.pi / x_range
    max_w = np.pi / delta_x
    omega = np.arange(min_w, max_w, 1.5*min_w)

    p['y0'] = np.mean(y)
    p['x0'] = 0
    p['a'] = (np.max(y) - np.min(y))/2
    # This sets the initial FWHM to be the dominant frequency component *
    p['width'] = 2/2.78 /omega[np.argmax(
        lombscargle(x, y, omega, precenter=True))]
    print(p['width'])


def fitting_function(x, p):

    #sinc^2 fitting function. Here the width is approx FWHM/2.78
    y = p['a']*(np.sinc((x-p['x0'])/p['width']))**2
    y += p['y0']

    return y

sinc_2 = FitBase(['x0', 'y0', 'a', 'width'], fitting_function,
                 parameter_initialiser=parameter_initialiser)

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    n_point = 300
    test_x = np.linspace(-5, 6, n_point)
    test_y = np.sinc(test_x)**2 + np.random.rand(n_point) * 0.1
    plt.figure()
    plt.plot(test_x, test_y)
    p_dict, p_error_dict, x_fit, y_fit = \
        sinc_2.fit(test_x, test_y, evaluate_function=True)
    plt.plot(x_fit, y_fit)
    plt.show()
    print(p_dict)