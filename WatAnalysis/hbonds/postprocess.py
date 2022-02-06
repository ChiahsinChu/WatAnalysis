"""
Functions for postprocessing of PartialHBAnalysis results
"""
import numpy as np
import os


def count_by_time(hbonds_result, start, stop, step=1):
    """
    Adapted from MDA
    """
    indices, tmp_counts = np.unique(hbonds_result[:, 0],
                                    axis=0,
                                    return_counts=True)
    indices -= start
    indices /= step
    counts = np.zeros_like(np.arange(start, stop, step))
    counts[indices.astype(np.intp)] = tmp_counts
    return np.arange(start, stop, step), counts


def lifetime(hbonds_result,
             start,
             stop,
             step,
             dt,
             tau_max=20,
             window_step=1,
             intermittency=0):
    """
    Adapted from MDA
    """
    from MDAnalysis.lib.correlations import autocorrelation, correct_intermittency

    frames = np.arange(start, stop, step)
    found_hydrogen_bonds = [set() for _ in frames]
    for frame_index, frame in enumerate(frames):
        for hbond in hbonds_result[hbonds_result[:, 0] == frame]:
            found_hydrogen_bonds[frame_index].add(frozenset(hbond[2:4]))

    intermittent_hbonds = correct_intermittency(found_hydrogen_bonds,
                                                intermittency=intermittency)
    tau_timeseries, timeseries, timeseries_data = autocorrelation(
        intermittent_hbonds, tau_max, window_step=window_step)
    output = np.vstack([tau_timeseries, timeseries])
    output[0] = output[0] * dt
    return output


def fit_biexponential(tau_timeseries, ac_timeseries):
    """Fit a biexponential function to a hydrogen bond time autocorrelation function

    Return the two time constants

    Adapted from MDA
    """
    from scipy.optimize import curve_fit

    def model(t, A, tau1, B, tau2):
        """Fit data to a biexponential function.
        """
        return A * np.exp(-t / tau1) + B * np.exp(-t / tau2)

    params, params_covariance = curve_fit(model, tau_timeseries, ac_timeseries,
                                          [1, 0.5, 1, 2])

    fit_t = np.linspace(tau_timeseries[0], tau_timeseries[-1], 1000)
    fit_ac = model(fit_t, *params)

    return params, fit_t, fit_ac


def get_graphs(hbonds_result, start, stop, step, output_dir):
    output_dir = os.path.realpath(output_dir)
    ts, counts = count_by_time(hbonds_result, start)
    tmp_count = 0
    for t, count in zip(ts, counts):
        tmp_result = hbonds_result[tmp_count:tmp_count+count]
        output = get_graph(tmp_result)
        np.save(output, output_dir + "/" + str(t) + ".txt")
        tmp_count = tmp_count + count

        
def get_graph(tmp_result):
    output = 0
    return output