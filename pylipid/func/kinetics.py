##############################################################################
# PyLipID: A python module for analysing protein-lipid interactions
#
# Author: Wanling Song
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
##############################################################################

"""This module contains functions for calculating interaction residence time and koff.
"""
import numpy as np
from scipy.optimize import curve_fit


__all__ = ["cal_koff", "cal_survival_func"]


def cal_koff(durations, t_total, timestep, nbootstrap=10, initial_guess=[1., 1., 1., 1.], cap=True):
    """Calculate residence time based on collected interaction durations via a normalised survival function.

    Parameters
    ----------
    durations : array_like
            Collected interaction durations
    t_total : scalar
            The duration or length, or the longest if using multiple simulations of different durations/lengths, of the
            simulation trajectories. Should be in the same time unit as durations.
    timestep : scalar
            :math:`\Delta t` of the survival function :math:`\sigma`. Often take the time step of the simulation
            trajectories or multiples of the trajectory time step. Should be in the same time unit as durations.
    nbootstrap : int, optional, default=10
            Number of bootstrapping for calculation. The default is 10.
    initial_quess : list, optional, default=(1., 1., 1., 1.)
            The initial guess for bi-exponential fitting to the survival function that is used by
            scipy.optimize.curve_fit.
    cap : bool, optional, default=True
            Cap the returned residence time to t_total.

    Returns
    ----------
    koff : scalar
            The calculated koff. In the unit of "tu"^-1 in which "tu" is the time unit used by the provided durations.
    res_time : scalar
            The calculated residence time. In the same time unit as the provided durations.
    properties : dict
            A dictionary of all the computed values, including the original and bootstrapped koffs, residence times, ks 
            of the bi-expo curve :math:`y=A*e^{(-k_1*x)}+B*e^{(-k_2*x)}` and :math:`R^2`.

    Notes
    ----------
    See `cal_survival_func` for the definition of the survival function :math:`\sigma`.
    A bi-exponential :math:`y=Ae^{-k_1\Delta t}+Be^{-k_2\Delta t}` is used to fit the survival rate curve. The two `ks`
    are stored in the returned dictionary of properties. The smaller k is regarded as the interaction koff and
    the residence time is calculated as :math:`{\frac{1}{koff}}`.

    See also
    -----------
    pylipid.plot.plot_koff
            Plotting function for interaction durations and the calculated survival function.

    """
    # calculate original residence time
    delta_t_list = np.arange(0, t_total, timestep)
    survival_func = cal_survival_func(durations, np.max(t_total), delta_t_list)
    survival_rates = np.array([survival_func[delta_t] for delta_t in delta_t_list])
    res_time, _, r_squared, params = _curve_fitting(survival_func, delta_t_list, initial_guess)
    if cap and res_time > t_total:
        res_time = t_total
    n_fitted = _bi_expo(np.array(delta_t_list), *params)
    r_squared = 1 - np.sum((np.nan_to_num(n_fitted) - np.nan_to_num(survival_rates)) ** 2) / np.sum(
        (survival_rates - np.mean(survival_rates)) ** 2)
    ks = [abs(k) for k in params[:2]]
    ks.sort() # the smaller k is considered as koff

    # calculate bootstrapped residence time
    if nbootstrap > 0:
        duration_boot_set = [np.random.choice(durations, size=len(durations)) for dummy in range(nbootstrap)]
        ks_boot_set = []
        r_squared_boot_set = []
        survival_rates_boot_set = []
        n_fitted_boot_set = []
        for duration_boot in duration_boot_set:
            survival_func_boot = cal_survival_func(duration_boot, np.max(t_total), delta_t_list)
            survival_rates_boot = np.array([survival_func_boot[delta_t] for delta_t in delta_t_list])
            _, _, r_squared_boot, params_boot = _curve_fitting(survival_func_boot, delta_t_list,
                                                                                                initial_guess)
            n_fitted_boot = _bi_expo(np.array(delta_t_list), *params_boot)
            r_squared_boot = 1 - np.sum((np.nan_to_num(n_fitted_boot) - np.nan_to_num(survival_rates_boot)) ** 2) / np.sum(
                (survival_rates_boot - np.mean(survival_rates_boot)) ** 2)
            ks_boot = [abs(k) for k in params_boot[:2]]
            ks_boot.sort()
            ks_boot_set.append(ks_boot)
            r_squared_boot_set.append(r_squared_boot)
            survival_rates_boot_set.append(survival_rates_boot)
            n_fitted_boot_set.append(n_fitted_boot)
    else:
        ks_boot_set = [0]
        r_squared_boot_set = [0]
        survival_rates_boot_set = [0]
        n_fitted_boot_set = [0]

    properties = {"ks": ks, "res_time": res_time, "delta_t_list": delta_t_list,
                  "survival_rates": survival_rates, "survival_rates_boot_set": survival_rates_boot_set,
                  "n_fitted": n_fitted, "n_fitted_boot_set": n_fitted_boot_set,
                  "ks_boot_set": ks_boot_set,
                  "r_squared": r_squared, "r_squared_boot_set": r_squared_boot_set}

    return ks[0], res_time, properties


def cal_survival_func(durations, t_total, delta_t_list):
    """Compute the normalised survival function :math:`\sigma` based on the given durations

    Parameters
    -----------
    durations : array_like
            Collected interaction durations
    t_total : scalar
            The duration or length, or the longest if using multiple simulations of different durations/lengths, of the
            simulation trajectories. Should be in the same time unit as durations.
    delta_t_list : array_like
            The list of :math:`\Delta t` for the survival function :math:`\sigma` to check the interaction survival rate.

    Returns
    -----------
    survival_func : dict
            The survival function :math:`\sigma` stored in a dictionary {delta_t: survival rate}.

    Notes
    -----------
    The normalised survival function

    .. math::
            \sigma\left(t\right)=\frac{1}{N_j}\frac{1}{T-t}\sum_{j=1}^{N_j}\sum_{\nu=0}^{T}{{\tilde{n}}_j\left(\nu,v+t\right)}

    where `T` is the duration of simulation trajectories (`t_total`), :math:`N_j` the number of interactions collected
    from simulations (i.e. len(durations)), and :math:`{\tilde{n}}_j(\nu,v+\Delta t)` is a function that takes the
    value 1 if an interaction appeared for a continuous duration of :math:`\Delta t` after forming the contact at time v,
    and takes the value 0 if otherwise.

    """
    num_of_contacts = len(durations)
    survival_func = {}
    for delta_t in delta_t_list:
        if delta_t == 0:
            survival_func[delta_t] = 1
            survival_func0 = float(sum([res_time - delta_t for res_time in durations if res_time >= delta_t])) / \
                     ((t_total - delta_t) * num_of_contacts)
        else:
            try:
                survival_func[delta_t] = float(sum([res_time - delta_t for res_time in durations if res_time >= delta_t])) / \
                                 ((t_total - delta_t) * num_of_contacts * survival_func0)
            except ZeroDivisionError:
                survival_func[delta_t] = 0
    return survival_func


def _curve_fitting(survival_func, delta_t_list, initial_guess):
    """Fit the exponential curve :math:`y=Ae^{-k_1\Delta t}+Be^{-k_2\Delta t}`"""
    survival_rates = np.nan_to_num([survival_func[delta_t] for delta_t in delta_t_list]) # y
    try:
        popt, pcov = curve_fit(_bi_expo, np.array(delta_t_list), np.array(survival_rates), p0=initial_guess, maxfev=100000)
        n_fitted = _bi_expo(np.array(delta_t_list, dtype=np.float128), *popt)
        r_squared = 1 - np.sum((np.nan_to_num(n_fitted) -
                                np.nan_to_num(survival_rates))**2)/np.sum((survival_rates - np.mean(survival_rates))**2)
        ks = [abs(k) for k in popt[:2]]
        koff = np.min(ks)
        res_time = 1/koff
    except RuntimeError:
        koff = 0
        res_time = 0
        r_squared = 0
        popt = [0, 0, 0, 0]
    return res_time, koff, r_squared, popt


def _bi_expo(x, k1, k2, A, B):
    """The exponential curve :math:`y=Ae^{-k_1\Delta t}+Be^{-k_2\Delta t}`"""
    return A*np.exp(-k1*x) + B*np.exp(-k2*x)
