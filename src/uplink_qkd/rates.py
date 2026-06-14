"""Coincidence and raw-overpass key-rate models for satellite QKD links."""

from math import erf

import numpy as np
from scipy.optimize import minimize


class secure_key_rates:
    """
    Model coincidence rates and secure key rates for detector-pair measurements.

    Parameters
    ----------
    d : int
        Number of detectors per communication partner.
    t_delta : float or array-like
        Timing imprecision used for the detector-pair coincidence windows.
    DC_A : float or array-like
        Dark-count rate or detector-wise dark-count rates for party A.
    DC_B : float or array-like
        Dark-count rate or detector-wise dark-count rates for party B.
    e_b : float
        Intrinsic bit-error probability.
    e_p : float
        Intrinsic phase-error probability.
    f : float, optional
        Error-correction efficiency factor.
    t_dead_A : float or array-like, optional
        Detector dead time or detector-wise dead times for party A.
    t_dead_B : float or array-like, optional
        Detector dead time or detector-wise dead times for party B.
    loss_format : {"loss", "dB"}, optional
        Format used for efficiencies supplied to performance calculations.
    custom : bool, optional
        If ``True``, defer performance optimisation until explicit method calls.
    B0 : float, optional
        Initial brightness scale used by the optimiser.

    """
    def __init__(self, d, t_delta, DC_A, DC_B, e_b, e_p, f=1.1, t_dead_A=0, t_dead_B=0, loss_format='loss', custom=False,B0=1):
        """
        Initialise detector, timing, error, and optimisation parameters.

        Parameters
        ----------
        d : int
            Number of detectors per communication partner.
        t_delta : float or array-like
            Timing imprecision used for the detector-pair coincidence windows.
        DC_A : float or array-like
            Dark-count rate or detector-wise dark-count rates for party A.
        DC_B : float or array-like
            Dark-count rate or detector-wise dark-count rates for party B.
        e_b : float
            Intrinsic bit-error probability.
        e_p : float
            Intrinsic phase-error probability.
        f : float, optional
            Error-correction efficiency factor.
        t_dead_A : float or array-like, optional
            Detector dead time or detector-wise dead times for party A.
        t_dead_B : float or array-like, optional
            Detector dead time or detector-wise dead times for party B.
        loss_format : {"loss", "dB"}, optional
            Format used for efficiencies supplied to performance calculations.
        custom : bool, optional
            If ``True``, defer performance optimisation until explicit method
            calls.
        B0 : float, optional
            Initial brightness scale used by the optimiser.

        Returns
        -------
        None

        """
        self.f = f
        self.d = d  # number of detectors per communication partner.
        self.bit_error = e_b
        self.phase_error = e_p
        
        self.set_darkcounts(DC_A, DC_B)
        self.set_jitter( t_delta)
        self.set_dead_time(t_dead_A, t_dead_B)

        self.loss_format = loss_format

        if not custom:
            self.optimal_params, self.optimal_key_rate = self.optimize_performance(B0)

    def __dB_to_loss__(self):
        """
        Convert stored detector efficiencies from dB loss to linear efficiency.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        self.efficiencies_A = 10 ** (-np.array(self.efficiencies_A) / 10)
        self.efficiencies_B = 10 ** (-np.array(self.efficiencies_B) / 10)

    def set_darkcounts(self, DC_A, DC_B):
        """
        Store dark-count rates for both communication partners.

        Parameters
        ----------
        DC_A : float or array-like
            Dark-count rate or detector-wise dark-count rates for party A.
        DC_B : float or array-like
            Dark-count rate or detector-wise dark-count rates for party B.

        Returns
        -------
        None

        """
        if isinstance(DC_A, int) or isinstance(DC_A, float):
            self.dark_counts_A = DC_A * np.ones(self.d)
        else:
            self.dark_counts_A = DC_A
        if isinstance(DC_B, int) or isinstance(DC_B, float):
            self.dark_counts_B = DC_B * np.ones(self.d)
        else:
            self.dark_counts_B = DC_B

    def set_jitter(self, t_delta):
        """
        Store timing imprecision values for detector-pair coincidence windows.

        Parameters
        ----------
        t_delta : float or array-like
            Timing imprecision value or detector-pair timing imprecision values.

        Returns
        -------
        None

        """
        if isinstance(t_delta, float) or isinstance(t_delta, int):
            self.timing_imprecision = t_delta * np.ones(self.d**2)
        else:
            self.timing_imprecision = t_delta

    def set_dead_time(self, t_dead_A, t_dead_B):
        """
        Store detector dead times for both communication partners.

        Parameters
        ----------
        t_dead_A : float or array-like
            Detector dead time or detector-wise dead times for party A.
        t_dead_B : float or array-like
            Detector dead time or detector-wise dead times for party B.

        Returns
        -------
        None

        """
        if isinstance(t_dead_A, float) or isinstance(t_dead_A, int):
            self.t_dead_A = t_dead_A * np.ones(self.d)
        else:
            self.t_dead_A = t_dead_A
        if isinstance(t_dead_B, float) or isinstance(t_dead_B, int):
            self.t_dead_B = t_dead_B * np.ones(self.d)
        else:
            self.t_dead_B = t_dead_B

    def __coincidence_window_loss__(self, x, j, k):
        """
        Calculate the timing-window acceptance for one detector pair.

        Parameters
        ----------
        x : float
            Coincidence-window duration.
        j : int
            Detector index for party A.
        k : int
            Detector index for party B.

        Returns
        -------
        window_efficiency : float
            Timing-window acceptance for detector pair ``(j, k)``.

        """
        return erf(np.sqrt(np.log(2)) * (x / self.timing_imprecision[j + k*self.d]))

    def __total_efficiency__(self, eff, b, t_dead):
        """
        Calculate detector efficiency including dead-time reduction.

        Parameters
        ----------
        eff : float
            Detector efficiency before dead-time reduction.
        b : float
            Source brightness.
        t_dead : float
            Detector dead time.

        Returns
        -------
        total_efficiency : float
            Detector efficiency after dead-time reduction.

        """
        return eff / (1+b*eff*t_dead/self.d)

    def __coincidences_measured__(self, x):
        """
        Calculate the measured coincidence rate.

        Parameters
        ----------
        x : list or array-like
            Coincidence-window duration and brightness, ``[t_CC, brightness]``.

        Returns
        -------
        coincidences : float
            Total measured coincidence rate.

        """
        result = 0
        for j in range(self.d):
            for k in range(self.d):
                # the contribution of true CC
                result += self.__coincidence_window_loss__(x[0], j, k) * x[1] * self.__total_efficiency__(
                    self.efficiencies_A[j], x[1], self.t_dead_A[k]) * self.__total_efficiency__(self.efficiencies_B[k], x[1], self.t_dead_B[k])
                # Contribution of accidental CC
                result += x[0] * (x[1]*self.__total_efficiency__(self.efficiencies_A[j], x[1], self.t_dead_A[k])+self.dark_counts_A[j]) * (
                    x[1]*self.__total_efficiency__(self.efficiencies_B[k], x[1], self.t_dead_B[k])+self.dark_counts_B[k])

        return result

    def __coincidences_erroneous__(self, x, bit_error):
        """
        Calculate the erroneous coincidence rate.

        Parameters
        ----------
        x : list or array-like
            Coincidence-window duration and brightness, ``[t_CC, brightness]``.
        bit_error : float
            Error probability used for the true-coincidence contribution.

        Returns
        -------
        erroneous_coincidences : float
            Total erroneous coincidence rate.

        """
        result = 0
        for j in range(self.d):
            for k in range(self.d):
                if not j == k:
                    # the contribution of true CC
                    result += bit_error * self.__coincidence_window_loss__(x[0], j, k) * x[1] * self.__total_efficiency__(
                        self.efficiencies_A[j], x[1], self.t_dead_A[k]) * self.__total_efficiency__(self.efficiencies_B[k], x[1], self.t_dead_B[k])
                    # Contribution of accidental CC
                    result += x[0] * (x[1]*self.__total_efficiency__(self.efficiencies_A[j], x[1], self.t_dead_A[k])+self.dark_counts_A[j]) * (
                        x[1]*self.__total_efficiency__(self.efficiencies_B[k], x[1], self.t_dead_B[k])+self.dark_counts_B[k])
                    # there's a factor 1/2 here usually but not according to eq.B13
                    
        return result
    
    def __binary_entropy__(self, x):
        """
        Calculate the binary entropy function.

        Parameters
        ----------
        x : float
            Input probability.

        Returns
        -------
        entropy : float
            Binary entropy evaluated at ``x``.

        """
        return -x * np.log2(x) - (1 - x) * np.log2(1 - x)

    def __objective__(self, x):
        """
        Calculate the negative asymptotic secure key-rate objective.

        Parameters
        ----------
        x : list or array-like
            Coincidence-window duration and brightness, ``[t_CC, brightness]``.

        Returns
        -------
        negative_key_rate : float
            Negative secure key rate used by the optimiser.

        """
        q = 0.5

        CC_m = self.__coincidences_measured__(x)
        E_b = self.__coincidences_erroneous__(x, self.bit_error) / CC_m
        E_p = self.__coincidences_erroneous__(x, self.phase_error) / CC_m

        return - q * CC_m * (1.0 - self.f * self.__binary_entropy__(E_b) - self.__binary_entropy__(E_p))

    def custom_performance(self, tcc, B, eff_A, eff_B):
        """
        Calculate key-rate performance for specified system parameters.

        Parameters
        ----------
        tcc : float
            Coincidence-window duration.
        B : float
            Source brightness.
        eff_A : float or array-like
            Efficiency or detector-wise efficiencies for party A.
        eff_B : float or array-like
            Efficiency or detector-wise efficiencies for party B.

        Returns
        -------
        key_rate : float
            Secure key rate for the specified parameters.

        """
        x=[tcc, B]
        
        if isinstance(eff_A, float) or isinstance(eff_A, int):
            self.efficiencies_A = eff_A * np.ones(self.d)
        else:
            self.efficiencies_A = eff_A
        if isinstance(eff_B, float) or isinstance(eff_B, int):
            self.efficiencies_B = eff_B * np.ones(self.d)
        else:
            self.efficiencies_B = eff_B
        if self.loss_format == 'dB':
            self.__dB_to_loss__()
        self.efficiencies_B /= self.d 
        self.efficiencies_A /= self.d 
        return - self.__objective__(x)

    def optimize_performance(self, eff_A, eff_B, B0=1):
        """
        Optimise coincidence-window duration and brightness.

        Parameters
        ----------
        eff_A : float or array-like
            Efficiency or detector-wise efficiencies for party A.
        eff_B : float or array-like
            Efficiency or detector-wise efficiencies for party B.
        B0 : float, optional
            Initial brightness scale used by the optimiser.

        Returns
        -------
        optimal_params : list
            Optimised coincidence-window duration and brightness.
        optimal_key_rate : float
            Secure key rate at the optimised parameters.

        """
        if isinstance(eff_A, float) or isinstance(eff_A, int):
            self.efficiencies_A = eff_A * np.ones(self.d)
        else:
            self.efficiencies_A = eff_A
        if isinstance(eff_B, float) or isinstance(eff_B, int):
            self.efficiencies_B = eff_B * np.ones(self.d)
        else:
            self.efficiencies_B = eff_B
        if self.loss_format == 'dB':
            self.__dB_to_loss__()
        self.efficiencies_B /= self.d 
        self.efficiencies_A /= self.d 
        def obj(x): return self.__objective__(
            [x[0]*self.timing_imprecision[0], x[1]*1e9])
        result = minimize(
            obj, [1, B0], bounds=[(0.001, 100), (1e-5, 1e3)])
        return [result.x[0]*self.timing_imprecision[0], result.x[1]*1e9], -result.fun


def raw_overpass(params, loss_profile, t_delta=0.4e-9,DC_A=200, DC_B=70, t_dead_A=25e-9, t_dead_B=45e-9,power=1):
    """
    Calculate average error rates and measured coincidences over an overpass.

    Parameters
    ----------
    params : list or array-like
        Model parameters ``[intrinsic_heralding_1550, intrinsic_heralding_780,
        qber, qx, Brightness, Tcc]``.
    loss_profile : array-like
        Link-loss profile sampled over the overpass.
    t_delta : float, optional
        Timing imprecision used for the coincidence windows.
    DC_A : float, optional
        Dark-count rate for party A.
    DC_B : float, optional
        Dark-count rate for party B.
    t_dead_A : float, optional
        Detector dead time for party A.
    t_dead_B : float, optional
        Detector dead time for party B.
    power : float, optional
        Multiplicative factor applied to the source brightness.

    Returns
    -------
    avg_qber : float
        Coincidence-weighted average bit-error rate.
    avg_qx : float
        Coincidence-weighted average phase-error rate.
    coincidences : float
        Total measured coincidences over the overpass.

    """
    intrinsic_heralding_1550, intrinsic_heralding_780, qber, qx, Brightness, Tcc = params
    Brightness*=power
    # Brightness, t_delta,intrinsic_heralding_1550, intrinsic_heralding_780,bit_err, phase_err= params
    setup = secure_key_rates(d=4, t_delta=t_delta, DC_A=DC_A, DC_B=DC_B, e_b=qber, e_p=qx, t_dead_A=t_dead_A, t_dead_B=t_dead_B, loss_format='dB', custom=True)
    E_b=[]
    E_p=[]
    CC_m_overpass = []
    for loss in loss_profile:
        setup.efficiencies_B = (intrinsic_heralding_780 + loss)* np.ones(setup.d)
        setup.efficiencies_A = intrinsic_heralding_1550* np.ones(setup.d)
        setup.__dB_to_loss__()
        CC_m = setup.__coincidences_measured__([Tcc, Brightness])
        E_b += [setup.__coincidences_erroneous__([Tcc, Brightness], setup.bit_error) / CC_m]
        E_p += [setup.__coincidences_erroneous__([Tcc, Brightness], setup.phase_error) / CC_m]
        CC_m_overpass += [CC_m]
    # integrate over the positive 
    avg_qber = np.sum(np.array(E_b)*np.array(CC_m_overpass))/np.sum(CC_m_overpass)
    avg_qx = np.sum(np.array(E_p)*np.array(CC_m_overpass))/np.sum(CC_m_overpass)
        
    return avg_qber, avg_qx, np.sum(CC_m_overpass)

def raw_overpass_cutoff(params, loss_profile, cutoff, t_delta=0.4e-9,DC_A=200, DC_B=70, t_dead_A=25e-9, t_dead_B=45e-9,power=1):
    """
    Calculate overpass quantities after applying an error-rate cutoff.

    Parameters
    ----------
    params : list or array-like
        Model parameters ``[intrinsic_heralding_1550, intrinsic_heralding_780,
        qber, qx, Brightness, Tcc]``.
    loss_profile : array-like
        Link-loss profile sampled over the overpass.
    cutoff : float
        Maximum accepted value of the average of bit- and phase-error rates.
    t_delta : float, optional
        Timing imprecision used for the coincidence windows.
    DC_A : float, optional
        Dark-count rate for party A.
    DC_B : float, optional
        Dark-count rate for party B.
    t_dead_A : float, optional
        Detector dead time for party A.
    t_dead_B : float, optional
        Detector dead time for party B.
    power : float, optional
        Multiplicative factor applied to the source brightness.

    Returns
    -------
    avg_qber : float
        Coincidence-weighted average bit-error rate after applying the cutoff.
    avg_qx : float
        Coincidence-weighted average phase-error rate after applying the cutoff.
    coincidences : float
        Total accepted measured coincidences over the overpass.

    """
    intrinsic_heralding_1550, intrinsic_heralding_780, qber, qx, Brightness, Tcc = params
    Brightness*=power
    # Brightness, t_delta,intrinsic_heralding_1550, intrinsic_heralding_780,bit_err, phase_err= params
    setup = secure_key_rates(d=4, t_delta=t_delta, DC_A=DC_A, DC_B=DC_B, e_b=qber, e_p=qx, t_dead_A=t_dead_A, t_dead_B=t_dead_B, loss_format='dB', custom=True)
    E_b=[]
    E_p=[]
    CC_m_overpass = []
    for loss in loss_profile:
        setup.efficiencies_B = (intrinsic_heralding_780 + loss)* np.ones(setup.d)
        setup.efficiencies_A = intrinsic_heralding_1550* np.ones(setup.d)
        setup.__dB_to_loss__()
        CC_m = setup.__coincidences_measured__([Tcc, Brightness])
        bit_error = setup.__coincidences_erroneous__([Tcc, Brightness], setup.bit_error) / CC_m
        phase_error = setup.__coincidences_erroneous__([Tcc, Brightness], setup.phase_error) / CC_m
        if (bit_error+phase_error)/2<cutoff:
            E_b += [bit_error]
            E_p += [phase_error]
            CC_m_overpass += [CC_m]
    # integrate over the positive 
    avg_qber = np.sum(np.array(E_b)*np.array(CC_m_overpass))/np.sum(CC_m_overpass)
    avg_qx = np.sum(np.array(E_p)*np.array(CC_m_overpass))/np.sum(CC_m_overpass)
        
    return avg_qber, avg_qx, np.sum(CC_m_overpass)

def raw_overpass_instant(params, loss_profile, t_delta=0.4e-9,DC_A=200, DC_B=70, t_dead_A=25e-9, t_dead_B=45e-9,power=1):
    """
    Calculate instantaneous error rates and coincidences over an overpass.

    Parameters
    ----------
    params : list or array-like
        Model parameters ``[intrinsic_heralding_1550, intrinsic_heralding_780,
        qber, qx, Brightness, Tcc]``.
    loss_profile : array-like
        Link-loss profile sampled over the overpass.
    t_delta : float, optional
        Timing imprecision used for the coincidence windows.
    DC_A : float, optional
        Dark-count rate for party A.
    DC_B : float, optional
        Dark-count rate for party B.
    t_dead_A : float, optional
        Detector dead time for party A.
    t_dead_B : float, optional
        Detector dead time for party B.
    power : float, optional
        Multiplicative factor applied to the source brightness.

    Returns
    -------
    E_b : numpy.ndarray
        Instantaneous bit-error rates over the loss profile.
    E_p : numpy.ndarray
        Instantaneous phase-error rates over the loss profile.
    CC_m_overpass : numpy.ndarray
        Instantaneous measured coincidences over the loss profile.

    """
    intrinsic_heralding_1550, intrinsic_heralding_780, qber, qx, Brightness, Tcc = params
    Brightness*=power
    # Brightness, t_delta,intrinsic_heralding_1550, intrinsic_heralding_780,bit_err, phase_err= params
    setup = secure_key_rates(d=4, t_delta=t_delta, DC_A=DC_A, DC_B=DC_B, e_b=qber, e_p=qx, t_dead_A=t_dead_A, t_dead_B=t_dead_B, loss_format='dB', custom=True)
    E_b=[]
    E_p=[]
    CC_m_overpass = []
    for loss in loss_profile:
        setup.efficiencies_B = (intrinsic_heralding_780 + loss)* np.ones(setup.d)
        setup.efficiencies_A = intrinsic_heralding_1550* np.ones(setup.d)
        setup.__dB_to_loss__()
        CC_m = setup.__coincidences_measured__([Tcc, Brightness])
        E_b += [setup.__coincidences_erroneous__([Tcc, Brightness], setup.bit_error) / CC_m]
        E_p += [setup.__coincidences_erroneous__([Tcc, Brightness], setup.phase_error) / CC_m]
        CC_m_overpass += [CC_m]
  
    return np.array(E_b), np.array(E_p), np.array(CC_m_overpass)
