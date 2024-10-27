import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, butter, filtfilt
from numpy.polynomial import Polynomial
import numpy as np
from scipy.optimize import least_squares
import lib.hapi as hap

# Example of a measured spectrum (just using a synthetic example here)
# Define wavelengths
wavelength = np.linspace(1500, 1700, 500)
smooth_baseline = np.exp(-0.002 * (wavelength - 1600)**2) + 0.02 * (wavelength - 1600)

# Add sharp absorption features at specific wavelengths (high-frequency gas absorption lines)
absorption_features = np.exp(-50 * (wavelength - 1540)**2)  # Absorption line near 1540 nm
absorption_features += np.exp(-40 * (wavelength - 1580)**2)  # Absorption line near 1580 nm
absorption_features += np.exp(-30 * (wavelength - 1620)**2)  # Absorption line near 1620 nm

# Combine to create the measured spectrum
measured_spectrum_test = smooth_baseline - 0.2 * absorption_features + 0.05 * np.random.normal(size=wavelength.shape)


# Method 1: Smoothing the spectrum using a Savitzky-Golay filter
def smooth_spectrum(measured_spectrum=measured_spectrum_test, window_length=11, polyorder=3):
    """
    Smooth the measured spectrum using a Savitzky-Golay filter.
    The savgol_filter (Savitzky-Golay filter) smooths the spectrum while preserving the general shape and avoiding sharp absorption features.
    :param measured_spectrum: The raw measured spectrum (1D array).
    :param window_length: The length of the filter window (must be odd).
    :param polyorder: The order of the polynomial used to fit the samples.
    :return: Smoothed spectrum.
    """
    print("Filtering...")
    return savgol_filter(measured_spectrum, window_length=window_length, polyorder=polyorder)

# Method 2: Fitting a low-order polynomial to the measured spectrum
def fit_polynomial(measured_wavelengths, measured_spectrum=measured_spectrum_test, degree=5):
    """
    Fit a low-order polynomial to the measured spectrum to get reference without gas absorption
    :param measured_wavelengths: The wavelength array corresponding to the spectrum.
    :param measured_spectrum: The raw measured spectrum (1D array).
    :param degree: Degree of the polynomial to fit.
    :return: Fitted polynomial spectrum.
    """
    p = Polynomial.fit(measured_wavelengths, measured_spectrum, degree)

    # Get smoothed and polynomial-fitted reference spectra
    #polynomial_fitted_spectrum = fit_polynomial(wavelength, measured_spectrum_test)
    #smoothed_spectrum = smooth_spectrum(measured_spectrum_test)

    return p(measured_wavelengths)


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def convolve_srf(data_in):
    '''

    :param data_in: sigma high res
    :return: sigma convolved with instrument function
    '''

    sigma_c = data_in #do
    return sigma_c


def truncate_dict(dict_in, wave_min, wave_max, dependent_key="spectrum", truncate_key="wavelength"):
    '''
    truncates spectrum stored as dict according to in and max wavelength
    :param dict_in:
    :return:
    '''
    # Extract wavelength and intensity lists
    wavelengths = np.array(dict_in[truncate_key])
    intensities = np.array(dict_in[dependent_key])
    print(wavelengths)
    print(intensities)

    # Apply the truncation based on the fit range
    mask = (wavelengths >= wave_min) & (wavelengths <= wave_max)
    wavelengths_truncated = wavelengths[mask]
    intensities_truncated = intensities[mask]

    # Update the dictionary with the truncated spectrum
    spectrum_truncated = {
        truncate_key: wavelengths_truncated,
        dependent_key: intensities_truncated
    }
    return spectrum_truncated



def test_fit(state_vector_in, dummy= True, observed = None, sigma=None, isfr = 1):
    '''

    :param wavelengths: wavelengths of observed (measured) transmittance
    :param dummy:
    :param observed_transmittance:
    :param sigma: data frame with sigmas of gases per columns
    :param state_vector_in initial guess
    :return:
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import least_squares
    wavelengths_meas = observed["wavelength"].values
    DOD_obs = np.log(observed["reference"] / observed["radiance"])
    plot_spec(wavelengths_meas, DOD_obs, title="Truncated measured DOD")

    sigma_CH4 = sigma["xsec_CH4"].values
    sigma_CO2 = sigma["xsec_CO2"].values
    sigma_H2O = sigma["xsec_H2O"].values
    wavelengths_mod = sigma["wavelength"].values #modeled wavelengths (because there may be shifts bewteen measured features and modeled there may be a shift, hence we have modeled wl and

    #maybe also convolve observed sectrum here, not in main
    #preconvolve sigmas so they have the same length as wavelengths
    #wavelengths_meas_conv, sigma_CH4, i, j, slit = hap.convolveSpectrum(Omega=wavelengths_meas, CrossSection=sigma_CH4,Resolution=isfr)
    #wavelengths_meas_conv, sigma_CO2, i, j, slit = hap.convolveSpectrum(Omega=wavelengths_meas, CrossSection=sigma_CO2,Resolution=isfr)
    #wavelengths_meas_conv, sigma_H2O, i, j, slit = hap.convolveSpectrum(Omega=wavelengths_meas, CrossSection=sigma_H2O,Resolution=isfr)

    #print("Utilities: Length wl obs, modeled, wl_meas_conv", len(wavelengths_meas), len(wavelengths_mod), len(wavelengths_meas_conv) )

    # measured wl and we fit the shift bewteen the wl)

    # Define a Lorentzian function
    def lorentzian(x, x0, gamma, A):
        """
        Lorentzian function.
        :param x: Wavelengths (independent variable)
        :param x0: Center of the absorption (peak)
        :param gamma: Width of the Lorentzian (gamma)
        :param A: Amplitude of the absorption
        :return: Lorentzian function evaluated at x
        """
        return 1 - A * (gamma ** 2 / ((x - x0) ** 2 + gamma ** 2))

    def model(params, wavelengths_mod):
        '''
        modeled transmittance
        :param params:
        :param wavelengths:
        :return:
        '''
        wl = wavelengths_mod
        a0, a1, a2, a3, vcd_CH4, vcd_CO2,vcd_H2O, delta_lambda = params
        wl_shifted = wl + delta_lambda
        #F(x) = model(x) of transmission
        #include spectral shift parameter
        # state vector set with first guess, dictionary
        polynom = a0 + a1*wl_shifted + a2*wl_shifted**2 + a3*wl_shifted ** 3
        print("model: length wl", wl_shifted.shape)
        print("model: length sigma", sigma_CO2.shape)
        #transm = np.exp(-np.multiply(sigma, vcd))
        DOD_mod = ((vcd_CO2*sigma_CO2) + (vcd_H2O*sigma_H2O) + (vcd_CH4*sigma_CH4))+polynom
        #plot_spec(wave_lbl_conv, sigma_CH4_conv, ylabel='Cross-section in cm^2 /molecule', title="Modelled CH4 Absorption cross section, convolved")

        return DOD_mod


    # Model with 3 Lorentzian minima

    def model_test(params, wavelengths_meas):
        """
        Model of transmittance spectrum with three Lorentzian absorption features.
        :param params: List of parameters for the three Lorentzians (x0_1, gamma_1, A_1, ..., x0_3, gamma_3, A_3)
        :param wavelengths: Array of wavelength values
        :return: Transmittance spectrum
        """
        # Unpack parameters for three Lorentzians
        x0_1, gamma_1, A_1 = params[0:3]
        x0_2, gamma_2, A_2 = params[3:6]
        x0_3, gamma_3, A_3 = params[6:9]

        # Sum of three Lorentzians
        transmittance = (lorentzian(wavelengths_meas, x0_1, gamma_1, A_1) *
                         lorentzian(wavelengths_meas, x0_2, gamma_2, A_2) *
                         lorentzian(wavelengths_meas, x0_3, gamma_3, A_3))

        return transmittance


    # True parameters for the 3 Lorentzian absorption features
    true_params = [1620, 2, 0.3,  # Lorentzian 1: Center 1620 nm, gamma 2 nm, amplitude 0.3
                   1650, 3, 0.5,  # Lorentzian 2: Center 1650 nm, gamma 3 nm, amplitude 0.5
                   1680, 1.5, 0.4]  # Lorentzian 3: Center 1680 nm, gamma 1.5 nm, amplitude 0.4

    # Generate synthetic "observed" transmittance spectrum with noise
    if dummy == True:
        Nair = 2.5e25
        length = 100
        true_params = [0, 0.01, 0.001, 0, 1.8e-6*Nair*length, 0.1]  # polynomial coeff a,b,c,d, vcd, spectral shift
        #observed_transmittance = model_test(true_params, wavelengths) + 0.01 * np.random.normal(size=wavelengths.shape)
        #model = model_test
        #initial_guess = [1622, 1.5, 0.2, 1652, 2.5, 0.4, 1678, 1.0, 0.3]
        DOD_obs = model(true_params, wavelengths_meas) + 0.01 * np.random.normal(size=wavelengths_meas.shape)

    # Define the residuals function (difference between observed and modeled transmittance)
    def residuals(params, wavelengths_mod, DOD_obs):
        return model(params, wavelengths_meas) - DOD_obs

    initial_guess = list(state_vector_in)

    # Initial guess for the parameters (centers, widths, amplitudes)

    # Perform the nonlinear least squares fit using Levenberg-Marquardt algorithm
    result = least_squares(residuals, initial_guess, args=(wavelengths_mod, DOD_obs), method='lm')

    # Extract the optimized parameters
    fitted_params = result.x
    print(f"Fitted Parameters: {fitted_params}")

    # Plot the observed vs. fitted spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(wavelengths_meas, DOD_obs, 'o', label='DOD_obs  (with noise)', markersize=3)
    plt.plot(wavelengths_meas, model(fitted_params, wavelengths_meas), label='Fitted Transmittance', color='red')
    plt.title('DOD_obs vs. Fitted Spectrum')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Transmittance')
    plt.legend()
    plt.grid(True)
    plt.show()


    return fitted_params




def test_fit_convolve_in_model(state_vector_in, dummy= True, observed = None, sigma=None, isfr = 1, I_ref = None):
    '''

    :param wavelengths: wavelengths of observed (measured) transmittance
    :param dummy:
    :param observed_transmittance:
    :param sigma: data frame with sigmas of gases per columns
    :param state_vector_in initial guess
    :return:
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import least_squares
    I_meas = observed["radiance"]
    DOD_obs = np.log(I_meas / I_ref)
    wavelengths_meas = observed["wavelength"].values

    sigma_CH4 = sigma["xsec_CH4"].values
    sigma_CO2 = sigma["xsec_CO2"].values
    sigma_H2O = sigma["xsec_H2O"].values
    wavelengths_mod = sigma["wavelength"].values #modeled wavelengths (because there may be shifts bewteen measured features and modeled there may be a shift, hence we have modeled wl and

    #maybe also convolve observed sectrum here, not in main
    #preconvolve sigmas so they have the same length as wavelengths
    wavelengths_meas_conv, sigma_CH4, i, j, slit = hap.convolveSpectrum(Omega=wavelengths_meas, CrossSection=sigma_CH4,Resolution=isfr)
    wavelengths_meas_conv, sigma_CO2, i, j, slit = hap.convolveSpectrum(Omega=wavelengths_meas, CrossSection=sigma_CO2,Resolution=isfr)
    wavelengths_meas_conv, sigma_H2O, i, j, slit = hap.convolveSpectrum(Omega=wavelengths_meas, CrossSection=sigma_H2O,Resolution=isfr)

    #print("Utilities: Length wl obs, modeled, wl_meas_conv", len(wavelengths_meas), len(wavelengths_mod), len(wavelengths_meas_conv) )

    # measured wl and we fit the shift bewteen the wl)

    # Define a Lorentzian function
    def lorentzian(x, x0, gamma, A):
        """
        Lorentzian function.
        :param x: Wavelengths (independent variable)
        :param x0: Center of the absorption (peak)
        :param gamma: Width of the Lorentzian (gamma)
        :param A: Amplitude of the absorption
        :return: Lorentzian function evaluated at x
        """
        return 1 - A * (gamma ** 2 / ((x - x0) ** 2 + gamma ** 2))

    def model(params, wavelengths_meas):
        '''
        modeled transmittance
        :param params:
        :param wavelengths:
        :return:
        '''
        wl = wavelengths_meas
        a0, a1, a2, a3, vcd_CH4, vcd_CO2,vcd_H2O, delta_lambda = params
        wl_shifted = wl + delta_lambda
        #F(x) = model(x) of transmission
        #include spectral shift parameter
        # state vector set with first guess, dictionary
        polynom = a0 + a1*wl_shifted + a2*wl_shifted**2 + a3*wl_shifted ** 3
        print("model: length wl", wl_shifted.shape)
        print("model: length sigma", sigma_CO2.shape)
        #transm = np.exp(-np.multiply(sigma, vcd))
        DOD = ((vcd_CO2*sigma_CO2) + (vcd_H2O*sigma_H2O) + (vcd_CH4*sigma_CH4))+polynom
        wavelengths_meas_conv, DOD_conv, i, j, slit = hap.convolveSpectrum(Omega=wl, CrossSection=DOD,Resolution=isfr)
        #plot_spec(wave_lbl_conv, sigma_CH4_conv, ylabel='Cross-section in cm^2 /molecule', title="Modelled CH4 Absorption cross section, convolved")

        return DOD_conv


    # Model with 3 Lorentzian minima

    def model_test(params, wavelengths_meas):
        """
        Model of transmittance spectrum with three Lorentzian absorption features.
        :param params: List of parameters for the three Lorentzians (x0_1, gamma_1, A_1, ..., x0_3, gamma_3, A_3)
        :param wavelengths: Array of wavelength values
        :return: Transmittance spectrum
        """
        # Unpack parameters for three Lorentzians
        x0_1, gamma_1, A_1 = params[0:3]
        x0_2, gamma_2, A_2 = params[3:6]
        x0_3, gamma_3, A_3 = params[6:9]

        # Sum of three Lorentzians
        transmittance = (lorentzian(wavelengths_meas, x0_1, gamma_1, A_1) *
                         lorentzian(wavelengths_meas, x0_2, gamma_2, A_2) *
                         lorentzian(wavelengths_meas, x0_3, gamma_3, A_3))

        return transmittance


    # True parameters for the 3 Lorentzian absorption features
    true_params = [1620, 2, 0.3,  # Lorentzian 1: Center 1620 nm, gamma 2 nm, amplitude 0.3
                   1650, 3, 0.5,  # Lorentzian 2: Center 1650 nm, gamma 3 nm, amplitude 0.5
                   1680, 1.5, 0.4]  # Lorentzian 3: Center 1680 nm, gamma 1.5 nm, amplitude 0.4

    # Generate synthetic "observed" transmittance spectrum with noise
    if dummy == True:
        Nair = 2.5e25
        length = 100
        true_params = [0, 0.01, 0.001, 0, 1.8e-6*Nair*length, 0.1]  # polynomial coeff a,b,c,d, vcd, spectral shift
        #observed_transmittance = model_test(true_params, wavelengths) + 0.01 * np.random.normal(size=wavelengths.shape)
        #model = model_test
        #initial_guess = [1622, 1.5, 0.2, 1652, 2.5, 0.4, 1678, 1.0, 0.3]
        DOD_obs = model(true_params, wavelengths_meas) + 0.01 * np.random.normal(size=wavelengths_meas.shape)

    # Define the residuals function (difference between observed and modeled transmittance)
    def residuals(params, wavelengths_meas, DOD_obs):
        return model(params, wavelengths_meas) - DOD_obs

    initial_guess = list(state_vector_in)

    # Initial guess for the parameters (centers, widths, amplitudes)

    # Perform the nonlinear least squares fit using Levenberg-Marquardt algorithm
    result = least_squares(residuals, initial_guess, args=(wavelengths_meas, DOD_obs), method='lm')

    # Extract the optimized parameters
    fitted_params = result.x
    print(f"Fitted Parameters: {fitted_params}")

    # Plot the observed vs. fitted spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(wavelengths_meas, DOD_obs, 'o', label='DOD_obs  (with noise)', markersize=3)
    plt.plot(wavelengths_meas, model(fitted_params, wavelengths_meas), label='Fitted Transmittance', color='red')
    plt.title('DOD_obs vs. Fitted Spectrum')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Transmittance')
    plt.legend()
    plt.grid(True)
    plt.show()


    return fitted_params



def plot_spec(wl, xsec, title = "", ylabel = ""):
    '''
    :param wl:  1D array
    :param xsec: 1D array
    :return:
    '''

    fig = plt.figure()
    #fig.canvas.set_window_title("methane")
    # plt.semilogy(waves, xsec, 'k-')
    plt.plot(wl, xsec, 'k-')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('wavelength (nm)')
    plt.show()
    plt.close()


if __name__ == '__main__':
    '''
    local_config = yaml.safe_load(open("../l1bl2_config_baseline.yaml"))

    wave_start = local_config["spec_settings"]["wave_start"]
    wave_end = local_config["spec_settings"]["wave_end"]
    wave_extend = local_config["spec_settings"]["wave_extend"]
    dwave_lbl = local_config["spec_settings"]["dwave"]
    wave_lbl = np.arange(wave_start - wave_extend, wave_end + wave_extend, dwave_lbl)

    state_v = np.zeros(3)
    state_v[0] = 0.2
    state_v[1] = 1572.
    state_v[2] = 0.01

    #f = fit(dummy=True, wavelength = wave_lbl) #dummy to test algorithm
    #f.state_vec = state_v
    test_fit(wavelengths = wave_lbl, dummy= True)
    '''