import sys
import os
import numpy as np
import yaml
import pickle
import pandas as pd

sys.path.append(os.path.abspath("/Users/queissman/AIRMO/Code"))
import py4cats
import lib.hapi as hap
import import_spectra
import utilities
import matplotlib.pyplot as plt
from lib import absorption_props, libATM, libRT

config_input_file = "/Users/queissman/AIRMO/Code/simpleDOAS/l1bl2_config_baseline.yaml"
project_path  = '../'
#input_path_spectra="/Users/queissman/AIRMO/DATA/Avantes_test/Test_Jetson2"#read raw spectra and wavelength cal files from
input_path_spectra="/Users/queissman/AIRMO/DATA/e2e_sim_result/241010_201912_467198718641/results"#read raw spectra and wavelength cal files from
hitran_path = "/Users/queissman/AIRMO/Code/simpleDOAS/Data/hapi_data_offline"
hitran_file_name = "hit_CH4_CO2_H2O.par"
fit_range = [1630,1674]#nm as in Krings et al.
#path_agl = "/Users/queissman/AIRMO/Code/e2e-simulator/airmo_e2e/data/AFGL"


'''
To do:
(- connect to Hitran database)
- Use more elaborate sigma calculation (Voigt line)
- read Avantes spectra and calibrate
- Concolve sigma with Instrument IRF
- Levenberg Marquardt fit

columns
Molecule ID 		I2
Isotopologue ID 		I1
ν 	cm-1 	F12.6
S 	cm-1/(molec·cm-2) 	E10.3
A 	s-1 	E10.3
γair 	cm-1·atm-1 	F5.4
γself 	cm-1·atm-1 	F5.3
E" 	cm-1 	F10.4
nair 		F4.2
δair 	cm-1·atm-1 	F8.6
'''

#Steps
#Load raw spectra

#create reference spectrum

#Divide by reference spectrum

#(high pass filter)

#Hitran model xsections sigma

#Compute Avantes transmission I/I_ref

# convolve sigma with instrument function

#Fit modeled transmission exp(- N_CH4 * sigma*)


def main():
    #Do not need this for now as we follow the wavelength from the measured spectra
    local_config = yaml.safe_load(open(config_input_file))
    nlay = local_config["atmosphere"]["nlay"]
    dzlay =local_config["atmosphere"]["dzlay"]
    psurf = local_config["atmosphere"]["psurf"]
    wave_start = local_config["spec_settings"]["wave_start"]
    wave_end = local_config["spec_settings"]["wave_end"]
    wave_extend = local_config["spec_settings"]["wave_extend"]
    dwave_lbl = local_config["spec_settings"]["dwave"]
    wave_lbl = np.arange(wave_start , wave_end , dwave_lbl)  # nm
    wave = wave_lbl
    nlev = nlay + 1  # number of levels
    zlay = (np.arange(nlay - 1, -1, -1) + 0.5) * dzlay  # altitude of layer midpoint
    zlev = np.arange(nlev - 1, -1, -1) * dzlay  # altitude of layer interfaces = levels
    print("wave_start and wave_end", wave_start, wave_end)


    #+++++++++++++++++++++++ model sigmas ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    '''
    #compute xsections SRON -- does not want to connect to Hitran
    atm = libATM.atmosphere_data(zlay, zlev, psurf)
    data_afgl = "/Users/queissman/AIRMO/Code/e2e-simulator/src/airmo_e2e/data/AFGL"
    atm.get_data_AFGL(os.path.join(data_afgl, local_config["std_atm"]) )
    molec = libRT.molecular_data(wave_lbl)
    iso_ids = [("CH4", 32), ("H2O", 1), ("CO2", 7)]  # see hapi manual  sec 6.6
    xsdb_path = "/Users/queissman/AIRMO/Code/e2e-simulator/src/airmo_e2e/data/hapi_data"
    molec.get_data_HITRAN(xsdb_path, iso_ids)
    zlay = [500] #for now
    #optics = absorption_props.OpticsAbsProp(wave_lbl, zlay)
    optics = libRT.OpticsAbsProp(wave_lbl, zlay)

    # Molecular absorption optical properties
    optics.cal_molec_xsec(molec, atm)
    '''
    #wl, xsecx = optics.get_xsect(xsdb_path, iso_ids)


    #Temporary until sigma computation
    read_sigma_from = os.path.join(input_path_spectra,"tmp", "optics_prop_241010_201912_467198718641.pkl" )
    data = pickle.load(open(read_sigma_from, "rb"))
    #for item in data:
    #    print(item)
    sigma_CH4 = data["molec_32"]["xsec"][:,-1]  #lowest layer only
    sigma_CO2 = data["molec_07"]["xsec"][:,-1]
    sigma_H2O = data["molec_01"]["xsec"][:,-1]
    #print("Min, max wavelength of modeleld Xsecs: ", np.min(xsecs_CH4), np.max(xsecs_CH4))

    #utilities.plot_spec(wave_lbl, sigma_CH4[:, -1], title="xsec methane as read from pickle file")  # only bottom layer
    #utilities.plot_spec(wave_lbl, sigma_CO2, title="xsec CO2 as read from pickle file")  # only bottom layer

    #test hitran simple
    #data = hitran_simple.read_hitran2012_parfile(os.path.join(hitran_path,hitran_file_name))
    #wl, xsec= hitran_simple.calculate_hitran_xsec(data, wavemin=wave_start/1000, wavemax=wave_end/1000, npts=len(wave_lbl), units="m^2")
    #utilities.plot_spec(wl, xsec)  # only bottom layer
    '''
    print(data["linecenter"].shape)
    dat_for_4cats = np.zeros((len(data),5))
    dat_for_4cats[:,0] = data["linecenter"]
    dat_for_4cats[:,1] = data["S"]
    dat_for_4cats[:,2] = data["Acoeff"]
    dat_for_4cats[:,3] = data["gamma-air"]
    dat_for_4cats[:,4] = data["gamma-self"]
    '''

    #lines=py4cats.lbl.hitran.extract_hitran(os.path.join(hitran_path,hitran_file_name))
    #works needs to be processed
    lines=py4cats.higstract(os.path.join(hitran_path,hitran_file_name), xLimits=[1e7/1680, 1e7/1500]  )


    wn_Limits = [1e7/wave_end, 1e7/wave_start]
    XS_dict = py4cats.lbl2xs(lines, pressure=None, temperature=None, xLimits=wn_Limits , lineShape='Voigt', sampling=5.0,
                   nGrids=3, gridRatio=8, nWidths=25.0, lagrange=2, verbose=False)
    print(XS_dict["CO2"].shape)

    wn_out = np.arange(wn_Limits[0],wn_Limits[1], (wn_Limits[1]-wn_Limits[0]) / len(XS_dict["CH4"]) )
    wl_out = 1e7/wn_out #interpolate to wave_lbl if neccesary
    #take py4cats
    sigma_CH4_ = np.flip(XS_dict["CH4"][:])
    sigma_CO2_ = np.flip(XS_dict["CO2"][:])
    sigma_H2O_ = np.flip(XS_dict["H2O"][:])
    wl_out = np.flip(wl_out)
    plt.plot(wave, sigma_CH4)
    plt.title("SRON")
    plt.plot(wl_out,sigma_CH4_)
    plt.title("py4cats")
    plt.show()

    #utilities.plot_spec(uGrid, XS, title="xsec methane as read from py4cats")  # only bottom layer

     #PREPARE SIGMA MODELED FROM HITRAN
    wave_lbl_conv, sigma_CH4_conv, i, j, slit = hap.convolveSpectrum(Omega=wave_lbl, CrossSection=sigma_CH4, Resolution=local_config["isrf_settings"]["fwhm"])
    wave_lbl_conv, sigma_CO2_conv, i, j, slit = hap.convolveSpectrum(Omega=wave_lbl, CrossSection=sigma_CO2, Resolution=local_config["isrf_settings"]["fwhm"])
    wave_lbl_conv, sigma_H2O_conv, i, j, slit = hap.convolveSpectrum(Omega=wave_lbl, CrossSection=sigma_H2O, Resolution=local_config["isrf_settings"]["fwhm"])


    #+++++++++++++++++++++++ Load measured spectra ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #we need a good spe ctrum to test!
    #I = import_spectra.load_avantes_spectra(path_in=input_path_spectra) #gets a data frame
    #load_synthetic_spectra looks for .nc file with SGM radiance spectrum

    wl_meas, I_meas = import_spectra.load_synthetic_spectra(path_in=input_path_spectra) #single spectrum just for demo
    print(np.min(wl_meas), np.max(wl_meas))
    print("Length of I", len(I_meas))
    if len(I_meas) == 0:
        exit("Could not find synthetic SGM spectra.")

    #get cross sections the SRON way
    # Internal lbl spectral grid

    #utilities.plot_spec(wl,Iref)
    #convolve sigma and high res spectrum
    print("number of wl radiance before convolution of xsec is ", wl_meas.shape)
    print("number of wl before convolution of xsec is ", wave_lbl.shape)

    #1) compute modeled DOD for each trace gas species
    #2) Add the DOD
    #3) Smooth DOD

    #Simulate measured spectrum by convolution
    wl_Imeas_conv, I_conv, i, j, slit = hap.convolveSpectrum(Omega=wl_meas, CrossSection=I_meas, Resolution=local_config["isrf_settings"]["fwhm"])
    utilities.plot_spec(wl_Imeas_conv, I_conv, ylabel='Radiance (phot m^-2 sr^-1 nm^-1', title="Measured radiance, convolved")

    #print("number of wl radiance after convolution of xsec is ", wl_Imeas_conv.shape)
    #print("number of wl xsec after convolution of xsec is ", wave_lbl_conv.shape)

    df_Imeas = pd.DataFrame(columns=["wavelength", "radiance"])
    df_Imeas["wavelength"] = wl_Imeas_conv
    df_Imeas["radiance"] = I_conv

    #simulate reference spectra from measured spectra (in reality these are measured in ambient or solar spectrum)
    #this polynomial is not to be confused with the polynomial in the fit
    I_ref = utilities.fit_polynomial(measured_spectrum=df_Imeas["radiance"], measured_wavelengths=df_Imeas["wavelength"], degree=8)
    df_Imeas["reference"] = I_ref
    # Plot the measured spectrum and the generated reference spectra

    plt.figure(figsize=(10, 6))
    plt.plot(df_Imeas["wavelength"], df_Imeas["radiance"], label='Measured Spectrum', color='blue', alpha=0.6)
    plt.plot(df_Imeas["wavelength"], df_Imeas["reference"], label='Reference spectra', color='red', linestyle='--')
    plt.title('Measured vs Smoothed and Polynomial-Fitted Reference Spectrum')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity (Arbitrary Units)')
    plt.legend()
    plt.grid(True)
    plt.show()


    #HIgh pass filter out aeorosol contribution
    #fs_ = 1./local_config["spec_settings"]["dwave"] #per nm
    #I_meas_ = utilities.butter_highpass_filter(df_Imeas["radiance"], fs = fs_, cutoff=0.1)
    #df_Imeas["radiance"] = I_meas_
    #utilities.plot_spec(df_Imeas["wavelength"],  df_Imeas["radiance"] , title="Highpass filtered measured radiance")
    #print("length high pass filtered radiance", len(df_Imeas["radiance"]))


    #+++++++++++++++++++++++ Truncate spectra and sigma to wavelength range of interest ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    df_Imeas = df_Imeas[df_Imeas["wavelength"] >= fit_range[0]]
    df_Imeas = df_Imeas[df_Imeas["wavelength"] <= fit_range[1]]

    #Truncate sigmas modeled
    df_xsec = pd.DataFrame(columns=["wavelength", "xsec"])
    df_xsec["wavelength"] = wave_lbl_conv
    df_xsec["xsec_CH4"] = sigma_CH4_conv*1e-4 #convert from cm^2 to m^2
    df_xsec["xsec_CO2"] = sigma_CO2_conv*1e-4
    df_xsec["xsec_H2O"] = sigma_H2O_conv*1e-4
    df_xsec = df_xsec[df_xsec["wavelength"] >= fit_range[0]] #xsec are modeled
    df_xsec = df_xsec[df_xsec["wavelength"] <= fit_range[1]] #xsec are modeled
    print("number of wl xsec after truncation of xsec is ", df_xsec.shape)


    #+++++++++++++++++++++++ Fit DOD modeled to DOD_obs ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #observed_transmittance = np.divide(I,Iref)
    #truncate spectrum to 1635 to 1655 nm
    Nair = 2.5e25 #compute properly from P, T
    length = 20000 #m since we use spectra modelled from satellite for test purposes
    statevector = np.zeros(8)
    statevector[0] = 0 #a
    statevector[1] = 0 #b
    statevector[2] = 0 #c
    statevector[3] = 0 #d
    statevector[4] = 1800e-9*Nair*length #vcd CH4 get fromreal data
    statevector[5] = 405e-6*Nair*length #vcd CO2 get fromreal data
    statevector[6] = 0.008*Nair*length #vcd H2O get fromreal data
    statevector[7] = 0.1 #spectral shift nm

    # f = fit(dummy=True, wavelength = wave_lbl) #dummy to test algorithm
    # f.state_vec = state_v
    print("Length of sigma trunc and Imeas trunc before fit ", len(df_xsec["xsec"]), len(df_Imeas) )
    state_vector_out = utilities.test_fit(state_vector_in=statevector, observed=df_Imeas,
                                          dummy=False, sigma = df_xsec, isfr = local_config["isrf_settings"]["fwhm"])
    print("Methane column density is ", state_vector_out[4])
    print("Methane ppm is ", 1e6*state_vector_out[4]/Nair/length)

if __name__ == '__main__':
    main()

