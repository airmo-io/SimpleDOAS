import numpy as np
from os.path import join
from os import walk
import pandas as pd
from datetime import datetime
import re
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import random
#import itertools
#import matplotlib.colors as mplcolors

plot_spectra = True

def load_synthetic_spectra(path_in):
    #I_measured = pd.DataFrame(columns=["wavelength", "radiance"])
    for root, dirs, files in walk(path_in):
        for file in files:
            if file.lower().startswith("airmo_sgm_radiance_individual_spectra"):
                # SGM radiance spectra
                print("Found ", file)
                path_to_nc_file = join(path_in, file)
                nc_fid = Dataset(path_to_nc_file)
                spectra = nc_fid.variables["radiance"]  # extract/copy the data
                wl = nc_fid.variables[
                "wavelength"]  # extract/copy the dataspectra = nc_fid.groups["OBSERVATION_DATA"].variables["radiance"]  # extract/copy the data

                print("wl", wl.shape)
                print("spectra shape", spectra.shape)
                plt.plot(wl[:], spectra[0, 0, :])  # lowest layer only
                plt.show()

    #I_measured["wavelength"] = wl
    #I_measured["radiance"] =

    return wl, spectra[0, 0, :]

def load_avantes_spectra(path_in, min_time = "2024-07-16-11-18-59", max_time="2024-07-16-11-19-20"):
    '''
    :param path,min_time minimum and maximum time stamp as in 2024-07-23-10-49-15 (YYYY-MM-DD-HH-MM-SS)
    :return: pandas df with spectra columns
                Timestamp  Wavelengths (nm)  Radiance (au)
            timestamps are in datetie object format
    '''
    datetime_min = datetime.strptime(min_time, '%Y-%m-%d-%H-%M-%S')
    datetime_max = datetime.strptime(max_time, '%Y-%m-%d-%H-%M-%S')

    counter = 0
    df_spectra =pd.DataFrame(columns=['Timestamp','Wavelengths (nm)', 'Radiance (au)'])

    #get wavelengths
    for root, dirs, files in walk(path_in):
        for file in files:
            #get wavelengths, one is enough
            if file.startswith("wavelength") and counter == 0:
                wavelength = np.load(join(path_in, file))
                lenwl = len(wavelength)
                #print("number of wavelengths this file ", l)
                #append to see drifts
                #wavelengths = np.append(wavelengths, np.array(wavelength), axis=0)
        counter +=1

    #get spectra
    matches = 0
    for root, dirs, files in walk(path_in):
        for file in files:
            try:
                match = re.search(r'\d+_(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}\.\d+\+\d{2}_\d{2})', file) #ged rid of sequence number (first number)
                if match:
                    t=match.group(1)
                time = t[:-13].replace(":", "_", 1).replace("_", "-").replace("T", " ")
            except:
                #quick and dirty, skip files with odd time stamp
                continue
            extracted_datetime = datetime.strptime(time, '%Y-%m-%d %H-%M-%S')
            if file.endswith(".npy") and extracted_datetime <= datetime_max and extracted_datetime >= datetime_min:
                matches +=1
                #print("Match: timestamp this file: ", time)

                print("open", file)
                spectrum = np.load(join(path_in, file))
                df_spectra_tmp = pd.DataFrame()
                print("time", time)

                df_spectra_tmp['Timestamp'] = [extracted_datetime]*len(wavelength)
                df_spectra_tmp['Wavelengths (nm)'] = wavelength
                df_spectra_tmp['Radiance (au)'] = spectrum
                df_spectra = pd.concat([df_spectra, df_spectra_tmp], ignore_index=True)

    print("Loaded ", matches, "spectra.")
    print(df_spectra)
    return df_spectra
