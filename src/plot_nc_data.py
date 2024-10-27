from netCDF4 import Dataset
import yaml
import os
import pickle
import matplotlib.pyplot as plt
#from airmo_e2e.config_manager import ConfigCollection


import numpy as np
#from scripts.compute_xsec.compute_xsec import results_dir

#nc files resuls from simulations
def open_nc_file(path,fname):
    path_to_nc_file = os.path.join(path,file)
    nc_fid = Dataset(path_to_nc_file)
    print(nc_fid)

    #for SGM atmosphere
    if fname.lower().startswith("airmo_sgm_atmosphere_individual_spectra"):
        dcol_ch4 = nc_fid.variables["dcol_ch4"]  # extract/copy the data
        print(dcol_ch4.shape)
    elif fname.lower().startswith("airmo_sgm_radiance_individual_spectra"):
        #SGM radiance spectra
        #spectra = nc_fid.groups["OBSERVATION_DATA"].variables["radiance"]  # extract/copy the data
        spectra = nc_fid.variables["radiance"]  # extract/copy the data
        #wl = nc_fid.groups["OBSERVATION_DATA"].variables["wavelengths"]  # extract/copy the dataspectra = nc_fid.groups["OBSERVATION_DATA"].variables["radiance"]  # extract/copy the data
        wl = nc_fid.variables["wavelength"]  # extract/copy the dataspectra = nc_fid.groups["OBSERVATION_DATA"].variables["radiance"]  # extract/copy the data

        print("wl", wl.shape)
        print("spectra shape", spectra.shape)
        plt.plot(wl[:],spectra[0, 0, :])  # lowest layer only
        plt.show()

    elif fname.lower().startswith("airmo_gm_individual_spectra"):
        sza = nc_fid.variables["sza"][:]  # extract/copy the data
        saa = nc_fid.variables["saa"][:]  # extract/copy the data
        vza = nc_fid.variables["vza"][:]  # extract/copy the data
        print("sza", sza)
        print("saa", saa)
        print("vza", vza)
#wl = xsec.variables["wave"]
#xsec = xsec.variables["xsec"]

#pickle files
def open_xsec_file(path, configs_file):
    for root, dirs, files in os.walk(path):
        for fname in files:
            if fname.lower().endswith(".pkl"):
                print("Found ", fname)
                data = pickle.load(open(os.path.join(path, fname), "rb"))
                for item in data:
                    print(item)
                xsecs_CH4 = data["molec_32"]["xsec"]
                print(xsecs_CH4.shape)
                with open(configs_file, "r") as file:
                    local_config= yaml.safe_load(file)
                print(local_config)
                wl = np.arange(
                    local_config["spec_settings"]["wave_start"],
                    local_config["spec_settings"]["wave_end"],
                    local_config["spec_settings"]["dwave"],
                )  # nm

                plt.plot(wl, xsecs_CH4[:,0]) #lowest layer only
                plt.show()
            else:
                print("no pkl file found")



if __name__ == "__main__":
    #basepath = "/Users/queissman/AIRMO/DATA/e2e_sim_result/241010_111630_514030979840/results"
    basepath = "/Users/queissman/AIRMO/DATA/e2e_sim_result/241010_124636_839204348259/results"

    file = "airmo_sgm_atmosphere_individual_spectra_241010_124636_839204348259.nc"
    #file = "airmo_gm_individual_spectra_241010_111630_514030979840.nc"
    #file = "airmo_sgm_radiance_individual_spectra_241010_111630_514030979840.nc"


    open_nc_file(basepath, file)

    exit()
    #to plot sigma
    path_to_config_file = os.path.join(results_dir, "configs", "l1bl2_config_baseline.yaml")

    print("opening file")
    open_xsec_file(path_to_nc_file, path_to_config_file)





