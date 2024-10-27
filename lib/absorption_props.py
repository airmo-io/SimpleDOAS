import sys
import os, io
from os.path import join
import pickle as pkl
import numpy as np
import yaml
from tqdm import tqdm
from lib import hapi as hp
from contextlib import nullcontext, redirect_stdout
import matplotlib.pyplot as plt
import utilities

trap = io.StringIO()
debug = 0
units = 'cm^2' #only for simple Hitran
#path_to_airmo = "/Users/queissman/AIRMO/Code/e2e-simulator/src"
#sys.path.append(path_to_airmo) #for airmo SRON
config_input_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'l1bl2_config_baseline.yaml') #same as for l12l2
path_xsec = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '../Data/AbsorptionXsections') #wavelengths and wavelength resolution must be as in config_input_file
path_AFGL = "../Data/AFGL"  #standard atmospheric profiles
hitran_file_name = "../Data/Hitran/CH4_Hitran_5000to6700.par"
Tair = 288 #get from measurement or renalaysis


class molecular_data:
    """
    # The molecular_data class collects method for calculating
    # the absorption cross sections of molecular absorbers
    #
    # CONTAINS
    # method __init__(self,wave)
    # method get_data_HITRAN(self,xsdbpath, hp_ids)
    """

    ###########################################################

    def __init__(self, wave):
        """
        # init class
        #
        # arguments:
        #            wave: array of wavelengths [wavelength] [nm]
        #            xsdb: dictionary with cross section data
        """
        self.xsdb = {}
        self.wave = wave

    ###########################################################
    def get_data_HITRAN(self, xsdbpath, hp_ids):
        """
        # Download line parameters from HITRAN web ressource via
        # the hapi tools, needs hapy.py in the same directory
        #
        # arguments:
        #            xsdbpath: path to location where to store the absorption data
        #            hp_ids: list of isotopologue ids, format [(name1, id1),(name2, id2) ...]
        #                    (see hp.gethelp(hp.ISO_ID))
        # returns:
        #            xsdb[id][path]: dictionary with paths to HITRAN parameter files
        """
        # check whether input is in range
        while True:
            if len(hp_ids) > 0:
                break
            else:
                print(
                    "ERROR! molecular_data.get_data_HITRAN: provide at least one species."
                )
                raise StopExecution

        with (
            redirect_stdout(trap) if debug < 2 else nullcontext()
        ):  # ignore output for debuglevel<2
            hp.db_begin(xsdbpath)

        wv_start = self.wave[0]
        wv_stop = self.wave[-1]
        # hp.gethelp(hp.ISO_ID)
        for id in hp_ids:
            key = "%2.2d" % id[1]
            self.xsdb[key] = {}
            self.xsdb[key]["species"] = id[0]
            # write 1 file per isotopologue
            self.xsdb[key]["name"] = "ID%2.2d_WV%5.5d-%5.5d" % (
                id[1],
                wv_start,
                wv_stop,
            )
            # Check if data files are already inplace, if not: download
            if not os.path.exists(
                os.path.join(xsdbpath, self.xsdb[key]["name"] + ".data")
            ) and not os.path.exists(
                os.path.join(xsdbpath, self.xsdb[key]["name"] + ".header")
            ):
                # wavelength input is [nm], hapi requires wavenumbers [1/cm]
                hp.fetch_by_ids(
                    self.xsdb[key]["name"], [id[1]], 1.0e7 / wv_stop, 1.0e7 / wv_start
                )


###########################################################


class OpticsAbsProp: #perhaps put this back in a seperate file
    def __init__(self):
        """
        # init class
        #
        # arguments:
        #            wave: array of wavelengths [wavelength] [nm]
        #            xsdb: dictionary with cross section data
        """
        self.xsdb = {}
        self.prop = {}
        print(config_input_file)
        self.local_config = yaml.safe_load(open(config_input_file))
        print("Configuration used: ", self.local_config)
        # local_config = config["l1bl2"]  # better use constants object
        # Vertical layering
        nlay = self.local_config["atmosphere"]["nlay"]
        dzlay = self.local_config["atmosphere"]["dzlay"]
        self.psurf = self.local_config["atmosphere"]["psurf"]
        wave_start = self.local_config["spec_settings"]["wave_start"]
        wave_end = self.local_config["spec_settings"]["wave_end"]
        wave_extend = self.local_config["spec_settings"]["wave_extend"]
        dwave_lbl = self.local_config["spec_settings"]["dwave"]
        wave_lbl = np.arange(wave_start , wave_end , dwave_lbl)  # nm
        self.wave = wave_lbl
        nlev = nlay + 1  # number of levels
        zlay = (np.arange(nlay - 1, -1, -1) + 0.5) * dzlay  # altitude of layer midpoint
        self.zlay = zlay
        self.zlev = np.arange(nlev - 1, -1, -1) * dzlay  # altitude of layer interfaces = levels

    def get_data_HITRAN(self, xsdbpath, hp_ids):
        """
        # Download line parameters from HITRAN web ressource via
        # the hapi tools, needs hapy.py in the same directory
        #
        # arguments:
        #            xsdbpath: path to location where to store the absorption data
        #            hp_ids: list of isotopologue ids, format [(name1, id1),(name2, id2) ...]
        #                    (see hp.gethelp(hp.ISO_ID))
        # returns:
        #            xsdb[id][path]: dictionary with paths to HITRAN parameter files
        """
        # check whether input is in range
        while True:
            if len(hp_ids) > 0:
                break
            else:
                print(
                    "ERROR! molecular_data.get_data_HITRAN: provide at least one species."
                )
                #raise StopExecution

        with (
            redirect_stdout(trap) if debug < 2 else nullcontext()
        ):  # ignore output for debuglevel<2
            hp.db_begin(xsdbpath)

        wv_start = self.wave[0]
        wv_stop = self.wave[-1]
        # hp.gethelp(hp.ISO_ID)
        for id in hp_ids:
            key = "%2.2d" % id[1]
            self.xsdb[key] = {}
            self.xsdb[key]["species"] = id[0]
            # write 1 file per isotopologue
            self.xsdb[key]["name"] = "ID%2.2d_WV%5.5d-%5.5d" % (
                id[1],
                wv_start,
                wv_stop,
            )
            # Check if data files are already inplace, if not: download
            if not os.path.exists(
                os.path.join(xsdbpath, self.xsdb[key]["name"] + ".data")
            ) and not os.path.exists(
                os.path.join(xsdbpath, self.xsdb[key]["name"] + ".header")
            ):
                # wavelength input is [nm], hapi requires wavenumbers [1/cm]
                hp.fetch_by_ids(
                    self.xsdb[key]["name"], [id[1]], 1.0e7 / wv_stop, 1.0e7 / wv_start
                )


    ###########################################################


    def cal_molec_xsec(self):
        """
        # Calculates molecular absorption cross sections
        #
        # arguments:
        #            molec_data: molec_data object
        #            atm_data: atmosphere_data object
        # returns:
        #            prop['molec_XX']: dictionary with optical properties with XXXX HITRAN identifier code
        #            prop['molec_XX']['xsec']: absorption optical thickness [wavelength, nlay] [-]
        """
        nlay = self.zlay.size
        nwave = self.wave.size
        nu_samp = 0.005  # Wavenumber sampling [1/cm] of cross sections
        # Loop over all isotopologues, id = HITRAN global isotopologue ID
        for id in self.xsdb.keys():
            name = "molec_" + id
            species = self.xsdb[id]["species"]
            print(species)
            # Write absorption optical depth [nwave,nlay] in dictionary / per isotopologue
            self.prop[name] = {}
            self.prop[name]["xsec"] = np.zeros((nwave, nlay))
            self.prop[name]["species"] = species
            # Check whether absorber type is in the atmospheric data structure


            # Loop over all atmospheric layers
            for ki in tqdm(range(len(self.zlay))):
                pi = self.atm_data["play"][ki]
                Ti = self.atm_data["tlay"][ki]
                # Calculate absorption cross section for layer
                nu, xs = hp.absorptionCoefficient_Voigt(
                    SourceTables=self.xsdb[id]["name"],
                    Environment={"p": pi / 1013.25, "T": Ti}, #1013.25 hPa standard pressure
                    WavenumberStep=nu_samp,
                )
                dim_nu = nu.size
                nu_ext = np.insert(nu, 0, nu[0] - nu_samp)
                nu_ext = np.append(nu_ext, nu[dim_nu - 1] + nu_samp)
                xs_ext = np.insert(xs, 0, 0.0)
                xs_ext = np.append(xs_ext, 0.0)
                # Interpolate on wavelength grid provided on input
                self.prop[name]["xsec"][:, ki] = np.interp(
                    self.wave, np.flip(1e7 / nu_ext), np.flip(xs_ext)
                )



    def get_xsect(self):
        '''
        main function perhaps exclude from class and init class OpticsAbsorbProp

        Get absorption cross sections for a simple DOAS retrieval test
        Returns:
        wl, xsecs
        '''
        #optics = libRT.OpticsAbsProp(wave=wave_lbl, zlay=zlay)
        # compute xsections SRON



        # Calculate optical properties
        #xsec file from SRON e2e
        no_pkl = True
        for root, dirs, files in os.walk(path_xsec):
          for fname in files:
            if fname.endswith(".pkl") and not self.local_config["xsec_forced"]:
                print("Found ", fname)
                data = pkl.load(open(os.path.join(path_xsec, fname), "rb"))
                for item in data:
                    print("Items in Optics property file")
                    print(item)
                xsecs = data["molec_32"]["xsec"] #methane
                print(xsecs.shape)
                utilities.plot_spec(self.wave, xsecs[:,-1]) #only bottom layer
                no_pkl = False
                break

        if no_pkl or self.local_config["xsec_forced"]:
                print("No xsec file from SRON e2e found. Computing xsec again")

                # model atmosphere, no we need real P, T for a single homog. layer
                '''
                import lib.libATM as libATM
                atm = libATM.atmosphere_data(self.zlay, self.zlev, self.psurf)
                atm.get_data_AFGL("../Data/AFGL" + self.local_config["std_atm"])
    
                # scale to some reference column mixing ratios
                xco2_ref = self.local_config["atmosphere"].get("xco2_guess", 405.0)  # ppm
                xco2 = np.sum(atm.CO2) / np.sum(atm.air) * 1.0e6
                atm.CO2 = xco2_ref / xco2 * atm.CO2
    
                xch4_ref = self.local_config["atmosphere"].get("xch4_guess", 1800.0)  # ppm
                xch4_profile = np.sum(atm.CH4) / np.sum(atm.air) * 1.0e9
                atm.CH4 *= xch4_ref / xch4_profile
    
                # Safe reference water, ch4 and co2 profiles
                ref_H2O = atm.H2O
                ref_CH4 = atm.CH4
                ref_CO2 = atm.CO2
                '''
                #we do simply (as we have one layer for now)
                self.atm_data = {}
                self.atm_data['zlev'] = np.array([100]) #100 m altitude
                self.atm_data['play'] = np.array([self.psurf]) #100 m altitude
                self.atm_data['tlay'] = np.array([Tair]) #100 m altitude

                #to get real data when we want to retrieve columns for real spectra use renalysis data
                '''
                import lib.libATM as libATM
                month = 3
                lon = 50.
                lat = 0.
                ecmf = libATM.atmosphere_data.get_data_ECMWF_ads_egg4(month, lon, lat)
                self.atm_data['play']  = ecmf.atmo["play"]
                etc.
                '''
                # Download molecular absorption parameter
                #iso_ids = [("CH4", 32), ("H2O", 1), ("CO2", 7)]  # see hapi manual  sec 6.6
                iso_ids = [("CH4", 32)]  # see hapi manual  sec 6.6
                xsdb_path = "/Users/queissman/AIRMO/Code/e2e-simulator/src/airmo_e2e/data/hapi_data"
                self.get_data_HITRAN(xsdb_path, iso_ids)

                # Molecular absorption optical properties
                self.cal_molec_xsec()
                xsecs = self.prop["molec_32"]["xsec"]

                # Dump optics.prop dictionary into temporary pkl file
                dump_to_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', "Data/AbsorptionXsections/optics_prop_new_calc.pkl")
                pkl.dump(self.prop, open(dump_to_file, "wb"))

                '''
                import hitran_simple
                # Init class with optics.prop dictionary
    
                hitran_data = hitran_simple.read_hitran2012_parfile(hitran_file_name) #replace by HAPI
                # print(hitran_data = hitran_data[""])
    
                # so far single layer (i.e., single P and T only)
                wl, xsecs = hitran_simple.calculate_hitran_xsec(hitran_data, wavemin=1550 * 1e-3,
                                                               wavemax=wave_end * 1e-3, npts=len(wave_lbl), units=units,
                                                               temp=Tair, pressure=1e-5 * psurf)  # use path_AFGL
                '''
                utilities.plot_spec(self.wave, xsecs[:,-1]) #only bottom layer


        return self.wave, xsecs




if __name__ == '__main__':
    xs = OpticsAbsProp()
    xs.get_xsect()
