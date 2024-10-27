# CONTAINS
# function clausius_clapeyron(T)
# function read_spectrum_l1_l2_h5py(filenamel1, filenamel2, i, j, wave_start, wave_end, information=False)
# function read_sun_spectrum_S5P(filename)
# function read_sun_spectrum_TSIS1HSRS(filename)
# function transmission(optics, surface, mu0, muv, phi)
# class atmosphere_data()
# class molecular_data()
# class optic_abs_prop()
# class StopExecution(Exception)
# class surface_prop()
###########################################################
import io
import os
import sys
from contextlib import nullcontext, redirect_stdout

import netCDF4 as nc
import numpy as np
from tqdm import tqdm

from lib import hapi as hp

trap = io.StringIO()
debug = 0
###########################################################
# Some global constants
hplanck = 6.62607015e-34  # Planck's constant [Js]
kboltzmann = 1.380649e-23  # Boltzmann's constant [J/K]
clight = 2.99792458e8  # Speed of light [m/s]
e0 = 1.6021766e-19  # Electron charge [C]
g0 = 9.80665  # Standard gravity at ground [m/s2]
NA = 6.02214076e23  # Avogadro number [#/mol]
Rgas = 8.314462  # Universal gas constant [J/mol/K]
XO2 = 0.2095  # Molecular oxygen mole fraction [-]
XCH4 = 1.8e-6  # std CH4 volume mixing ratio 1800 ppb
MDRYAIR = 28.9647e-3  # Molar mass dry air [kg/mol]
MH2O = 18.01528e-3  # Molar mass water [kg/mol]
MCO2 = 44.01e-3  # Molar mass CO2 [kg/mol]
MCH4 = 16.04e-3  # Molar mass CH4 [kg/mol]
LH2O = 2.5e6  # Latent heat of condensation for water [J/kg]
PSTD = 101325  # Standard pressure [Pa]

###########################################################


def set_debuglevel(i):
    global debug
    debug = i
    return


###########################################################


def read_sun_spectrum_S5P(filename):
    """
    # Read sun spectrum Sentinel-5 Precursor (S5P) format
    #
    # in: filepath to solar spectrum
    # out: dictionary with wavelength [nm], irradiance [mW nm-1 m-2], irradiance [ph s-1 cm-2 nm-1]
    """
    # check whether input is in range
    while True:
        if os.path.exists(filename):
            break
        else:
            print("ERROR! read_spectrum_S5P: filename does not exist.")
            raise StopExecution

    # Read data from file
    raw = np.genfromtxt(filename, skip_header=42, unpack=True)

    # Write sun spectrum in dictionary
    sun = {}
    sun["wl"] = raw[0, :]
    sun["mWnmm2"] = raw[1, :]
    sun["phscm2nm"] = raw[2, :]

    return sun


###########################################################


def read_sun_spectrum_TSIS1HSRS(filename):
    """
    # Read sun spectrum TSIS-1 HSRS, downloaded from
    # https://lasp.colorado.edu/lisird/data/tsis1_hsrs,
    # Coddington et al., GRL, 2021, https://doi.org/10.1029/2020GL091709
    # NETCDF format: 'pip install netCDF4'
    #
    # in: filepath to solar spectrum
    # out: dictionary with wavelength [nm], irradiance [W m-2 nm-1]
    """
    # check whether input is in range
    while True:
        if os.path.exists(filename):
            break
        else:
            print("ERROR! read_spectrum_TSIS1HSRS: filename does not exist.")
            sys.exit(filename)

    # Open netcdf file
    ds = nc.Dataset(filename)
    # print(ds.variables)

    # Write sun spectrum in dictionary
    sun = {}
    sun["Wm2nm"] = ds["SSI"][:]  # Solar spectral irradiance [W m-2 nm-1]
    sun["wl"] = ds["Vacuum Wavelength"][:]  # Vacuum Wavelength [nm]
    sun["phsm2nm"] = ds["SSI"][:] / (hplanck * clight) * sun["wl"] * 1e-9

    ds.close

    return sun


def interpolate_sun(sun, wave_lbl):
    spectra = sun["phsm2nm"]  # photons/(cm2 nm)
    wave_sun = sun["wl"][:]
    sun_lbl = np.interp(wave_lbl, wave_sun, spectra)
    # later the first guess of the albedo
    return sun_lbl


############################ß###############################


def transmission(sun_lbl, optics, surface, mu0, muv, deriv=False):
    """
    # Calculate transmission solution given
    # geometry (mu0,muv) using matrix algebra
    #
    # arguments:
    #            optics: optic_prop object
    #            surface: surface_prop object
    #            mu0: cosine of the solar zenith angle [-]
    #            muv: cosine of the viewing zenith angle [-]
    # returns:
    #            rad_trans: single scattering relative radiance [wavelength] [1/sr]
    """
    if not (0.0 <= mu0 <= 1.0 and -1.0 <= muv <= 1.0):
        sys.exit("ERROR! transmission: input out of range.")

    # Number of wavelengths and layers
    nwave = optics.prop["taua"][:, 0].size
    nlay = optics.prop["taua"][0, :].size

    # Total vertical optical thickness per layer (Delta tau_k) [nwave,nlay]
    tauk = optics.prop["taua"]
    # total optical thickness per spectral bin [nwave]
    tautot = np.zeros([nwave])
    tautot[:] = np.sum(tauk, axis=1)
    mueff = abs(1.0 / mu0) + abs(1.0 / muv)
    fact = mu0 / np.pi
    exptot = np.exp(-tautot * mueff)

    rad_trans = sun_lbl * fact * surface.alb * exptot

    if deriv:
        dev_tau = (
            -mueff * rad_trans
        )  # this is the derivative with resoect to tautot and tauk because d tautot /dtauk = 1
        dev_alb = fact * exptot * sun_lbl
        return rad_trans, dev_tau, dev_alb
    else:
        return rad_trans


############################################################


def nonscat_fwd_model(isrf, sun_lbl, atm, optics, surface, mu0, muv, dev=None):
    species = [spec for spec in dev if spec[0:5] == "molec"]

    optics.set_opt_depth_species(atm, species)

    deriv = True
    rad_lbl, dev_tau_lbl, dev_alb_lbl = transmission(
        sun_lbl, optics, surface, mu0, muv, deriv
    )

    fwd = {}

    fwd["rad"] = isrf.isrf_convolution(rad_lbl)
    fwd["rad_lbl"] = rad_lbl
    fwd["alb0"] = isrf.isrf_convolution(dev_alb_lbl)
    fwd["alb1"] = isrf.isrf_convolution(dev_alb_lbl * surface.spec)
    fwd["alb2"] = isrf.isrf_convolution(dev_alb_lbl * surface.spec**2)
    fwd["alb3"] = isrf.isrf_convolution(dev_alb_lbl * surface.spec**3)

    nwave = fwd["rad"].size
    nlay = optics.prop[species[0]]["taualt"][0, :].size

    for spec in species:
        # derivative with respect to a scaling of the total optical depth
        fwd[spec] = isrf.isrf_convolution(
            np.sum(optics.prop[spec]["taualt"], axis=1) * dev_tau_lbl
        )
        # derivative with respect to a scaling of the layer optical deoth
        fwd["layer_" + spec] = np.zeros((nwave, nlay))
        for klay in range(optics.prop[spec]["taualt"][0, :].size):
            fwd["layer_" + spec][:, klay] = isrf.isrf_convolution(
                optics.prop[spec]["taualt"][:, klay] * dev_tau_lbl
            )

    return fwd


###########################################################


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


class OpticsAbsProp:
    """
    # The optic_ssc_prop class collects methods to
    # calculate optical scattering and absorption
    # properties of the single scattering atmosphere
    #
    # CONTAINS
    # method __init__(self, wave, zlay)
    # method cal_isoflat(self, atmosphere, taus_prior, taua_prior)
    # method cal_rayleigh(self, rayleigh, atmosphere, mu0, muv, phi)
    # method combine(self)
    """

    def __init__(self, wave, zlay):
        """
        # init class
        #
        # arguments:
        #            prop: dictionary of contributing phenomena
        #            prop[wave]: array of wavelengths [wavelength] [nm]
        #            prop[zlay]: array of vertical height layers, midpoints [nlay] [m]
        """
        self.prop = {}
        self.wave = wave
        self.zlay = zlay

    def cal_molec_xsec(self, molec_data, atm_data):
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
        for id in molec_data.xsdb.keys():
            name = "molec_" + id
            species = molec_data.xsdb[id]["species"]
            print(species)
            # Write absorption optical depth [nwave,nlay] in dictionary / per isotopologue
            self.prop[name] = {}
            self.prop[name]["xsec"] = np.zeros((nwave, nlay))
            self.prop[name]["species"] = species
            # Check whether absorber type is in the atmospheric data structure

            if species not in atm_data.__dict__.keys():
                print(
                    "WARNING! optic_prop.cal_molec: absorber type not in atmospheric data.",
                    id,
                    species,
                )
            else:
                # Loop over all atmospheric layers
                for ki in tqdm(range(len(atm_data.zlay))):
                    pi = atm_data.play[ki]
                    Ti = atm_data.tlay[ki]
                    # Calculate absorption cross section for layer
                    nu, xs = hp.absorptionCoefficient_Voigt(
                        SourceTables=molec_data.xsdb[id]["name"],
                        Environment={"p": pi / PSTD, "T": Ti},
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

    def set_opt_depth_species(self, atm, species):
        """
        # calaculates absorption optical depth from the various species and combines those as specified
        #
        # arguments:
        #            atmospheric input and species to be combined to total optical depth
        # returns:
        #            prop[taua]: total absorption optical thickness array [wavelength, nlay] [-]
        """
        # check whether input is in range
        while True:
            if len(species) > 0:
                break
            else:
                print("ERROR! optic_prop.combine: name of prop dictionary required.")
                raise StopExecution

        nlay = self.zlay.size
        nwave = self.wave.size

        conv = 1.0e-4  # cross sections are given in cm^2, atmospheric densities in m^2

        for name in self.prop.keys():
            if name[0:5] == "molec":
                self.prop[name]["taualt"] = np.zeros((nwave, nlay))
                spec = self.prop[name]["species"]
                for ki in range(nlay):
                    self.prop[name]["taualt"][:, ki] = (
                        self.prop[name]["xsec"][:, ki]
                        * atm.__getattribute__(spec)[ki]
                        * conv
                    )
        self.prop["taua"] = np.zeros((nwave, nlay))
        for name in species:
            self.prop["taua"] = self.prop["taua"] + self.prop[name]["taualt"]
            self.prop[name]["tau_molec"] = np.zeros((nwave, nlay))

        # plt.plot(np.sum(self.prop['taua'], axis=1))
        # sys.exit('set_opt')
        # sys.exit()
