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
import configparser
import os
import sys
from copy import deepcopy

import netCDF4 as nc
import numpy as np
from netCDF4 import Dataset, num2date
#from pyproj import Geod
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm
from lib import constants, libNumTools


###########################################################

'''
#cannot install geod library on this old mac so decomment this for now
def get_microHH_atm(s2_lat, s2_lon, path_data, microHH_settings, kernel_settings):
    #   time stamp and latitiude, longotude of the source

    time_stamp = microHH_settings["time_stamp"]
    lat_lon_src = microHH_settings["lat_lon_src"]

    # Read the data at a specified time and and microHH simulation
    print("Getting microHH data ...")
    data = read_simulated_variable(path_data, ["co2_m", "no", "no2"], int(time_stamp))
    print("                     ...done")

    # g = Geod(ellps='clrk66') # Use Clarke 1866 ellipsoid.
    g = Geod(ellps="WGS84")  # The World Geodetic System 1984 (WGS84) reference

    # collect grid information to calculate lat lon grid
    xgrid = data.grid.xc
    ygrid = data.grid.yc
    zgrid = data.grid.zc
    xsrc = data.co2_m.source[0]
    ysrc = data.co2_m.source[1]
    zsrc = data.co2_m.source[2]
    nx = xgrid.size
    ny = ygrid.size
    nz = zgrid.size

    long_max = 14.75

    # 2D distance in meters with longitude, latitude of the points
    azimuth1, azimuth2, distance_2d = g.inv(
        lat_lon_src[1], lat_lon_src[0], long_max, lat_lon_src[0]
    )
    nx_plus = np.int32(distance_2d / data.grid.dx) + 1

    print("Expanding microHH domain...")
    # Expand the microHH data granule, need

    xadd = np.arange(1, nx_plus + 1) * data.grid.dx + data.grid.xc[nx - 1]
    data.grid.xc = np.concatenate((data.grid.xc, xadd))
    data.grid.nx = data.grid.xc.size
    data.grid.x_nodes = np.concatenate((data.grid.x_nodes, xadd))
    co2_add = np.empty((nz, ny, nx_plus))
    no_add = np.empty((nz, ny, nx_plus))
    no2_add = np.empty((nz, ny, nx_plus))
    for iz in range(data.grid.nz):
        for iy in range(data.grid.ny):
            co2_add[iz, iy, :] = data.co2_m.conc[iz, iy, nx - 1]
            no_add[iz, iy, :] = data.no.conc[iz, iy, nx - 1]
            no2_add[iz, iy, :] = data.no2.conc[iz, iy, nx - 1]
    data.co2_m.conc = np.concatenate((data.co2_m.conc, co2_add), axis=2)
    data.no.conc = np.concatenate((data.no.conc, no_add), axis=2)
    data.no2.conc = np.concatenate((data.no2.conc, no2_add), axis=2)

    print("Adding latitude/longitude coordinates...")
    data.grid.__setattr__("latc", np.empty([data.grid.ny, data.grid.nx]))
    data.grid.__setattr__("lonc", np.empty([data.grid.ny, data.grid.nx]))

    #   spatial grid for target lat/lon grid.
    transform = libNumTools.TransformCoords(lat_lon_src)
    s2_xc, s2_yc = transform.latlon2xymts(s2_lat, s2_lon)

    # add source offset
    s2_xc = s2_xc + xsrc
    s2_yc = s2_yc + ysrc

    print("Data spatial convolution...")
    # Define the settings for the convolution
    conv_settings = {}
    if kernel_settings["type"] == "2D Gaussian":
        fwhm_x = kernel_settings["fwhm_x"]
        fwhm_y = kernel_settings["fwhm_y"]
        fsize = kernel_settings["size_factor"]

        conv_settings["type"] = kernel_settings["type"]
        conv_settings["1D kernel extension"] = np.int0(
            fsize * np.max([fwhm_x, fwhm_y]) / np.min([data.grid.dx, data.grid.dy])
        )
        # convert all kernel parameter in units of sampling distance
        conv_settings["fwhm x"] = np.int0(float(fwhm_x) / data.grid.dx)
        conv_settings["fwhm y"] = np.int0(float(fwhm_y) / data.grid.dy)

    # convolution of co2 field
    data.__setattr__("conv_co2_m", np.empty([data.grid.nz, data.grid.ny, data.grid.nx]))
    data.__setattr__("conv_no", np.empty([data.grid.nz, data.grid.ny, data.grid.nx]))
    data.__setattr__("conv_no2", np.empty([data.grid.nz, data.grid.ny, data.grid.nx]))

    for iz in range(data.grid.nz):
        data.conv_co2_m[iz, :, :] = libNumTools.convolution_2d(
            data.co2_m.conc[iz, :, :], conv_settings
        )
        data.conv_no[iz, :, :] = libNumTools.convolution_2d(
            data.no.conc[iz, :, :], conv_settings
        )
        data.conv_no2[iz, :, :] = libNumTools.convolution_2d(
            data.no2.conc[iz, :, :], conv_settings
        )

    print("Interpolating data to S2 mesh...")

    # interpolate data to the S2 input grid
    # up to here CO2 is giving as mixing ratio per layer. We vconvert it to subcolumns
    dim_act = s2_lat[0, :].size
    dim_alt = s2_lat[:, 0].size

    microHH_data = Emptyclass()
    microHH_data.__setattr__("lat", s2_lat)
    microHH_data.__setattr__("lon", s2_lon)
    microHH_data.__setattr__("zlay", np.empty([data.grid.nz]))
    microHH_data.__setattr__("zlev", np.empty([data.grid.nz + 1]))
    microHH_data.__setattr__("co2", np.empty([dim_alt, dim_act, data.grid.nz]))
    microHH_data.__setattr__("no", np.empty([dim_alt, dim_act, data.grid.nz]))
    microHH_data.__setattr__("no2", np.empty([dim_alt, dim_act, data.grid.nz]))

    # micro air density kg/m3 Thus, conversion required 1/mdryair * Avogadro
    for iz in tqdm(range(data.grid.nz)):
        conv_fact = data.grid.dz * data.density[iz] / constants.MDRYAIR * constants.NA
        interp_CO2 = RegularGridInterpolator(
            (data.grid.yc, data.grid.xc), data.conv_co2_m[iz, :, :]
        )
        interp_NO = RegularGridInterpolator(
            (data.grid.yc, data.grid.xc), data.conv_no[iz, :, :]
        )
        interp_NO2 = RegularGridInterpolator(
            (data.grid.yc, data.grid.xc), data.conv_no2[iz, :, :]
        )
        microHH_data.co2[:, :, iz] = conv_fact * interp_CO2((s2_yc, s2_xc))
        microHH_data.no[:, :, iz] = conv_fact * interp_NO((s2_yc, s2_xc))
        microHH_data.no2[:, :, iz] = conv_fact * interp_NO2((s2_yc, s2_xc))
        microHH_data.zlay[iz] = data.grid.zc[iz]
        microHH_data.zlev[iz] = data.grid.z_nodes[iz]
    microHH_data.zlev[data.grid.nz] = data.grid.z_nodes[data.grid.nz]

    return microHH_data
'''

def get_AFGL_atm_homogenous_distribution(
    AFGL_path, nlay, dzlay, xco2_ref=405, xch4_ref=None, xh2o_ref=1.0e4
):
    if xch4_ref is None:
        xch4_ref = constants.XCH4 * 1.0e9
    # Vertical layering assuming equidistance gridding in geometrical distance
    nlev = nlay + 1  # number of levels
    # counting from top to bottom
    zlay = (np.arange(nlay - 1, -1, -1.0) + 0.5) * dzlay  # altitude of layer midpoint
    zlev = (
        np.arange(nlev - 1, -1, -1.0) * dzlay
    )  # altitude of layer interfaces = levels
    psurf = 101300.0  # Pa
    atm = atmosphere_data(zlay, zlev, psurf)
    atm.get_data_AFGL(AFGL_path)

    xco2 = np.sum(atm.CO2) / np.sum(atm.air) * 1.0e6
    xch4 = np.sum(atm.CH4) / np.sum(atm.air) * 1.0e9
    xh2o = np.sum(atm.H2O) / np.sum(atm.air) * 1.0e6
    air = np.sum(atm.air)

    atm.CO2 = xco2_ref / xco2 * atm.CO2
    atm.CH4 = xch4_ref / xch4 * atm.CH4
    atm.H2O = xh2o_ref / xh2o * atm.H2O

    return atm


def apply_single_layer_enhancement(atm_std, local_config, nlay):
    """Apply enhancement to the atmospheric standard profile.

    This function applies an enhancement to the atmospheric standard profile based on the provided configuration.
    The enhancement is added at a specific layer.

    Args:
        atm_std (Type): The atmospheric standard profile.
        local_config (Type): The sgm configuration.
        nlay (Type): The number of layers in the atmospheric standard profile.

    Raises:
        ValueError: If the enhancement layer is out of range.
        ValueError: If the enhancement value is not numeric.
    """
    local_config = local_config["atmosphere"].get("individual_spectra", {})
    enhancement_layer = local_config.get("enhancement_layer", None)
    enhancement = local_config.get("enhancement", None)
    if enhancement_layer is not None and enhancement is not None:
        if enhancement_layer < 0 or enhancement_layer >= nlay:
            raise ValueError(
                f"enhancement_layer must be between 0 and {nlay - 1} (inclusive)"
            )

        if not isinstance(enhancement, (int, float)):
            raise ValueError("enhancement must be a numeric value")

        atm_std.CH4[enhancement_layer] = (
            atm_std.CH4[enhancement_layer] + enhancement * np.sum(atm_std.air) * 1.0e-9
        )


class atmosphere_data:
    """
    The atmosphere_data class collects data of
    the thermodynamic state and the composition of
    the atmosphere.

    CONTAINS

    method __init__(self,zlay,zlev) |
    method get_data_AFGL(self,filename) |
        method get_data_ECMWF_ads_egg4(self,filename, month, longitude, latitude)
    """

    ###########################################################

    def __init__(self, zlay, zlev, psurf):
        """
        init class

        Parameters
        ----------
        zlay: array of vertical height layers, mid-points [nlay] [m]
        zlev: array of vertical height levels, boundaries [nlev=nlay+1] [m]
        psurf: scalar of surface pressure [Pa]

        Returns
        -------
        .zlev: array of vertical height levels, boundaries [nlev=nlay+1] [m]
        .zlay: array of vertical height layers [nlay] [m]
        """
        self.__setattr__("zlay", zlay)
        self.__setattr__("dzlay", zlev[0 : zlev.size - 1] - zlev[1 : zlev.size])
        self.__setattr__("zlev", zlev)
        self.__setattr__("psurf", psurf)

    ###########################################################

    def get_data_AFGL(self, filename):
        """
        Read atmospheric data from AFGL database
        file, interpolate on output height grid

        Parameters
        ----------
        filename: str
                file with AFGL data

        Returns
        -------
        .tlev: temperature level profile [nlev] [K]
        .tlay: temperature layer profile [nlev] [K]
        .plev: pressure level profile [nlev] [hPa]
        .play: pressure layer profile [nlay] [hPa]
        .air:  air partial column profile [nlay] [#/m^2]
        .O3:   o3 partial column profile [nlay] [#/m^2]
        .H2O:  h2o prtial column profile [nlay] [#/m^2]
        .CO2:  co2 partial column profile [nlay] [#/m^2]
        .NO2:  no2 partial column profile [nlay] [#/m^2]
        .O2:   o2 partial column profile [nlay] [#/m^2]
        .CH4:  CH4 partial column profile [nlay] [#/m^2]
        """
        # check whether input is in range
        while True:
            if os.path.exists(filename):
                break
            else:
                sys.exit("ERROR! atmosphere_data.get_data_AFGL: file does not exist.")

        # Read AFGL file
        atm_in = np.genfromtxt(filename, skip_header=2)

        zalt_in = atm_in[:, 0] * 1.0e3  # height [km] -> [m]
        press_in = atm_in[:, 1] * 1.0e2  # pressure [Pa]
        # print("pressure",press_in)
        temp_in = atm_in[:, 2]  # temperature [K]
        # print("temperature",temp_in)
        air_in = atm_in[:, 3] * 1e6  # air number density [#/m^3]
        # print("air_in",air_in)
        o3_in = atm_in[:, 4] * 1e6 / air_in  # o3 number density -> mole fraction [-]
        o2_in = atm_in[:, 5] * 1e6 / air_in  # o2 number density -> mole fraction [-]
        h2o_in = atm_in[:, 6] * 1e6 / air_in  # h2o number density -> mole fraction [-]
        # print("h2o",h2o_in)
        co2_in = atm_in[:, 7] * 1e6 / air_in  # co2 number density -> mole fraction [-]
        no2_in = atm_in[:, 8] * 1e6 / air_in  # no2 number density -> mole fraction [-]
        nlev_in = zalt_in.size  # number of input levels

        # [#/m^2 * 1/hPa] air column above P is P*NA/constants.MDRYAIR/g from p = m*kg/area
        sp = constants.NA / (constants.MDRYAIR * constants.g0)

        # truncate or extrapolate the AFGL profile depending on psurf
        # print(press_in[press_in.size-1], self.psurf)
        if press_in[press_in.size - 1] < self.psurf:
            # extrapolation required
            dz = (
                np.log(self.psurf / press_in[press_in.size - 1])
                * constants.Rgas
                * temp_in[temp_in.size - 1]
                / (constants.grav * constants.MDRYAIR)
            )
            press_in = np.append(press_in, self.psurf)
            zalt_in = np.append(zalt_in + dz, 0.0)
            temp_in = np.append(temp_in, temp_in[temp_in.size - 1])
            air_in = np.append(
                air_in,
                press_in[press_in.size - 1]
                * constants.NA
                / (constants.Rgas * temp_in[temp_in.size - 1]),
            )
            o3_in = np.append(o3_in, o3_in[o3_in.size - 1])
            o2_in = np.append(o2_in, o2_in[o2_in.size - 1])
            h2o_in = np.append(h2o_in, h2o_in[h2o_in.size - 1])
            co2_in = np.append(co2_in, co2_in[co2_in.size - 1])
            no2_in = np.append(no2_in, no2_in[no2_in.size - 1])
            nlev_in = nlev_in - 1
        elif press_in[press_in.size - 1] > self.psurf:
            # interpolation required
            # self.psurf is in the interval [press_in[intv], press_in[intv-1]]
            intv = np.searchsorted(press_in, self.psurf)
            press_in = np.append(press_in[0:intv], self.psurf)
            temp_in = temp_in[0 : intv + 1]
            air_in = np.append(
                air_in[0:intv],
                press_in[press_in.size - 1]
                * constants.NA
                / (constants.Rgas * temp_in[temp_in.size - 1]),
            )
            o3_in = o3_in[0 : intv + 1]
            o2_in = o2_in[0 : intv + 1]
            h2o_in = h2o_in[0 : intv + 1]
            co2_in = co2_in[0 : intv + 1]
            no2_in = no2_in[0 : intv + 1]
            zalt_in = zalt_in[0 : intv + 1]
            dz = (
                np.log(press_in[press_in.size - 1] / press_in[press_in.size - 2])
                * constants.Rgas
                * temp_in[temp_in.size - 1]
                / (constants.grav * constants.MDRYAIR)
            )
            zalt_in = np.append(zalt_in[0:intv] - zalt_in[intv - 1] + dz, 0)

        # Interpolate temperature [K] on output layers
        # Flip arrays because our heights are descending
        # (from top to bottom), while np.interp expects ascending order
        self.__setattr__(
            "tlev",
            np.flip(np.interp(np.flip(self.zlev), np.flip(zalt_in), np.flip(temp_in))),
        )
        self.__setattr__(
            "tlay",
            np.flip(np.interp(np.flip(self.zlay), np.flip(zalt_in), np.flip(temp_in))),
        )
        # print("temp",self.tlay,self.tlev)
        # Calculate pressure [hPa] on output levels and layers
        self.__setattr__(
            "plev",
            np.flip(np.interp(np.flip(self.zlev), np.flip(zalt_in), np.flip(press_in))),
        )
        self.__setattr__(
            "play",
            np.flip(np.interp(np.flip(self.zlay), np.flip(zalt_in), np.flip(press_in))),
        )
        # print("press",self.play,self.plev)
        # Calculate the vertical column of air above pressure level
        # and use this to calculate the partial vertical air columns per layer [#/m^2].
        # Partial columns have the advantage that multiplication with cross sections
        # yields optical depth.
        nlev = len(self.zlev)
        nlay = len(self.zlay)
        # [#/m^2 * 1/hPa] air column above P is P*NA/constants.MDRYAIR/g from p = m*g/area
        sp = constants.NA / (constants.MDRYAIR * constants.g0)
        vc_air = sp * self.plev  # air column [#/m^2] above pressure level
        self.air = vc_air[1:nlev] - vc_air[0 : nlev - 1]  # [#/m^2]
        self.air[0] = vc_air[
            0
        ]  # [#/m^2] uppermost layer extends to infinity in terms of number of molecules
        # print(self.air)
        # Interpolate mole fractions on output height grid
        # and then calculate partial columns per layer [#/m^2]
        # ozone
        self.__setattr__(
            "O3",
            np.flip(np.interp(np.flip(self.zlay), np.flip(zalt_in), np.flip(o3_in)))
            * self.air,
        )
        # water vapor
        self.__setattr__(
            "H2O",
            np.flip(np.interp(np.flip(self.zlay), np.flip(zalt_in), np.flip(h2o_in)))
            * self.air,
        )
        # co2
        self.__setattr__(
            "CO2",
            np.flip(np.interp(np.flip(self.zlay), np.flip(zalt_in), np.flip(co2_in)))
            * self.air,
        )
        # no2
        self.__setattr__(
            "NO2",
            np.flip(np.interp(np.flip(self.zlay), np.flip(zalt_in), np.flip(no2_in)))
            * self.air,
        )
        # o2 use a constant mixing ratio
        self.__setattr__("O2", constants.XO2 * self.air)
        # ch4, it is not in the AFGL record so we assume a constant mixing ratio with altitude.
        self.__setattr__("CH4", constants.XCH4 * self.air)

        return

    ###########################################################

    def get_data_ECMWF_ads_egg4(self, filename, month, longitude, latitude):
        """
        Read atmospheric data provided by ECMWF ADS EGG4 run:
        https://ads.atmosphere.copernicus.eu/cdsapp#!/dataset/cams-global-ghg-reanalysis-egg4-monthly

        Download script:

        import cdsapi
        c = cdsapi.Client()
        c.retrieve(
             'cams-global-ghg-reanalysis-egg4-monthly',
             {
                 'variable': [
                     'carbon_dioxide', 'geopotential', 'methane',
                     'relative_humidity', 'temperature',
                 ],
                 'pressure_level': [
                     '1', '2', '3',
                     '5', '7', '10',
                     '20', '30', '50',
                     '70', '100', '150',
                     '200', '250', '300',
                     '400', '500', '600',
                     '700', '800', '850',
                     '900', '925', '950',
                     '1000',
                 ],
                 'year': '2016',
                 'month': [
                     '01', '02', '03',
                     '04', '05', '06',
                     '07', '08', '09',
                     '10', '11', '12',
                 ],
                 'product_type': 'monthly_mean',
                 'format': 'netcdf',
             },
             'download.nc'

        Parameters
        ----------
        filename: str
                filepath to ECMWF netcdf file
        month: int
                [01 ... 12]
        longitude: int
                [0 ... 360 degree]
        latitude: int
                 [-90 ... 90 degree]
        Returns
        -------
        temp: temperature profile [nlay] [K]
        plev: pressure level profile [nlev] [hPa]
        atmo[tlev]: temperature level profile [nlev] [K]
        atmo[tlay]: temperature layer profile [nlev] [K]
        atmo[plev]: pressure level profile [nlev] [hPa]
        atmo[play]: pressure layer profile [nlay] [hPa]
        atmo[AIR]: air partial column profile [nlay] [#/m^2]
        atmo[H2O]: h2o partial column profile [nlay] [#/m^2]
        atmo[CO2]: co2 partial column profile [nlay] [#/m^2]
        atmo[CH4]: no2 partial column profile [nlay] [#/m^2]
        atmo[O2]: o2 partial column profile [nlay] [#/m^2]
        """
        # check whether input is in range
        while True:
            if (
                os.path.exists(filename)
                and 1 <= month <= 12
                and -90.0 <= latitude <= 90.0
                and 0.0 <= longitude <= 360.0
            ):
                break
            elif not os.path.exists(filename):
                print("ERROR! read_ecmwf_ads_egg4: filename does not exist.")
                raise StopExecution
            else:
                print("ERROR! read_ecmwf_ads_egg4: input out of range.")
                raise StopExecution

        # Open netcdf file
        ds = nc.Dataset(filename)
        # print(ds.variables)

        # Select month index, latitude/longitude index (next neighbour)
        itime = int(month - 1)
        ilat = np.argmin(abs(ds["latitude"][:] - latitude))
        ilon = np.argmin(abs(ds["longitude"][:] - longitude))

        # ECMWF: Geopotential [m2 s-2] converted to height [m], approximate use of g0
        zalt_in = np.array([d / g0 for d in ds["z"][itime, :, ilat, ilon]])
        nlev_in = zalt_in.size
        # ECMWF: Pressure [hPa]
        press_in = ds["level"][:]
        # ECMWF: Temperature [K]
        temp_in = ds["t"][itime, :, ilat, ilon]
        # ECMWF: Humidity [%] converted to water vapor mole fraction [mol/mol] via Clausius-Clapeyron equation
        pS = [clausius_clapeyron(Ti) for Ti in temp_in]
        # ECMWF: Mole fraction is partial pressure over dry total pressure, partial pressure is rel. hum. * sat. vapor pressure
        h2o_in = np.array(
            [
                d / 100.0 * pSi / (pi - d / 100.0 * pSi)
                for d, pSi, pi in zip(ds["r"][itime, :, ilat, ilon], pS, press_in)
            ]
        )
        # ECMWF: Carbon dioxide mass mixing ratio [kg kg-1] converted to mole fraction [mol/mol]
        co2_in = np.array([d / MCO2 * MDRYAIR for d in ds["co2"][itime, :, ilat, ilon]])
        # ECMWF: Methane mass mixing ratio [kg kg-1] converted to mole fraction [mol/mol]
        ch4_in = np.array([d / MCH4 * MDRYAIR for d in ds["ch4"][itime, :, ilat, ilon]])
        ds.close

        ########################################################################################################################################
        # [#/m^2 * 1/hPa] air column above P is P*NA/constants.MDRYAIR/g from p = m*g/area
        sp = constants.NA / (constants.MDRYAIR * constants.g0) * 1.0e2
        # air_in=sp
        # truncate or extrapolate the AFGL profile depending on psurf
        # print(press_in[press_in.size-1], self.atmo['psurf'])

        if press_in[press_in.size - 1] < self.atmo["psurf"]:
            # extrapolation required
            dz = (
                np.log(self.atmo["psurf"] / press_in[press_in.size - 1])
                * constants.Rgas
                * temp_in[temp_in.size - 1]
                / (constants.grav * constants.MDRYAIR)
            )
            press_in = np.append(press_in, self.atmo["psurf"])
            zalt_in = np.append(zalt_in - dz, 0.0)
            temp_in = np.append(temp_in, temp_in[temp_in.size - 1])
            # air_in   = np.append(air_in,press_in[press_in.size-1]*constants.NA*1.E-4/(constants.Rgas *temp_in[temp_in.size-1]))
            # o3_in    = np.append(o3_in, o3_in[o3_in.size-1])
            # o2_in    = np.append(o2_in, o2_in[o2_in.size-1])
            h2o_in = np.append(h2o_in, h2o_in[h2o_in.size - 1])
            co2_in = np.append(co2_in, co2_in[co2_in.size - 1])
            ch4_in = np.append(ch4_in, ch4_in[ch4_in.size - 1])
            # no2_in   = np.append(no2_in, no2_in[no2_in.size-1])
            nlev_in = nlev_in - 1
        elif press_in[press_in.size - 1] > self.atmo["psurf"]:
            # interpolation required
            # self.atmo['psurf'] is in the interval [press_in[intv], press_in[intv-1]]
            intv = np.searchsorted(press_in, self.atmo["psurf"])
            press_in = np.append(press_in[0:intv], self.atmo["psurf"])
            temp_in = temp_in[0 : intv + 1]
            # air_in   = np.append(air_in[0:intv],press_in[press_in.size-1]*constants.NA*1.E-4/(constants.Rgas *temp_in[temp_in.size-1]))
            # o3_in    = o3_in[0:intv+1]
            # o2_in    = o2_in[0:intv+1]
            h2o_in = h2o_in[0 : intv + 1]
            co2_in = co2_in[0 : intv + 1]
            ch4_in = ch4_in[0 : intv + 1]
            # no2_in   = no2_in[0:intv+1]
            zalt_in = zalt_in[0 : intv + 1]
            dz = (
                np.log(press_in[press_in.size - 1] / press_in[press_in.size - 2])
                * constants.Rgas
                * temp_in[temp_in.size - 1]
                / (constants.grav * constants.MDRYAIR)
            )
            zalt_in = np.append(zalt_in[0:intv] - zalt_in[intv - 1] + dz, 0)

        # Interpolate temperature [K] on output layers
        # Flip arrays because our heights are descending
        # (from top to bottom), while np.interp expects ascending order
        self.atmo["tlay"] = np.flip(
            np.interp(np.flip(self.atmo["zlay"]), np.flip(zalt_in), np.flip(temp_in))
        )
        self.atmo["tlev"] = np.flip(
            np.interp(np.flip(self.atmo["zlev"]), np.flip(zalt_in), np.flip(temp_in))
        )

        # Calculate pressure [hPa]
        self.atmo["play"] = np.flip(
            np.interp(np.flip(self.atmo["zlay"]), np.flip(zalt_in), np.flip(press_in))
        )
        self.atmo["plev"] = np.flip(
            np.interp(np.flip(self.atmo["zlev"]), np.flip(zalt_in), np.flip(press_in))
        )

        # Calculate the vertical column of air above pressure level
        # and use this to calculate the partial vertical air columns per layer [#/m^2].
        # Partial columns have the advantage that multiplication with cross sections
        # yields optical depth.
        nlev = len(self.atmo["zlev"])
        sp = (
            NA / (MDRYAIR * g0) * 1.0e2
        )  # [#/m^2 * 1/hPa] air column above P is P*NA/Mair/g from p = m*g/area
        vc_air = sp * self.atmo["plev"]  # air column [#/m^2] above pressure level
        self.atmo["AIR"] = vc_air[1:nlev] - vc_air[0 : nlev - 1]  # [#/m^2]
        self.atmo["AIR"][0] = vc_air[
            0
        ]  # [#/m^2] uppermost layer extends to infinity in terms of number of molecules

        # Interpolate mole fractions on output height grid
        # and then calculate partial columns per layer [#/m^2]
        # water vapor
        self.atmo["H2O"] = (
            np.flip(
                np.interp(np.flip(self.atmo["zlay"]), np.flip(zalt_in), np.flip(h2o_in))
            )
            * self.atmo["AIR"]
        )
        # co2
        self.atmo["CO2"] = (
            np.flip(
                np.interp(np.flip(self.atmo["zlay"]), np.flip(zalt_in), np.flip(co2_in))
            )
            * self.atmo["AIR"]
        )
        # no2
        self.atmo["CH4"] = (
            np.flip(
                np.interp(np.flip(self.atmo["zlay"]), np.flip(zalt_in), np.flip(ch4_in))
            )
            * self.atmo["AIR"]
        )
        # self.atmo['CH4'] = XCH4*self.atmo['AIR']
        # o2 use a constant mixing ratio
        self.atmo["O2"] = constants.XO2 * self.atmo["AIR"]

    ###########################################################

    def get_albedo_flat(self, alb_prior):
        """
        Generate spectrally flat albedo array

        Parameters
        ----------
        alb_prior: albedo value to be used throughout spectral range
        Returns
        -------
        alb: constant albedo array [wavelength]
        """
        # check whether input is in range
        while True:
            if 0.0 <= alb_prior <= 1.0:
                break
            else:
                print(
                    "ERROR! surface_prop.get_albedo_flat: albedo prior needs to be in [0...1]."
                )
                raise StopExecution

        # Read data from file
        self.alb = np.array([alb_prior for wi in self.wave])

    ###########################################################
    def get_albedo_poly(self, wave, c_lst):
        """
         Parameters
        ----------
        lambda : wavelenght [nm].
        c_lst: list of constants

        Returns
        -------
        albedo polynomial dependent on wavelenght

        """

        albedo = np.zeros(len(wave))

        for i in range(0, len(c_lst)):
            albedo = albedo + c_lst[i] * (wave) ** (i)

        self.alb = albedo

    ###########################################################
    def get_albedo_CUSTOM(self, filename, sfname):
        """
        Read albedo from custom database. This is generic typical
        data. For a comprehensive albedo database, have a look
        at: https://speclib.jpl.nasa.gov/

        Parameters
        ----------
        filename: str
                file with albedo database
        sfname: str
                name of surface type
                [sand,soil,snow,vegetation,water]
        Returns
        -------
        alb: albedo array interpolated to wavelength [wavelength]
        """
        # check whether input is in range
        sftypes = ["sand", "soil", "snow", "vegetation", "water"]
        while True:
            if os.path.exists(filename) and sfname in sftypes:
                break
            else:
                print("ERROR! surface_prop.get_albedo_CUSTOM: input out of range.")
                raise StopExecution

        # Read data from file
        raw = np.genfromtxt(
            filename, skip_header=15
        )  # read file into numpy array, skip 15 header lines

        # Index of surface type
        isf = sftypes.index(sfname)

        # Interpolate albedo to wavelength array
        wave_org = raw[:, 0]
        alb_org = raw[:, isf + 1]
        self.alb = np.interp(self.wave, wave_org, alb_org)

    ###########################################################
    def get_albedo_ECOSTRESS(self, filename):
        """
        Read albedo from ECOSTRESS database
        at: https://speclib.jpl.nasa.gov/

        Parameters
        ----------
        filename: file with albedo database
        Returns
        -------
        alb: albedo array interpolated to wavelength [wavelength]
        """
        # Check whether input is in range
        while True:
            if os.path.exists(filename):
                break
            else:
                print("ERROR! surface_prop.get_albedo_ECOSTRESS: input out of range.")
                raise StopExecution

        # Read data from file
        raw = np.genfromtxt(filename, skip_header=21, unpack=True)

        wv_in = np.array([a * 1e3 for a in raw[0, :]])  # wavelength [nm]
        alb_in = np.array([a / 1e2 for a in raw[1, :]])  # albedo [0...1]
        # Check if wavelength in ascending order. If not, flip arrays.
        if wv_in[0] > wv_in[-1]:
            # Interpolate albedo to wavelength array
            self.alb = np.interp(self.wave, np.flip(wv_in), np.flip(alb_in))
        else:
            self.alb = np.interp(self.wave, wv_in, alb_in)


class Emptyclass:
    pass


def read_microHH_data(filename, keys_to_read, time=2):
    """
    Read the data

    Parameters
    ----------
    filename : TYPE
        DESCRIPTION.
    keys_to_read : TYPE
        DESCRIPTION.
    time : Integer, optional
        This represents the time index corresponding to a time. The default is 2.

    Returns
    -------
    data container at a time. CO2 is in moles/m^3

    """

    # longitude :   Zonal location
    # latitude :   Meridional location
    # ps :   Surface air pressure
    # zsurf: Surface elevation
    # p :    Air pressure
    # ph :   air pressure at cell edge
    # zh :   height above surface at cell edge [m]
    # z :    layer height above surface [m]
    # ta :   Absolute temperature
    # hus :  Specific humidity
    # ua :   Zonal wind
    # va :   Meridional wind
    # wa :   Vertical wind
    # CO2_PP_L :     CO2 tracer mole fraction low release
    # CO2_PP_M :     CO2 tracer mole fraction medium release
    # CO2_PP_H :     CO2 tracer mole fraction high release
    # CO2_PP_M_ECW : CO2 tracer mole fraction medium release, from individual tower groups

    # time: size of 97, starts at 2018-05-23 00:00:00, time step is 15mins
    #       So 0 is 2018-05-23 00:00:00
    #          1 is 2018-05-23 00:00:15
    #          97 is 2018-05-24 00:00:00

    fl = nc.Dataset(filename, "r")
    # Outside these indices in lat-lon data is masked so no use reading it
    i1 = 36
    i2 = 264
    j1 = 8
    j2 = 252

    # this is dict
    data = Emptyclass()

    # read all data
    for ky in keys_to_read:
        # if "CO2" in ky:
        #     co2_tmp = (fl[ky][time, :, i1:i2, j1:j2]).filled(fill_value=0)
        #     if 'p' not in data.__dict__.keys():
        #         data.__setattr__('p', fl['p'][time, :, i1:i2, j1:j2])
        #     if 'ta' not in data.__dict__.keys():
        #         data.__setattr__("ta", fl['ta'][time, :, i1:i2, j1:j2])
        #     # pV = nRT => n/V = p/RT [N/m2  mol K/J  1/K]  = [mol/ m3]
        #     # air_density = data.p/(data.ta * constants.Rgas)

        #     data.__setattr__(ky, (co2_tmp*data.p)/(data.ta*constants.Rgas).data)
        if ky in ["longitude", "latitude"]:
            data.__setattr__(ky, (fl[ky][i1:i2, j1:j2]).data)
        elif ky in ["ps", "zsurf"]:
            data.__setattr__(ky, (fl[ky][time, i1:i2, j1:j2]).data)
        else:
            data.__setattr__(ky, fl[ky][time, :, i1:i2, j1:j2])
    fl.close()

    return data


def combine_microHH_standard_atm(microHH, atm_std):
    nalt = microHH.lat[:, 0].size
    nact = microHH.lat[0, :].size
    #   index of standard atmosphere that corresp
    print("Combining microHH and AFGL model aonds to TOA of microHH")
    idx = (np.abs(atm_std.zlev - np.max(microHH.zlev))).argmin()

    if atm_std.zlev[idx] != np.max(microHH.zlev):
        sys.exit(
            "vertical grids are not alligned in libATM.combine_microHH_standard_atm"
        )
    #    zlev_tmp = np.concatenate()
    zmicroHH = microHH.zlay
    zatm = atm_std.zlay
    nlay = zatm.size

    #   define a mask for vertical integration of microHH data
    ztop = atm_std.zlev[idx:nlay]
    zbot = atm_std.zlev[idx + 1 : nlay + 1]

    midx = {}
    for ilay in range(ztop.size):
        midx[ilay] = np.where((zmicroHH < ztop[ilay]) & (zmicroHH >= zbot[ilay]))

    #   integrate microHH and use it to replace the lowest four values of atm_std

    atm = np.ndarray((nalt, nact), np.object_)

    # for NO we assume no contribution above upper model boundary of microHH

    for ialt in tqdm(range(nalt)):
        for iact in range(nact):
            atm[ialt, iact] = deepcopy(atm_std)
            atm[ialt, iact].__setattr__("NO", np.zeros(nlay))

            for ilay in range(ztop.size):
                atm[ialt, iact].CO2[idx + ilay] = atm[ialt, iact].CO2[
                    idx + ilay
                ] + np.sum(microHH.co2[ialt, iact, midx[ilay]])
                atm[ialt, iact].NO[idx + ilay] = np.sum(
                    microHH.no[ialt, iact, midx[ilay]]
                )
                atm[ialt, iact].NO2[idx + ilay] = atm[ialt, iact].NO2[
                    idx + ilay
                ] + np.sum(microHH.no2[ialt, iact, midx[ilay]])
    return atm


class DataCont:
    """Empty container for data"""

    pass


def _get_default_files(dir_data):
    """Gets all the .nc files in the folder

    Parameters
    ----------
    dir_data : string
        Directory of the data

    Returns
    -------
    List of strings
        List of file names
    """
    default_files = []
    for file in os.listdir(dir_data):
        if "default" in file:
            default_files.append(file)
    default_files.sort()
    return default_files


def _read_density(dir_data, _files, time):
    """Read density at a given time

    Parameters
    ----------
    dir_data : String
        Directory of the data
    _files : List of Strings
        List of file names with extension .nc
    time : Int64
        Time of the simulation of MicroHH

    Returns
    -------
    rho : Array
        Density at the given time
    """
    for _fl in _files:
        ff = Dataset(dir_data + _fl, "r")
        # id of time
        ix = np.where(np.isclose(ff["time"][:].data, time))[0]
        if len(ix) > 0:
            # rho for that time
            rho = ff["thermo/rho"][ix[0], :].data
            ff.close()
            return rho
        else:
            ff.close()
            continue


def _read_datatime(dir_data, _files, time):
    """Read density at a given time

    Parameters
    ----------
    dir_data : String
        Directory of the data
    _files : List of Strings
        List of file names with extension .nc
    time : Int64
        Time of the simulation of MicroHH

    Returns
    -------
    rho : Array
        Density at the given time
    """
    for _fl in _files:
        ff = Dataset(dir_data + _fl, "r")
        # id of time
        ix = np.where(np.isclose(ff["time"][:].data, time))[0]
        if len(ix) > 0:
            datetime = num2date(ff["time"][ix[0]], ff["time"].units)
            # rho for that time
            ff.close()
            return datetime
        else:
            ff.close()
            continue


def _get_ini_file(dir_data):
    """get ini file

    Parameters
    ----------
    dir_data : String
        Directory of the data

    Returns
    -------
    String
        File name of ini file
    """
    # find the file with ini extension in the folder
    for file in os.listdir(dir_data):
        if file.endswith(".ini"):
            return dir_data + file


def _get_domain(flname_ini):
    """Gets the grid of the given domain

    Parameters
    ----------
    flname_ini : string
        File name of the ini file

    Returns
    -------
    grid : data class
        Data class containing the grid of the microHH

    """

    # Create a grid container
    grid = DataCont()
    config = configparser.ConfigParser()
    config.read(flname_ini)
    grid.__setattr__("nz", config.getint("grid", "ktot"))
    grid.__setattr__("ny", config.getint("grid", "jtot"))
    grid.__setattr__("nx", config.getint("grid", "itot"))
    grid.__setattr__("zsize", config.getfloat("grid", "zsize"))
    grid.__setattr__("ysize", config.getfloat("grid", "ysize"))
    grid.__setattr__("xsize", config.getfloat("grid", "xsize"))
    grid.__setattr__("dz", grid.zsize / grid.nz)
    grid.__setattr__("dy", grid.ysize / grid.ny)
    grid.__setattr__("dx", grid.xsize / grid.nx)
    grid.__setattr__("x_nodes", np.linspace(0, grid.xsize, grid.nx + 1))
    grid.__setattr__("y_nodes", np.linspace(0, grid.ysize, grid.ny + 1))
    grid.__setattr__("z_nodes", np.linspace(0, grid.zsize, grid.nz + 1))
    grid.__setattr__("xc", 0.5 * (grid.x_nodes[1:] + grid.x_nodes[:-1]))
    grid.__setattr__("yc", 0.5 * (grid.y_nodes[1:] + grid.y_nodes[:-1]))
    grid.__setattr__("zc", 0.5 * (grid.z_nodes[1:] + grid.z_nodes[:-1]))
    return grid


def _get_source_strength(flname_ini, prefix):
    """get source and strength of the source

    Parameters
    ----------
    flname_ini : string
        Ini file
    prefix : string
        Type of gas

    Returns
    -------
    source : Vector[3]
        Source location [x,y,z]
    strength : Float64
        Strength of the emission in kilomoles/s
    """
    config = configparser.ConfigParser()
    config.read(flname_ini)
    # get source section and the index of the source
    source_sec = config["source"]
    srcs = source_sec.get("sourcelist")
    idx = (srcs.split(",")).index(prefix)

    # get source locations
    xloc = float((source_sec["source_x0"]).split(",")[idx])
    yloc = float((source_sec["source_y0"]).split(",")[idx])
    zloc = float((source_sec["source_z0"]).split(",")[idx])
    source = [xloc, yloc, zloc]
    # get strength
    strength = float((source_sec["strength"]).split(",")[idx])
    return source, strength


def _read_var(dir_data, prefix, time):
    """Read the data based on prefix

    Parameters
    ----------
    dir_data : String
        directory of the data
    prefix : String
        Gas that needs to be read.
    time : Int
        Time of the simulation

    Returns
    -------
    bdata : 1d-Array of Float64
        Array containing the binary data
    """
    with open(dir_data + prefix + "." + str(time).zfill(7)) as f:
        rectype = np.dtype(np.float64)
        bdata = np.fromfile(f, dtype=rectype)
        f.close()
    return bdata


def _get_variable(prefix, dir_data, ini_filename, time, dim):
    """Get source, strength and data from prefix at a given time.

    Parameters
    ----------
    prefix : String
        Gas that needs to be read.
    dir_data : String
        directory of the data
    ini_filename : String
        Ini file
    time : Int64
        Time of simulation in seconds
    dim : Array [3] Int64
        [nz, ny, nx] dimensions to reshape the binary data

    Returns
    -------
    Data : DataCont Class
        Data container with data
    """
    tmp = DataCont()
    # Get source from the ini file
    source, strength = _get_source_strength(ini_filename, prefix)
    tmp.__setattr__("source", source)
    tmp.__setattr__("strength", strength)
    bdata = _read_var(dir_data, prefix, time)
    tmp.__setattr__("conc", bdata.reshape(dim))
    return tmp


def read_simulated_variable(
    dir_data,
    prefix,
    time,
):
    """read a gas variable of microHH

    Parameters
    ----------
    dir_data : String
        directory of the data
    prefix : List of Strings or a String
        Gases/gas that need to be read.
    time : Int64
        Time of the simulation
        Example: co2.0034000 has prefix as "co2" and time as 340000
    Returns
    -------
    Data : class
        Container containing different variables.
    """
    data = DataCont()
    # Get the domain from the ini file
    ini_filename = _get_ini_file(dir_data)
    data.__setattr__("grid", _get_domain(ini_filename))

    # Read data
    dim = np.array(
        [data.grid.nz, data.grid.ny, data.grid.nx]
    )  # dimensions of binary data
    # Check if prefix is a list or one variable
    if isinstance(prefix, list):
        # Get data for each prefix
        for each_prefix in prefix:
            data.__setattr__(
                each_prefix,
                _get_variable(each_prefix, dir_data, ini_filename, time, dim),
            )
    else:
        data.__setattr__(
            prefix, _get_variable(prefix, dir_data, ini_filename, time, dim)
        )

    # get default files
    default_files = _get_default_files(dir_data)
    # Get time of the simulation
    data.__setattr__("datetime", _read_datatime(dir_data, default_files, time))
    # read density : rho
    data.__setattr__("density", _read_density(dir_data, default_files, time))
    return data


def combine_varon_standard_atm(varon, atm_std):
    nalt = varon.y_range.size
    nact = varon.x_range.size

    # index of standard atmosphere that corresponds to the closest z layer in the varon z layers
    print("Combining varon and AFGL model")
    idx = (np.abs(atm_std.zlev - np.max(varon.z_lev))).argmin()

    if atm_std.zlev[idx] != np.max(varon.z_lev):
        sys.exit("vertical grids are not alligned in libATM.combine_varon_standard_atm")
    #    zlev_tmp = np.concatenate()
    zvaron = varon.z_lay
    zatm = atm_std.zlay
    nlay = zatm.size

    #   define a mask for vertical integration of varon data
    ztop = atm_std.zlev[idx:nlay]
    zbot = atm_std.zlev[idx + 1 : nlay + 1]

    midx = {}
    for ilay in range(ztop.size):
        midx[ilay] = np.where((zvaron < ztop[ilay]) & (zvaron >= zbot[ilay]))

    atm = np.ndarray((nalt, nact), np.object_)

    # for NOW we assume no contribution above upper model boundary of varon

    for ialt in tqdm(range(nalt)):
        for iact in range(nact):
            atm[ialt, iact] = deepcopy(atm_std)

            for ilay in range(ztop.size):
                atm[ialt, iact].CH4[idx + ilay] = atm[ialt, iact].CH4[
                    idx + ilay
                ] + np.sum(varon.ch4_afgl[ialt, iact, midx[ilay]])

    return atm
