# collection of numerical tools
import numpy as np
import scipy


def get_isrf(isrf_parameter, wave, wave_meas):
    dmeas = wave_meas.size
    dlbl = wave.size
    isrfct = np.zeros(shape=(dmeas, dlbl))

    if isrf_parameter["type"] == "Gaussian":
        fwhm = isrf_parameter["fwhm"]
        const = fwhm**2 / (4 * np.log(2))

        for l, wmeas in enumerate(wave_meas):
            wdiff = wave - wmeas
            mask1 = wdiff >= -1.5 * fwhm
            mask2 = wdiff <= 1.5 * fwhm
            idx = np.logical_and(mask1, mask2)
            isrfct[l, idx] = np.exp(-wdiff[idx] ** 2 / const)
            isrfct[l, :] = isrfct[l, :] / np.sum(isrfct[l, :])

    mask = isrfct > 0.0

    return isrfct, mask


class isrfct:
    def __init__(self, wave_target, wave_input):
        self.wave_target = wave_target
        self.wave_input = wave_input
        self.isrf = {}

    def get_isrf(self, parameter):
        nwave_target = self.wave_target.size
        nwave_input = self.wave_input.size
        self.isrf["isrf"] = np.zeros((nwave_target, nwave_input))

        if parameter["type"] == "Gaussian":
            const = parameter["fwhm"] ** 2 / (4 * np.log(2))
            istart = []
            iend = []
            for l, wmeas in enumerate(self.wave_target):
                wdiff = self.wave_input - wmeas
                istart.append(np.argmin(np.abs(wdiff + 1.5 * parameter["fwhm"])))
                iend.append(np.argmin(np.abs(wdiff - 1.5 * parameter["fwhm"])))

                self.isrf["isrf"][l, istart[l] : iend[l]] = np.exp(
                    -wdiff[istart[l] : iend[l]] ** 2 / const
                )
                self.isrf["isrf"][l, :] = self.isrf["isrf"][l, :] / np.sum(
                    self.isrf["isrf"][l, :]
                )

            self.isrf["istart"] = istart
            self.isrf["iend"] = iend

    def isrf_convolution(self, spectrum):
        nwave_target = self.wave_target.size
        spectrum_conv = np.empty(nwave_target)

        for iwav in range(nwave_target):
            istart = self.isrf["istart"][iwav]
            iend = self.isrf["iend"][iwav]
            spectrum_conv[iwav] = self.isrf["isrf"][iwav, istart:iend].dot(
                spectrum[istart:iend]
            )
        return spectrum_conv


def gaussian_kernel_2D(
    size: int, fwhm_x: float, fwhm_y: float, center: float = None
) -> np.array:
    """make a gaussian kernel

    Args:
        size (int): kernel size (extent in each dimension)
        fwhm_x (float): the x width in step units
        fwhm_y (float): the y  width in step units
        center (float, optional): the center position. Defaults to None (auto, center).

    Returns:
        np.array: the kernel array *not normalised*
    """
    # Make a 2 dimensional gaussian kernel as a product of two Gaussian
    # fwhm are given in units of pixel samples

    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    kernel = np.exp(-((x - x0) ** 2) / fwhm_x**2 - ((y - y0) ** 2) / fwhm_y**2)
    return kernel / kernel.sum()


def Gaussian2D_deprecated(size, fwhm_x, fwhm_y, center=None):
    """DEPRECATED SRON FUNCTION"""
    # Make a 2 dimensional gaussian kernel as a product of two Gaussian
    # fwhm are given in units of pixel samples

    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4 * np.log(2) * ((x - x0) ** 2) / fwhm_x**2) * np.exp(
        -4 * np.log(2) * ((y - y0) ** 2) / fwhm_y**2
    )


def convolution_2d_deprecated(data, settings):
    """DEPRECATED SRON FUNCTION"""
    # convolve the data array with a kernel defined in settings

    if settings["type"] == "2D Gaussian":
        kernel = Gaussian2D_deprecated(
            settings["1D kernel extension"], settings["fwhm x"], settings["fwhm y"]
        )

    data_conv = scipy.signal.convolve(data, kernel, mode="same")

    return data_conv


def convolution_2d(data: np.array, settings: dict) -> np.array:
    """convolve the data array with a kernel defined in settings

    Args:
        data (np.array): the input data
        settings (dict): the convolution settings

    Returns:
        np.array: the convoluted array
    """

    if settings["type"] == "2D Gaussian":
        kernel = gaussian_kernel_2D(
            settings["1D kernel extension"], settings["fwhm x"], settings["fwhm y"]
        )

    kernel = kernel / kernel.sum()

    data_conv = scipy.signal.convolve(data, kernel, mode="same")

    return data_conv


def print_attributes(class_object):
    attributes = [attr for attr in dir(class_object) if not attr.startswith("__")]
    print(attributes)
    return


def find_nearest(a, a0):
    "Element in nd array `a` closest to the scalar value `a0`"
    idx = np.abs(a - a0).argmin()
    return a.flat[idx], idx


def xy2area(xy_2d):
    """Calculate the area between consecutive coordinates in a 2d plane.
    Input must be 2d, otherwise area is undefined.
    Returned area is padded.
    This accounts for the distortion caused by the viewing angle, centered around the center of the 2d input.
    Args:
        xy_2d (np.array[np.array[x,y]]): 2d array of x,y coordinates
    Returns:
        area (np.array[np.array]): 2d array of associated area, same size as input, padded with edge value
    """

    def area_between_points(x1, y1, x2, y2, x3, y3, x4, y4):
        area = abs(
            x1 * y2
            - x2 * y1
            + x2 * y3
            - x3 * y2
            + x3 * y4
            - x4 * y3
            + x4 * y1
            - x1 * y4
        )
        area = area * 0.5
        return area

    dim_x, dim_y, _ = np.shape(xy_2d)
    if dim_x < 2 or dim_y < 2:
        raise ValueError("Cannot calculate area for data with fewer than 2 entries!")
    # Each area element needs four points to calculate -> the final array will be smaller than input by 1
    dim_x_area = dim_x - 1
    dim_y_area = dim_y - 1
    area = np.zeros((dim_x_area, dim_y_area))

    x = xy_2d[:, :, 0]
    y = xy_2d[:, :, 1]

    area[:, :] = area_between_points(
        x[:-1, :-1],
        y[:-1, :-1],
        x[1:, :-1],
        y[1:, :-1],
        x[1:, 1:],
        y[1:, 1:],
        x[:-1, 1:],
        y[:-1, 1:],
    )

    # TODO This associates the area of a pixel with its left and lowermost coordinate
    # Padding is done with edge value to obtain original size.
    # Other padding and association strategies to be done, impact minor.

    pad_width = ((0, 1), (0, 1))
    area = np.pad(area, pad_width, mode="edge")
    return area


def lat2d_lon2d_to_area(lat2d: np.ndarray, lon2d: np.ndarray) -> np.ndarray:
    """From a 2d input of lat and lon coordinates, calculates the area of each associated pixel.
    Input must be 2d, otherwise area is undefined.
    Returned area is padded.
    This accounts for the distortion caused by the viewing angle, centered around the center of the 2d input.
    Args:
        lat2d (np.ndarray): 2d array of latitudes
        lon2d (np.ndarray): 2d array of longitudes
    Returns:
        np.ndarray: 2d array of associated area, same size as input, padded with edge value
    """
    ref_latlon = [
        0,
        0,
    ]  # [deg] irrelevant position, as area is calculated based on diffs
    transform = TransformCoords(ref_latlon)
    x_mts, y_mts = transform.latlon2xymts(lat2d, lon2d)
    xy_mts = np.stack((x_mts, y_mts), axis=-1)
    area = xy2area(xy_mts)
    return area


# =========================================================================


class TransformCoords:
    """A class to transform coordinates"""

    def __init__(self, origin):
        """Initialize class based on the Origin

        Parameters
        ----------
        origin : [lat, lon] Float64
            Origin in lat-lon coordinates
        """
        self.rd = np.pi / 180.0
        phi0 = origin[0] * self.rd
        self.ld0 = origin[1] * self.rd
        self.s_p0 = np.sin(phi0)
        self.c_p0 = np.cos(phi0)
        self.fact = 6371.0
        self.factmts = self.fact * 1000

    def latlon2xykm(self, lat, lon):
        """latlon2xykm Convert from lat lon to xy in km

        Parameters
        ----------
        lat : Matrix/Vector/Float64
            Latitude
        lon : Matrix/Vector/Float64
            Longitude

        Returns
        -------
        x  : Matrix/Vector/Float64
            x in km
        y  : Matrix/Vector/Float64
            y in km
        """
        ld = lon * self.rd
        phi = lat * self.rd
        s_p = np.sin(phi)
        c_p = np.cos(phi)
        ll = ld - self.ld0
        c_l = np.cos(ll)
        s_l = np.sin(ll)
        c_pl = c_p * c_l
        w = np.sqrt(
            2.0 / (np.maximum(1.0 + self.s_p0 * s_p + self.c_p0 * c_pl, 1.0e-10))
        )
        x = c_p * s_l * w
        y = (self.c_p0 * s_p - self.s_p0 * c_pl) * w
        return x * self.fact, y * self.fact

    def latlon2xymts(self, lat, lon):
        """latlon2xymts Convert from lat lon to xy in km

        Parameters
        ----------
        lat : Matrix/Vector/Float64
            Latitude
        lon : Matrix/Vector/Float64
            Longitude

        Returns
        -------
        x  : Matrix/Vector/Float64
            x in mts
        y  : Matrix/Vector/Float64
            y in mts
        """
        ld = lon * self.rd
        phi = lat * self.rd
        s_p = np.sin(phi)
        c_p = np.cos(phi)
        ll = ld - self.ld0
        c_l = np.cos(ll)
        s_l = np.sin(ll)
        c_pl = c_p * c_l
        w = np.sqrt(
            2.0 / (np.maximum(1.0 + self.s_p0 * s_p + self.c_p0 * c_pl, 1.0e-10))
        )
        x = c_p * s_l * w
        y = (self.c_p0 * s_p - self.s_p0 * c_pl) * w
        return x * self.factmts, y * self.factmts

    def xykm2latlon(self, x1, y1):
        """xykm2latlon Convert from x, y to lat-lon

        Parameters
        ----------
        x1 : Matrix/Vector/Float64
            x coordinate in km
        y1 : Matrix/Vector/Float64
            y coordinate in km

        Returns
        -------
        lat  : Matrix/Vector/Float64
            latitude
        lon  : Matrix/Vector/Float64
            Longitude
        """
        x, y = x1 / self.fact, y1 / self.fact
        p = np.maximum(np.sqrt(x**2 + y**2), 1.0e-10)
        c = 2.0 * np.arcsin(p / 2.0)
        s_c = np.sin(c)
        c_c = np.cos(c)
        phi = np.arcsin(c_c * self.s_p0 + y * s_c * self.c_p0 / p)
        ld = self.ld0 + np.arctan2(x * s_c, (p * self.c_p0 * c_c - y * self.s_p0 * s_c))
        lat = phi / self.rd
        lon = ld / self.rd
        if isinstance(lat, np.ndarray):
            lat[lat > 90.0] -= 180.0
            lat[lat < -90.0] += 180.0
            lon[lon > 180.0] -= 360.0
            lon[lon < -180.0] += 360.0
        else:
            if abs(lat) > 90.0:
                if lat > 0:
                    lat = lat - 180.0
                else:
                    lat = lat + 180.0
            if abs(lon) > 180.0:
                if lon > 0:
                    lon = lon - 360.0
                else:
                    lon = lon + 360.0
        return lat, lon

    def xymts2latlon(self, x1, y1):
        """xymts2latlon Convert from x, y to lat-lon

        Parameters
        ----------
        x1 : Matrix/Vector/Float64
            x coordinate in m
        y1 : Matrix/Vector/Float64
            y coordinate in m

        Returns
        -------
        lat  : Matrix/Vector/Float64
            latitude
        lon  : Matrix/Vector/Float64
            Longitude
        """
        x, y = x1 / self.factmts, y1 / self.factmts
        p = np.maximum(np.sqrt(x**2 + y**2), 1.0e-10)
        c = 2.0 * np.arcsin(p / 2.0)
        s_c = np.sin(c)
        c_c = np.cos(c)
        phi = np.arcsin(c_c * self.s_p0 + y * s_c * self.c_p0 / p)
        ld = self.ld0 + np.arctan2(x * s_c, (p * self.c_p0 * c_c - y * self.s_p0 * s_c))
        lat = phi / self.rd
        lon = ld / self.rd
        if isinstance(lat, np.ndarray):
            lat[lat > 90.0] -= 180.0
            lat[lat < -90.0] += 180.0
            lon[lon > 180.0] -= 360.0
            lon[lon < -180.0] += 360.0
        else:
            if abs(lat) > 90.0:
                if lat > 0:
                    lat = lat - 180.0
                else:
                    lat = lat + 180.0
            if abs(lon) > 180.0:
                if lon > 0:
                    lon = lon - 360.0
                else:
                    lon = lon + 360.0
        return lat, lon
