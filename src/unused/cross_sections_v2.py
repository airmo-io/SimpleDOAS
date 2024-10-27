
import numpy as np
import math
import time


MAXLINES = 100000
NVALUES = 5
PI = 3.1415926535897932385
DELTAKAPPA = 0.0603 / 2
KAPPALO = 5882
KAPPAHI = 6666
path_hitran='/Users/queissman/AIRMO/DATA/02_hit08_CO2.par'

def compute_abs_cross_section():
    # Placeholder for VAL array
    VAL = np.zeros((MAXLINES, NVALUES), dtype=np.float64)

    def ILNZ(K, K0, G):
        """Returns the indefinite integral of the Lorentz line shape function."""
        return math.atan2(K - K0, G) / PI

    # Read input file (HiTran data)
    with open(path_hitran, 'r') as infile:
        NLINES = 0
        for line in infile:
            if NLINES >= MAXLINES:
                break
            parts = line #.split()
            print(line)
            #ia, ib = int(parts[0]), int(parts[1])  # Unused variables
            VAL[NLINES, :] = [float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5]), float(parts[6])]
            print(VAL)
            NLINES += 1
            #else:
            #    print("Not enough columns in HITRAN file")

    # Open output file
    with open('/Users/queissman/AIRMO/DATA/sigma_abs.csv', 'w') as outfile:
        start_time = time.time()  # Optional timer

        # Calculate cross sections, kappa is wave number
        for KAPPA in np.arange(KAPPALO + DELTAKAPPA, KAPPAHI + DELTAKAPPA, 2 * DELTAKAPPA):
            S = 0.0
            for LINE in range(NLINES):
                V1 = ILNZ(KAPPA + DELTAKAPPA, VAL[LINE, 0], VAL[LINE, 3])
                V2 = ILNZ(KAPPA - DELTAKAPPA, VAL[LINE, 0], VAL[LINE, 3])
                S += VAL[LINE, 1] * (V1 - V2)

            S /= (2 * DELTAKAPPA)

            # Write to output files (CSV format)
            outfile.write(f"{KAPPA:.6E},{S:.6E}\n")
            print(f"{KAPPA:.6E},{S:.6E}")  # Optional console output

        end_time = time.time()  # Optional timer
        #print(f"{NLINES} Lines processed, took {end_time - start_time:.2f} seconds")

