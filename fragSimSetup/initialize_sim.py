# Standard python modules
import json
import random as rnd
from math import cos, sin, sqrt

# These modules need to be pip installed.
import numpy as np
# For Jupyter/IPython notebook
from tqdm import tnrange

# From fragSimSetup
from .dMdEdist import dMdEdist
from .constants import sim_constants as sc
from .funcs import Cfunc, beta_dist, mstar_dist, r_tidal, rstar_func


class TDEsim:

    def __init__(self, Nstars):
        self.Nstars = Nstars
        self.dmde = dMdEdist()
        self.forces = []
        self.data = []

    def initialize(self):
        m_hole = sc.m_hole
        star_masses = []
        star_radii = []
        tidal_radii = []

        for star in tnrange(self.Nstars, desc='Star', leave=False):
            self.data.append({})
            # Randomly drawn mass of star
            xstar = rnd.random()
            m_star = mstar_dist(xstar) * sc.Msun
            star_masses.append(m_star)

            # Determined radius of star
            r_star = (rstar_func(m_star / sc.Msun) * sc.Rsun)
            star_radii.append(r_star)

            # Determined tidal radius of star
            r_t = r_tidal(m_star, r_star)
            tidal_radii.append(r_t)

            # Randomly draw beta
            xbeta = rnd.random()
            beta = beta_dist(xbeta)

            # Calculate periapse radius
            r_p = r_t / beta

            # Set position of star; random sphere point picking
            u1 = rnd.uniform(-1.0, 1.0)
            th1 = rnd.uniform(0., 2. * np.pi)
            star_direc = np.array([sqrt(1.0 - (u1)**2) * cos(th1),
                                   sqrt(1.0 - (u1)**2) * sin(th1),
                                   u1])
            star_vec = [r_p * d for d in star_direc]

            # Core temperature of star, expansion factor
            Tc = (((sc.G * m_star**2) /
                   ((2 * r_star) * 4 * np.pi * r_star**3)) /
                  ((m_star * sc.kb) / (4.0 / 3.0 * np.pi *
                                       r_star**3 * sc.mu * sc.mp)))
            expf = (Tc / (5.0e3))**(3.0 / 2.0)

            # Final number density, mass of fragments, number of fragments
            nfinal = ((0.5 * m_star * Cfunc(beta)) /
                      (sc.mu * sc.mp * expf * 4.0 / 3.0 * np.pi * r_star**3))
            m_frag = (2.0 * sc.Msun * (sc.cs / (2.0e4))**3 *
                      (nfinal / 1.0e3)**(-0.5))
            Nfrag = int((0.5 * m_star * Cfunc(beta)) / (m_frag))

            while(Nfrag == 0):
                # Randomly draw beta
                xbeta = rnd.random()
                beta = beta_dist(xbeta)

                # Core temperature of star, expansion factor
                Tc = (((sc.G * m_star**2) /
                       ((2 * r_star) * 4 * np.pi * r_star**3)) /
                      ((m_star * sc.kb) / (4.0 / 3.0 * np.pi *
                                           r_star**3 * sc.mu * sc.mp)))
                expf = (Tc / (5.0e3))**(3.0 / 2.0)

                # Final number density, mass of fragments, number of fragments
                nfinal = ((0.5 * m_star * Cfunc(beta)) /
                          (sc.mu * sc.mp * expf * 4.0 / 3.0 *
                           np.pi * r_star**3))
                m_frag = (2.0 * sc.Msun * (sc.cs / (2.0e4))**3 *
                          (nfinal / 1.0e3)**(-0.5))
                Nfrag = int((0.5 * m_star * Cfunc(beta)) / (m_frag))

            # Calculate binding energy spread
            NRGs = self.dmde.energy_spread(beta, Nfrag)

            # Converted NRGs list from cgs to proper units
            nrg_scale = ((r_star / sc.Rsun)**(-1.0) *
                         (m_star / sc.Msun)**(2.0 / 3.0) *
                         (m_hole / sc.Msun / 1.0e6)**(1.0 / 3.0))
            energies = [(nrg_scale * nrg) for nrg in NRGs]

            # Calculating velocities
            vels = [sqrt((2.0 * g) + (2 * sc.G * m_hole / r_p))
                    for g in energies]

            # Randomly draw velocity vector direction
            phi2 = rnd.uniform(0., 2. * np.pi)

            x = star_vec[0]
            y = star_vec[1]
            z = star_vec[2]
            r = np.linalg.norm(star_vec)

            randomvelvec = [
                (x * (r - z + z * cos(phi2)) - r * y * sin(phi2)) /
                (r**2 * sqrt(2.0 - 2.0 * z / r)),
                (y * (r - z + z * cos(phi2)) + r * x * sin(phi2)) /
                (r**2 * sqrt(2.0 - 2.0 * z / r)),
                ((r - z) * z - (x**2 + y**2) * cos(phi2)) /
                (r**2 * sqrt(2.0 - 2.0 * z / r))
            ]

            velocity_vec = np.cross(star_vec, randomvelvec)
            n = np.linalg.norm(velocity_vec)

            fragx = []
            fragy = []
            fragz = []
            fragvx = []
            fragvy = []
            fragvz = []

            # Distance spread for fragments
            rads = [r_star * float(f) / float(Nfrag + 1)
                    for f in range(Nfrag + 1)]
            rads.pop(0)

            for fi, frag in enumerate(tnrange(Nfrag,
                                              desc='Fragment', leave=False)):

                # Velocity vector of fragment
                vel = vels[frag]
                frag_velvec = [vel * v / n for v in velocity_vec]
                # Position vector of Fragment
                rad = rads[frag]
                frag_posvec = [(r_p + rad) * p for p in star_direc]

                fragx.append(frag_posvec[0])
                fragy.append(frag_posvec[1])
                fragz.append(frag_posvec[2])
                fragvx.append(frag_velvec[0])
                fragvy.append(frag_velvec[1])
                fragvz.append(frag_velvec[2])

            self.data[star]['x'] = fragx
            self.data[star]['y'] = fragy
            self.data[star]['z'] = fragz
            self.data[star]['vx'] = fragvx
            self.data[star]['vy'] = fragvy
            self.data[star]['vz'] = fragvz
            self.data[star]['mfrag'] = m_frag

        # Creating JSON file
        with open('data1000.json', 'w') as fp:
            json.dump(self.data, fp)
