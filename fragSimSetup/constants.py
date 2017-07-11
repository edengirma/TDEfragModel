from astropy import units as u
from math import sqrt


class sim_constants():
    Msun = 2.0e33
    G = 6.67e-8
    AU = 1.5e13
    Rsun = 7e10

    m_hole = (4.0e6 * Msun)
    m_bulge = (3.76e9 * Msun)
    m_disk = (6.0e10 * Msun)
    m_halo = (1.0e12 * Msun)

    # All distances initialized in AU:
    r_halo = (4.125e9 * AU)

    # Bulge and disk parameters
    ab = (2.0e7 * AU)
    ad = (5.7e8 * AU)
    bd = (6.2e7 * AU)

    # Nuclear cluster parameters
    rc = (1.0 * u.pc).to('cm').value

    # Boltzmann constant, m_proton, mu
    kb = 1.4e-16
    mp = 1.7e-24
    mu = 0.5
    gamma = 5.0/3.0
    cs = sqrt((gamma * kb * 5.0e3)/(mu*mp))
