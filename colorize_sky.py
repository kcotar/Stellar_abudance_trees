import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

def _prepare_ra_dec(data):
    ra = data['ra']
    idx_trans = ra > 180
    if len(idx_trans) > 0:
        ra[idx_trans] -= 360
    ra = np.deg2rad(ra)
    dec = np.deg2rad(data['dec'])
    return ra, dec


def plot_ra_dec_locations(data, path='sky_pos.png'):
    # plt.subplot(111, projection='mollweide')
    # ra, dec = _prepare_ra_dec(data)
    # plt.scatter(ra, dec, lw=0, c='black', s=0.4)
    # plt.grid(True)
    # plt.colorbar()
    # plt.tight_layout()
    # plt.savefig(path, dpi=500)
    # plt.close()
    plt.figure()
    map = Basemap(projection='moll', lon_0=0)
    map.drawparallels(np.arange(-90., 95., 5.))
    map.drawmeridians(np.arange(0., 365., 5.))
    ra, dec = _prepare_ra_dec(data)
    map.scatter(ra, dec, lw=0, c='black', s=0.4)
    ax = plt.gca()
    ax.set_xlim((np.min(ra), np.max(ra)))
    ax.set_ylim((np.min(dec), np.max(dec)))
    plt.tight_layout()
    plt.savefig(path, dpi=250)
    plt.close()


def plot_ra_dec_attribute(data, attribute, path='sky_pos_attribute.png'):
    # plt.subplot(111, projection='mollweide')
    # ra, dec = _prepare_ra_dec(data)
    # plt.scatter(ra, dec, lw=0, c=data[attribute], s=0.4)
    # plt.grid(True)
    # plt.colorbar()
    # plt.tight_layout()
    # plt.show()
    # plt.savefig(path, dpi=500)
    # plt.close()
    plt.figure()
    map = Basemap(projection='moll', lon_0=0)
    map.drawparallels(np.arange(-90., 95., 5.))
    map.drawmeridians(np.arange(0., 365., 5.))
    ra, dec = _prepare_ra_dec(data)
    map.scatter(ra, dec, lw=0, c=data[attribute], s=2)
    ax = plt.gca()
    ax.set_xlim((np.min(ra), np.max(ra)))
    ax.set_ylim((np.min(dec), np.max(dec)))
    plt.tight_layout()
    plt.savefig(path, dpi=250)
    plt.close()