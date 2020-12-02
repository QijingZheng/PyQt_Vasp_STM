#!/usr/bin/env python3
################################################################################
import sys, os
import numpy as np
import argparse
import logging

import matplotlib as mpl
from matplotlib.figure import Figure

from ase.calculators.vasp import VaspChargeDensity

################################################################################

LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
logger = logging.getLogger(__name__)
logging.basicConfig(level=LOGLEVEL)

mpl.rcParams['axes.linewidth'] = 0.0

FNAME = None

################################################################################
class vaspParchg(VaspChargeDensity):
    def __init__(self, inputFile='PARCHG'):
        super(vaspParchg, self).__init__(inputFile)

        self.chg_n = np.asarray(self.chg)[0]
        self.poscar= self.atoms[0]

        self.ngrid = self.chg_n.shape
        self.abc   = np.linalg.norm(self.poscar.cell, axis=1)

        self.NX, self.NY, self.NZ = self.ngrid
        self.a, self.b, self.c    = self.abc

        self.zmax_pos = self.poscar.positions[:,-1].max()
        self.zmax_ind = int(self.zmax_pos / self.c * self.NZ) + 1
        self.fname = inputFile

    def isoHeight(self, zcut):
        """ Return the iso-height cut the charge density """

        img = self.chg_n[:,:,zcut]

        # return np.tile(img, repeat).T
        return img

    def isoCurrent(self, zcut, pc=None, ext=0.15):
        """ Return the iso-current cut the charge density """

        zext = int(self.NZ * ext)
        zcut_min = zcut - zext
        if zcut_min < self.zmax_ind:
            zcut_min = self.zmax_ind
        zcut_max = zcut + zext

        if pc is None:
            c = np.average(self.chg_n[:,:, zcut])
        else:
            tmp = np.zeros(zcut_max - zcut_min)
            for ii in range(tmp.size):
                tmp[ii] = np.average(self.chg_n[:,:,zcut_min + ii])
            c = np.linspace(tmp.min(), tmp.max(), 100)[pc]

        # height of iso-current 
        img = np.argmin(np.abs(self.chg_n[:,:,zcut_min:zcut_max] - c), axis=2)

        return img
        # img_ext =  np.tile(img, repeat)
        # return img_ext.T


    def plotIsoHeightImg(self, zcut, repeat=[1,1], cmap='Greys_r'):
        dat = self.isoHeight(zcut)
        NX = self.NX
        NY = self.NY
        a = (self.poscar.cell[0,0] + \
             self.poscar.cell[0,1] * 1j) / NX
        b = (self.poscar.cell[1,0] + \
             self.poscar.cell[1,1] * 1j) / NY
        gx, gy = np.mgrid[0:NX*repeat[0], 0:NY*repeat[1]]
        tmp = gx * a + gy * b
        STMXCoord = np.real(tmp)
        STMYCoord = np.imag(tmp)
        STMData   = np.tile(dat, (repeat[0], repeat[1]))

        fig = Figure((6.0, 6.0), dpi=600)
        fig.subplots_adjust(left=0.05, right=0.95,\
                                 bottom=0.05, top=0.95)
        axes = fig.add_subplot(111)
        axes.set_aspect('equal')
        axes.axis('off')

        xmin, xmax = STMXCoord.min(), STMXCoord.max()
        ymin, ymax = STMYCoord.min(), STMYCoord.max()
        axes.set_xlim(xmin, xmax)
        axes.set_ylim(ymin, ymax)

        axes.pcolormesh(STMXCoord, STMYCoord, STMData,
                        cmap=cmap,
                        shading='gouraud')

        fig.savefig('isoH_{}_{}.png'.format(self.fname, zcut))


def parseArgs():
    parser = argparse.ArgumentParser(description="A command line tool to plot iso-height STM image")
    parser.add_argument('fname',
                        type=str,
                        help='file names of PARCHG',
                        nargs='+')
    parser.add_argument('-z',
                        type=int,
                        help='tip height, unit: grid index',
                        action='store',
                        required=True)
    parser.add_argument('-r', '--repeat',
                        type=list,
                        help='count repeat image in a and b direction',
                        action='store',
                        nargs=2,
                        default=[1, 1])
    parser.add_argument('-c', '--cmap',
                        type=str,
                        help='colormap to use, see https://matplotlib.org/tutorials/colors/colormaps.html to choose which you favor, default Greys_r',
                        default='Greys_r')
    return parser.parse_args()


if '__main__' == __name__:
    parser = parseArgs()
    logger.debug(f'fname = {parser.fname}')
    logger.debug(f'z = {parser.z}')
    logger.debug(f'repeat = {parser.repeat}')
    logger.debug(f'colormap = {parser.cmap}')

    for fn in parser.fname:
        logger.info("Parsing {} ...".format(fn))
        pchg = vaspParchg(fn)
        logger.info("Ploting ...")
        pchg.plotIsoHeightImg(parser.z, repeat=parser.repeat, cmap=parser.cmap)
        pass
    pass
