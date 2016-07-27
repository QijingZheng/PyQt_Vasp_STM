#!/usr/bin/env python

################################################################################
import sys, os
import numpy as np

from PyQt4.QtGui import *
from PyQt4.QtCore import *

import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar

from ase.calculators.vasp import VaspChargeDensity

################################################################################
mpl.rcParams['axes.linewidth'] = 0.0

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

    def isoHeight(self, zcut, repeat=(1, 1)):
        """ Return the iso-height cut the charge density """

        img = self.chg_n[:,:,zcut]

        return np.tile(img, repeat).T

    def isoCurrent(self, zcut, repeat=(1, 1), pc=None, ext=0.15):
        """ Return the iso-current cut the charge density """

        zext = int(self.NZ * ext)
        zcut_min = zcut - zext
        if zcut_min < self.zmax_ind:
            zcut_min = self.zmax_ind
        zcut_max = zcut + zext

        if pc is None:
            c = np.average(self.chg_n[:,:, zcut])
            print "kaka"
        else:
            tmp = np.zeros(zcut_max - zcut_min)
            for ii in range(tmp.size):
                tmp[ii] = np.average(self.chg_n[:,:,zcut_min + ii])
            c = np.linspace(tmp.min(), tmp.max(), 100)[pc]

        # height of iso-current 
        img = np.argmin(np.abs(self.chg_n[:,:,zcut_min:zcut_max] - c), axis=2)

        img_ext =  np.tile(img, repeat)
        return img_ext.T

################################################################################
class Form(QMainWindow):
    def __init__(self, parent=None):
        super(Form, self).__init__(parent)

        self.filename = None
        self.VaspPchg = None
        self.zcut = None
        self.pc = None
        self.cmap = 'hot'
        self.dpi = 300
        self.dat = None
        self.repeat_x = 1
        self.repeat_y = 1

        # 0 for IsoHeight and 1 for IsoCurrent STM image
        self.whicISO = 0

        # self.create_menu()
        self.create_main_frame()
        self.create_status_bar()
        
        self.on_show()

        self.setWindowTitle('PyQt & matplotlib demo: STM Simulation')
        # self.resize(700, 420)

    def load_file(self):
        self.filename = QFileDialog.getOpenFileName(self,
            'Open a data file', '.', '')
        
        if self.filename:
            self.VaspPchg = vaspParchg(self.filename)
            self.zcut = self.VaspPchg.zmax_ind
            self.cutSpinBox.setValue(self.zcut)
            self.on_show()
            for irow in range(3):
                self.InfoLabs[irow][0].setText(u'%d' % self.VaspPchg.ngrid[irow])
                self.InfoLabs[irow][1].setText(u'%.2f \u212B' % self.VaspPchg.abc[irow])

    def save_img(self):
        self.imgName = QFileDialog.getSaveFileName(self,
                       'Save Image:', './untitled.png', '')
        if self.imgName:
            self.fig.savefig(str(self.imgName), dpi=self.dpi)

    def save_dat(self):
        self.datName = QFileDialog.getSaveFileName(self,
                       'Save Data:', './untitled.dat', '')
        if self.datName and self.dat:
            np.savetxt(str(self.datName), self.dat)
    
    def on_show(self):
        self.axes.clear()        
        self.axes.set_xticks([])
        self.axes.set_yticks([])

        if self.VaspPchg:
            zcut = self.zcut

            # isoHeight STM image
            if self.whicISO == 0:
                self.dat = self.VaspPchg.isoHeight(zcut,
                                              repeat=(self.repeat_x, self.repeat_y))
            # isoCurrent STM image
            else:
                self.dat = self.VaspPchg.isoCurrent(zcut, pc=self.pc,
                                               repeat=(self.repeat_x, self.repeat_y))

            self.axes.imshow(self.dat, # extent=exts,
                             origin='lower',
                             cmap=self.cmap,
                             interpolation='bicubic')

        self.canvas.draw()

    def zcutChanged(self, zcut):
        self.zcut = zcut
        self.pc = None
        if self.VaspPchg:
            txt = u'Zm = %.2f \u212B;   Zn =' % self.VaspPchg.zmax_pos
            self.zcutToAngLabel.setText(txt + u'<span style="color: #FF0000; font-weight: bold;"> %.2f &#8491;</span>' %
                                        (float(self.zcut) / self.VaspPchg.NZ * self.VaspPchg.c))
            self.on_show()

    def pcChanged(self, pc):
        self.pc = pc
        if self.whicISO == 1:
            self.on_show()

    def isoChanged(self, ii):
        self.whicISO = ii
        self.on_show()

    def createLeftPart(self):
        self.fig = Figure((4.5, 4.5), dpi=self.dpi)
        self.fig.subplots_adjust(left=0.05, right=0.95,
                                 bottom=0.05, top=0.95)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.main_frame)
        
        self.axes = self.fig.add_subplot(111)
        self.mpl_toolbar = NavigationToolbar(self.canvas, self.main_frame)
        
        self.left_vbox = QVBoxLayout()
        self.left_vbox.addWidget(self.canvas)
        self.left_vbox.addWidget(self.mpl_toolbar)
    
    def createRightPart(self):

        self.loadButton = QPushButton("&Load")
        self.loadButton.clicked.connect(self.load_file)
        self.saveImgButton = QPushButton("Save &Img")
        self.saveImgButton.clicked.connect(self.save_img)
        self.saveDatButton = QPushButton("Save &Dat")
        self.saveDatButton.clicked.connect(self.save_dat)
        self.cutButton = QPushButton("&Apply")
        self.cutButton.clicked.connect(self.on_show)

        self.isoComboBox = QComboBox()
        self.isoComboBox.addItem('Iso Height STM')
        self.isoComboBox.addItem('Iso Current STM')
        self.isoComboBox.currentIndexChanged.connect(self.isoChanged)

        self.cutLabel = QLabel("ZCUT =")
        self.cutSpinBox = QSpinBox()
        self.cutSpinBox.setRange(0, 1500)
        self.cutSpinBox.setSingleStep(1)
        self.cutSpinBox.valueChanged.connect(self.zcutChanged)
        self.zcutToAngLabel = QLabel()
        self.zcutToAngLabel.setMaximumHeight(25)
        self.zcutToAngLabel.setAlignment(Qt.AlignRight|Qt.AlignVCenter)
        # self.zcutToAngLabel.setFrameStyle(QFrame.StyledPanel|QFrame.Sunken)
        
        self.pcLabel = QLabel("Percentage =")
        self.pcSpinBox = QSpinBox()
        self.pcSpinBox.setRange(0, 99)
        self.pcSpinBox.setSingleStep(1)
        self.pcSpinBox.valueChanged.connect(self.pcChanged)

        repeatXLabel = QLabel('Repeat X')
        repeatYLabel = QLabel('Repeat Y')
        repeatXSpinBox = QSpinBox()
        repeatYSpinBox = QSpinBox()
        repeatXSpinBox.setRange(1, 10)
        repeatYSpinBox.setRange(1, 10)
        repeatXSpinBox.valueChanged.connect(self.rx_changed) 
        repeatYSpinBox.valueChanged.connect(self.ry_changed) 

        for lab in [repeatXLabel, repeatYLabel, self.cutLabel, self.pcLabel]:
            lab.setAlignment(Qt.AlignRight|Qt.AlignVCenter)

        self.right_vbox = QGridLayout()
        self.right_vbox.addWidget(self.isoComboBox, 0, 0, 1, 2)
        self.right_vbox.addWidget(repeatXLabel, 1, 0)
        self.right_vbox.addWidget(repeatYLabel, 2, 0)
        self.right_vbox.addWidget(repeatXSpinBox, 1, 1)
        self.right_vbox.addWidget(repeatYSpinBox, 2, 1)
        self.right_vbox.addWidget(self.cutLabel, 3, 0)
        self.right_vbox.addWidget(self.cutSpinBox, 3, 1)
        self.right_vbox.addWidget(self.pcLabel, 4, 0)
        self.right_vbox.addWidget(self.pcSpinBox, 4, 1)
        self.right_vbox.addWidget(self.zcutToAngLabel, 5, 0, 1, 2)
        self.right_vbox.addWidget(self.loadButton, 9, 0)
        self.right_vbox.addWidget(self.cutButton, 9, 1)
        self.right_vbox.addWidget(self.saveImgButton, 10, 0)
        self.right_vbox.addWidget(self.saveDatButton, 10, 1)

        # create info box
        self.createInfoGroup()
        self.right_vbox.addLayout(self.InfoBox, 6, 0, 3, 2)


    def createInfoGroup(self):

        self.InfoBox = QVBoxLayout()
        self.InfoGroup = QGroupBox("Info")
        self.infoGrid = QGridLayout()
        self.infoGrid.setVerticalSpacing(10)
        InfoTags = [["NX =","  a ="],
                    ["NY =","  b ="],
                    ["NZ =","  c ="]]
        self.InfoLabs = []

        for irow in range(3):
            tmp = []
            for icol in range(2):
                self.infoGrid.addWidget(QLabel(InfoTags[irow][icol]),
                                        irow, icol*2)
                label = QLabel()
                # label.setFrameStyle(QFrame.StyledPanel|QFrame.Sunken)
                label.setAlignment(Qt.AlignRight|Qt.AlignVCenter)
                if icol == 0:
                    label.setMinimumWidth(30)
                else:
                    label.setMinimumWidth(55)
                label.setMaximumHeight(30)
                self.infoGrid.addWidget(label, irow, icol*2+1)
                tmp += [label]
            self.InfoLabs += [tmp]
        self.InfoGroup.setLayout(self.infoGrid)

        self.InfoBox.addWidget(self.InfoGroup)
        self.InfoBox.addStretch(1)

    def create_main_frame(self):
        self.main_frame = QWidget()
        # self.main_frame.resize(550, 420)
        
        # left part
        self.createLeftPart()
        # right part
        self.createRightPart()

        # put together
        hbox = QHBoxLayout()
        hbox.addLayout(self.left_vbox)
        hbox.addLayout(self.right_vbox)
        self.main_frame.setLayout(hbox)

        self.setCentralWidget(self.main_frame)
    
    def create_status_bar(self):
        self.status_text = QLabel("Please load a data file")
        self.statusBar().addWidget(self.status_text, 1)

    def rx_changed(self, ii):
        self.repeat_x = ii

    def ry_changed(self, ii):
        self.repeat_y = ii

if __name__ == "__main__":
    app = QApplication(sys.argv)
    form = Form()
    form.show()
    sys.exit(app.exec_())
