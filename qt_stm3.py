#!/usr/bin/env python3
################################################################################
import sys, os
import numpy as np

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from ase.calculators.vasp import VaspChargeDensity

################################################################################
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

################################################################################
class Form(QMainWindow):
    def __init__(self, parent=None):
        super(Form, self).__init__(parent)

        self.filename = None
        self.VaspPchg = None
        self.zcut = None
        self.pc = None
        self.cmap = 'hot'
        self.dpi = 800
        self.repeat_x = 1
        self.repeat_y = 1
        self.vmin = None
        self.vmax = None
        self.selected_cmaps = ['hot', 'hot_r',
                               'Blues', 'Blues_r',
                               'bwr', 'bwr_r',
                               'coolwarm', 'coolwarm_r',
                               'seismic', 'seismic_r',
                               'RdBu', 'RdBu_r', 
                               'Greys', 'Spectral',
                               'rainbow']
            

        # 0 for IsoHeight and 1 for IsoCurrent STM image
        self.whicISO = 0

        # self.create_menu()
        self.create_main_frame()
        self.create_status_bar()
        
        self.on_show()

        self.setWindowTitle('PyQt & matplotlib demo: STM Simulation')
        # self.resize(700, 420)
        if None != FNAME:
            self.filename = FNAME
            self.load_file()

    def updateZV(self):

        zmsg = u'Zm = %.2f \u212B;   Zn = %.2f \u212B' % \
               (self.VaspPchg.zmax_pos, (float(self.zcut) / self.VaspPchg.NZ * self.VaspPchg.c))
        vmsg = u'vmin = %.2E;   vmax = %.2E' % (self.STMData.min(), self.STMData.max())
        self.statusBar.showMessage(zmsg + ';    ' + vmsg)

    def load_file(self):
        if None == self.filename:
            self.filename, _filter = QFileDialog.getOpenFileName(self,
                'Open a data file', '.', '')
        
        if self.filename:
            self.VaspPchg = vaspParchg(self.filename)
            self.zcut = self.VaspPchg.zmax_ind
            self.cutSpinBox.setValue(self.zcut)
            self.cutSpinBox.setMaximum(self.VaspPchg.NZ)
            for irow in range(3):
                self.InfoLabs[irow][0].setText(u'%d' % self.VaspPchg.ngrid[irow])
                self.InfoLabs[irow][1].setText(u'%.2f \u212B' % self.VaspPchg.abc[irow])
            self.GenerateData()
            self.updateZV()
        self.filename = None


    def save_img(self):
        if self.whicISO == 0:
            defaultImgName = f'./STM_H_{self.zcut}.png'
        else:
            defaultImgName = f'./STM_C_{self.zcut}.png'

        imgName, _ = QFileDialog.getSaveFileName(self,
                       'Save Image:', defaultImgName, '')
        if imgName:
            self.fig.savefig(str(imgName), dpi=self.dpi)

    def save_dat(self):
        if self.whicISO == 0:
            defaultDatName = './STM_H_{}_{}x{}.npy'.format(self.zcut, self.repeat_x, self.repeat_y)
        else:
            defaultDatName = './STM_C_{}_{}x{}.npy'.format(self.zcut, self.repeat_x, self.repeat_y)
        datName, _ = QFileDialog.getSaveFileName(self,
                       'Save Data:', defaultDatName, '')

        if datName and self.STMData.size != 0:
            xx = np.vstack((self.STMXCoord,
                            self.STMYCoord,
                            self.STMData))
            np.savetxt(str(datName), xx)
    
    def GenerateData(self):
         
        if self.VaspPchg:
            # isoHeight STM image
            if self.whicISO == 0:
                dat = self.VaspPchg.isoHeight(self.zcut)
            # isoCurrent STM image
            else:
                dat = self.VaspPchg.isoCurrent(self.zcut, pc=self.pc)

            NX = self.VaspPchg.NX
            NY = self.VaspPchg.NY
            a  = (self.VaspPchg.poscar.cell[0,0] + \
                  self.VaspPchg.poscar.cell[0,1] * 1j) / NX
            b  = (self.VaspPchg.poscar.cell[1,0] + \
                  self.VaspPchg.poscar.cell[1,1] * 1j) / NY
            gx, gy = np.mgrid[0:NX * self.repeat_x, 0:NY*self.repeat_y]
            tmp = gx * a + gy * b
            self.STMXCoord = np.real(tmp)
            self.STMYCoord = np.imag(tmp)
            self.STMData   = np.tile(dat, (self.repeat_x, self.repeat_y))

            self.on_show()

    def on_show(self):
        self.axes.clear()        
        # self.axes.set_aspect('equal')
        self.axes.axis('off')

        if self.VaspPchg:
            xmin, xmax = self.STMXCoord.min(), self.STMXCoord.max()
            ymin, ymax = self.STMYCoord.min(), self.STMYCoord.max()
            self.axes.set_xlim(xmin, xmax)
            self.axes.set_ylim(ymin, ymax)

            self.axes.pcolormesh(self.STMXCoord, self.STMYCoord, self.STMData,
                                 cmap=self.cmap,
                                 vmin=self.vmin,
                                 vmax=self.vmax,
                                 shading='gouraud'
                                 )

        self.canvas.draw()

    def createPlotPart(self):
        self.fig = Figure((6.0, 6.0), dpi=self.dpi)
        self.fig.subplots_adjust(left=0.05, right=0.95,
                                 bottom=0.05, top=0.95)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.main_frame)
        
        self.axes = self.fig.add_subplot(111)
        self.axes.set_aspect('equal')
        self.axes.axis('off')
        self.mpl_toolbar = NavigationToolbar(self.canvas, self.main_frame)
        
        self.Plot_vbox = QVBoxLayout()
        self.Plot_vbox.addWidget(self.canvas)
        self.Plot_vbox.addWidget(self.mpl_toolbar)
    
    def createParaPart(self):

        self.loadButton = QPushButton("&Load")
        self.loadButton.clicked.connect(self.load_file)
        self.saveImgButton = QPushButton("Save &Img")
        self.saveImgButton.clicked.connect(self.save_img)
        self.saveDatButton = QPushButton("Save &Dat")
        self.saveDatButton.clicked.connect(self.save_dat)
        self.cutButton = QPushButton("&Apply")
        self.cutButton.clicked.connect(self.GenerateData)

        self.isoComboBox = QComboBox()
        # self.isoComboBox.setMaximumWidth(80)
        self.isoComboBox.addItems(['Iso Height',
                                   'Iso Current'])
        self.isoComboBox.currentIndexChanged.connect(self.isoChanged)

        self.cutLabel = QLabel("Zc")
        self.cutSpinBox = QSpinBox()
        self.cutSpinBox.setRange(0, 1500)
        self.cutSpinBox.setSingleStep(1)
        self.cutSpinBox.valueChanged.connect(self.zcutChanged)

        self.cmapLabel = QLabel('Cm')
        self.cmapComboBox = QComboBox()
        self.cmapComboBox.addItems(self.selected_cmaps)
        self.cmapComboBox.currentIndexChanged.connect(self.cmapChanged)
        
        self.pcLabel = QLabel("Pc")
        self.pcSpinBox = QSpinBox()
        self.pcSpinBox.setRange(0, 99)
        self.pcSpinBox.setSingleStep(1)
        self.pcSpinBox.valueChanged.connect(self.pcChanged)

        repeatXLabel = QLabel('Px')
        repeatYLabel = QLabel('Py')
        repeatXSpinBox = QSpinBox()
        repeatYSpinBox = QSpinBox()
        repeatXSpinBox.setRange(1, 10)
        repeatYSpinBox.setRange(1, 10)
        repeatXSpinBox.valueChanged.connect(self.rx_changed) 
        repeatYSpinBox.valueChanged.connect(self.ry_changed) 

        vminLabel = QLabel('min')
        self.vminLineEdit = QLineEdit()
        # self.vminLineEdit.setFixedWidth(60)
        vmaxLabel = QLabel('max')
        self.vmaxLineEdit = QLineEdit()
        font = QFont()
        font.setPointSize(10)
        self.vminLineEdit.setFont(font)
        self.vmaxLineEdit.setFont(font)
        # self.vminLineEdit.textChanged.connect(self.vminChanged)
        # self.vmaxLineEdit.textChanged.connect(self.vmaxChanged)
        self.vminLineEdit.returnPressed.connect(self.vminChanged)
        self.vmaxLineEdit.returnPressed.connect(self.vmaxChanged)
        # self.vmaxLineEdit.setFixedWidth(60)

        for lab in [repeatXLabel, repeatYLabel, self.cutLabel, self.pcLabel,
                    self.cmapLabel, vminLabel, vmaxLabel]:
            lab.setAlignment(Qt.AlignRight|Qt.AlignVCenter)
            lab.setFixedWidth(30)
            lab.setFrameStyle(QFrame.Panel|QFrame.Raised)

        # for widget in [repeatXSpinBox, repeatYSpinBox, self.cutSpinBox,
        #                self.pcSpinBox, self.cmapComboBox]:
        #     widget.setFixedWidth(60)

        self.inputGroup = QGroupBox('Input')
        self.inputGroup.setMaximumSize(QSize(220, 16777215))
        inputGridBox  = QGridLayout()
        inputGridBox.addWidget(self.isoComboBox, 0, 2, 1, 2)
        # inputGridBox.addWidget(self.isoComboBox, 0, 0)
        inputGridBox.addWidget(repeatXLabel, 1, 0)
        inputGridBox.addWidget(repeatYLabel, 1, 2)
        inputGridBox.addWidget(repeatXSpinBox, 1, 1)
        inputGridBox.addWidget(repeatYSpinBox, 1, 3)
        inputGridBox.addWidget(self.cutLabel, 3, 0)
        inputGridBox.addWidget(self.cutSpinBox, 3, 1)
        inputGridBox.addWidget(self.pcLabel, 3, 2)
        inputGridBox.addWidget(self.pcSpinBox, 3, 3)
        inputGridBox.addWidget(self.cmapLabel, 4, 2)
        inputGridBox.addWidget(self.cmapComboBox, 4, 3)
        inputGridBox.addWidget(vminLabel, 5, 0)
        inputGridBox.addWidget(vmaxLabel, 5, 2)
        inputGridBox.addWidget(self.vminLineEdit, 5, 1)
        inputGridBox.addWidget(self.vmaxLineEdit, 5, 3)
        self.inputGroup.setLayout(inputGridBox)

        self.actionButton = QGridLayout()
        self.actionButton.addWidget(self.loadButton, 0, 0)
        self.actionButton.addWidget(self.cutButton, 0, 1)
        self.actionButton.addWidget(self.saveImgButton, 1, 0)
        self.actionButton.addWidget(self.saveDatButton, 1, 1)

        # create info box
        self.createInfoGroup()
        # self.Para_vbox.addLayout(self.InfoBox, 6, 0, 3, 2)

        self.Para_vbox = QVBoxLayout()
        self.Para_vbox.addWidget(self.inputGroup)
        self.Para_vbox.addWidget(self.InfoGroup)
        self.Para_vbox.addStretch(1)
        self.Para_vbox.addLayout(self.actionButton)


    def createInfoGroup(self):

        # self.InfoBox = QVBoxLayout()
        self.InfoGroup = QGroupBox("Grid")
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
                label.setFrameStyle(QFrame.Panel|QFrame.Sunken)
                label.setAlignment(Qt.AlignRight|Qt.AlignVCenter)
                if icol == 0:
                    label.setMinimumWidth(30)
                else:
                    label.setMinimumWidth(55)
                label.setMinimumHeight(20)
                self.infoGrid.addWidget(label, irow, icol*2+1)
                tmp += [label]
            self.InfoLabs += [tmp]
        self.InfoGroup.setLayout(self.infoGrid)
        # self.InfoBox.addWidget(self.InfoGroup)
        # self.InfoBox.addStretch(1)

    def create_main_frame(self):
        self.main_frame = QWidget()
        # self.main_frame.resize(550, 420)
        
        # Para part
        self.createParaPart()
        # Plot part
        self.createPlotPart()
        # Vertical Line
        Vline = QFrame()
        Vline.setFrameShape(QFrame.VLine)
        Vline.setFrameShadow(QFrame.Sunken)
        # Vline.setObjectName(_fromUtf8("line"))

        # put together
        hbox = QHBoxLayout()
        hbox.addLayout(self.Para_vbox)
        hbox.addWidget(Vline)
        hbox.addLayout(self.Plot_vbox)
        self.main_frame.setLayout(hbox)

        self.setCentralWidget(self.main_frame)
    
    def create_status_bar(self):
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)

        self.statusBar.showMessage('Please load a data file!')
        # self.status_text = QLabel("Please load a data file")
        # self.statusBar().addWidget(self.status_text, 1)

    def rx_changed(self, ii):
        self.repeat_x = ii

    def ry_changed(self, ii):
        self.repeat_y = ii

    def pcChanged(self, pc):
        self.pc = pc
        self.vmin = None
        self.vmax = None
        
        if self.whicISO == 1:
            self.GenerateData()

    # def vminChanged(self, xx):
    def vminChanged(self):
        xx = self.vminLineEdit.text()

        try:
            self.vmin = float(str(xx))
        except ValueError:
            self.vmin = None

        if (self.vmax is not None) and (self.vmin < self.vmax):
            self.on_show()

    # def vmaxChanged(self, xx):
    def vmaxChanged(self):
        xx = self.vmaxLineEdit.text()

        try:
            self.vmax = float(str(xx))
        except ValueError:
            self.vmax = None

        if (self.vmin is not None) and (self.vmin < self.vmax):
            self.on_show()

    def cmapChanged(self, ii):
        self.cmap = self.selected_cmaps[ii]
        self.on_show()

    def zcutChanged(self, zcut):
        self.zcut = zcut
        self.pc = None
        self.vmin = None
        self.vmax = None

        self.vminLineEdit.clear()
        self.vmaxLineEdit.clear()

        if self.VaspPchg:
            self.GenerateData()
            self.updateZV()

    def isoChanged(self, ii):
        self.whicISO = ii
        self.GenerateData()

################################################################################
if __name__ == "__main__":
    if 2 == len(sys.argv) and "" != sys.argv[1]:
        FNAME = sys.argv[1]
    print(sys.argv)
    app = QApplication(sys.argv)
    form = Form()
    form.show()
    sys.exit(app.exec_())
