#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 15:00:06 2021

@author: robgc
"""
from multiprocessing import set_start_method
import csv
import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from backend import trans_plot, pop_plot
from vals_413 import d12_413, d23_413, spontaneous_21_413, spontaneous_32_413, kp_413, kc_413, \
                        func_Ic_413, func_Ip_413, func_omega_c_413, func_omega_p_413
from vals_318 import d12_318, d23_318, spontaneous_21_318, spontaneous_32_318, kp_318, kc_318, \
                        func_Ic_318, func_Ip_318, func_omega_c_318, func_omega_p_318
import time


class UI(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(UI, self).__init__(*args, **kwargs)
        self.setWindowTitle("Sr88 3 Level System Simulator")
        self.screen = QDesktopWidget().screenGeometry(-1)
        #self.setFixedSize(int(self.screen.height()*0.9), self.screen.height()-100)
        self.overallLayout = QHBoxLayout()
        self.leftLayout = QVBoxLayout()
        self.rightLayout = QVBoxLayout()
        self.overallLayout.addLayout(self.leftLayout)
        self.overallLayout.addLayout(self.rightLayout)
        self.overallLayout.setSpacing(10)
        self._centralWidget = QWidget(self)
        self.setCentralWidget(self._centralWidget)
        self._centralWidget.setLayout(self.overallLayout)
        self.add_toolbar()
        self.add_dropdowns()
        self.add_inputs()
        self.add_checkboxes()
        self.add_images()
    
    def add_toolbar(self):
        self.toolbar = QToolBar()
        self.addToolBar(Qt.BottomToolBarArea, self.toolbar)
        self.load_button = QAction("Load CSV", self)
        self.load_button.setStatusTip("Load CSV")
        self.load_button.triggered.connect(self.load_csv)
        self.toolbar.addAction(self.load_button)
        self.save_button = QAction("Save CSV", self)
        self.save_button.setStatusTip("Save CSV")
        self.save_button.triggered.connect(self.save_csv)
        self.toolbar.addAction(self.save_button)
        self.clear_button = QAction("Clear", self)
        self.clear_button.setStatusTip("Clear")
        self.toolbar.addAction(self.clear_button)
        self.clear_button.triggered.connect(self.clear_text)
        self.plot_button = QAction("Plot Transmission", self)
        self.plot_button.setStatusTip("Plot Transmission")
        self.toolbar.addAction(self.plot_button)
        self.plot_button.triggered.connect(self.transmission)
        self.pop_button = QAction("Plot Population", self)
        self.pop_button.setStatusTip("Plot Population")
        self.toolbar.addAction(self.pop_button)
        self.pop_button.triggered.connect(self.population)
        self.exit_button = QAction("Exit", self)
        self.exit_button.setStatusTip("Exit")
        self.toolbar.addAction(self.exit_button)
        self.exit_button.triggered.connect(self.close)
        
    def add_images(self):
        self.leveldiagram = QLabel()
        self.levelpixmap = QPixmap('imgs/3ls.PNG')
        self.leveldiagram.setPixmap(self.levelpixmap)
        self.leveldiagram.resize(self.levelpixmap.width(), self.levelpixmap.height())
        self.rightLayout.addWidget(self.leveldiagram, alignment=Qt.AlignCenter)
        self.master_eq = QLabel()
        self.master_pix = QPixmap('imgs/master2.PNG')
        self.master_eq.setPixmap(self.master_pix)
        self.master_eq.resize(self.master_pix.width(), self.master_pix.height())
        self.rightLayout.addWidget(self.master_eq, alignment=Qt.AlignCenter) 
    
    def add_dropdowns(self):
        self.system_choice = QComboBox()
        self.system_choice.addItems(["Sr88 Singlet Rydberg", "Sr88 Triplet Rydberg"])
        self.system_choice.setCurrentIndex(0)
        self.leftLayout.addWidget(self.system_choice)
        self.input_type = QComboBox()
        self.input_type.addItems(["Enter Laser Powers and Diameters", "Enter Laser Intensities", "Enter Rabi Frequencies"])
        self.input_type.currentIndexChanged.connect(self.grey)
        self.input_type.setCurrentIndex(0)
        self.leftLayout.addWidget(self.input_type)
        
        
    def add_checkboxes(self):
        self.doppler = QCheckBox("Include Doppler Broadening?")
        self.transit = QCheckBox ("Include Transit Time Broadening?")
        self.leftLayout.addWidget(self.doppler)
        self.leftLayout.addWidget(self.transit)
        
    def add_inputs(self):
        self.labels = {}
        self.boxes = {}
        inputs_layout = QGridLayout()
        
        labels = {"Probe Laser Power (W)" : (0, 0), 
                  "Coupling Laser Power (W)" : (1, 0),
                  "Probe Laser Diameter (m)" : (2, 0),
                  "Coupling Laser Diameter (m)" : (3, 0),
                  "Probe Laser Intensity (W/m^2)" : (4, 0), 
                  "Coupling Laser Intensity (W/m^2)" : (5, 0), 
                  "Probe Rabi Frequency (Hz)" : (6, 0), 
                  "Coupling Rabi Frequency (Hz)" : (7, 0), 
                  "Probe Laser Linewidth (Hz)" : (8, 0), 
                  "Coupling Laser Linewidth (Hz)" : (9, 0),
                  "Atomic Density (m^-3)" : (10, 0),
                  "Atomic Beam Width (m)" : (11, 0), 
                  "Oven Temperature (K)" : (12, 0), 
                  "Atomic Beam Divergence Angle (Rad)" : (13, 0),
                  "Minimum Detuning (Hz)" : (14, 0),
                  "Maximum Detunung (Hz)" : (15, 0),
                  "Number of Plotted Points" : (16, 0),
                  "Coupling Laser Detuning (Hz)" : (17, 0)}
        
        boxes = {"pp" : (0, 1), 
                 "cp" : (1, 1), 
                 "pd" : (2, 1),
                 "cd" : (3, 1),
                 "Ip" : (4, 1), 
                 "Ic" : (5, 1), 
                 "omega_p" : (6, 1), 
                 "omega_c" : (7, 1),
                 "lwp" : (8, 1),
                 "lwc" : (9, 1),
                 "density" : (10, 1), 
                 "sl" : (11, 1), 
                 "T" : (12, 1), 
                 "alpha" : (13, 1),
                 "dmin" : (14, 1),
                 "dmax" : (15, 1),
                 "steps" : (16, 1),
                 "delta_c" : (17, 1)}
        
        for text, pos in labels.items():
            self.labels[text] = QLabel(text)
            inputs_layout.addWidget(self.labels[text], pos[0], pos[1])
        
        for text, pos in boxes.items():
            self.boxes[text] = QLineEdit()
            inputs_layout.addWidget(self.boxes[text], pos[0], pos[1])
        
        laser_params = ["Ip", "Ic", "omega_p", "omega_c"]
        for i in laser_params:
            self.boxes[i].setReadOnly(True)

        self.leftLayout.addLayout(inputs_layout)

    def get_text(self, parameter):
        return self.boxes[parameter].text()
    
    def set_text (self, parameter, text):
        self.boxes[parameter].setText(text)

    def clear_text(self):
        for parameter, val in self.boxes.items():
            self.set_text(parameter, "")
        
    def get_params(self):
        sim_vals = {}
        for param, box in self.boxes.items():
            sim_vals[param] = self.get_text(param)
        return sim_vals
        
    def transmission(self):
        vals = self.get_params()
        for param, val in vals.items():
        	if vals[param] == "0":
        		vals[param] = 0
        	if vals[param] == "":
        		vals[param] = 0
        	else:
        		vals[param] = float(val)
        vals["dmin"] = int(vals["dmin"])
        vals["dmax"] = int(vals["dmax"])
        vals["steps"] = int(vals["steps"])
        
        if self.system_choice.currentIndex() == 0:
            if self.input_type.currentIndex() == 0:
                vals["Ip"] = func_Ip_413(vals["pp"], vals["pd"])
                vals["Ic"] = func_Ic_413(vals["cp"], vals["cd"])
                vals["omega_p"] = func_omega_p_413(vals["Ip"])
                vals["omega_c"] = func_omega_c_413(vals["Ic"])
            if self.input_type.currentIndex() == 1:
                vals["omega_p"] = func_omega_p_413(vals["Ip"])
                vals["omega_c"] = func_omega_c_413(vals["Ic"])
        
        if self.system_choice.currentIndex() == 1:
            if self.input_type.currentIndex() == 0:
                vals["Ip"] = func_Ip_318(vals["pp"], vals["pd"])
                vals["Ic"] = func_Ic_318(vals["cp"], vals["cd"])
                vals["omega_p"] = func_omega_p_318(vals["Ip"])
                vals["omega_c"] = func_omega_c_318(vals["Ic"])
            if self.input_type.currentIndex() == 1:
                vals["omega_p"] = func_omega_p_318(vals["Ip"])
                vals["omega_c"] = func_omega_c_318(vals["Ic"])
        
        if self.doppler.isChecked():
            gauss = "Y"
        else:
            gauss = "N"

        if self.transit.isChecked():
        	tt = "Y"
        else:
        	tt = "N"
        if vals["pd"] == 0:
        	tt = "N"
        if vals["cd"] == 0:
        	tt = "N"
        
        if self.system_choice.currentIndex() == 0:
            dlist, tlist = trans_plot(vals["delta_c"], vals["omega_p"], vals["omega_c"], spontaneous_32_413, spontaneous_21_413, 
                       vals["lwp"], vals["lwc"], vals["dmin"], vals["dmax"], vals["steps"], gauss, kp_413, kc_413, 
                       vals["density"], d12_413, vals["sl"], vals["T"], vals["alpha"], vals["pd"], vals["cd"], tt)
            self.save_dialog()
            
        if self.system_choice.currentIndex() == 1:
            dlist, tlist = trans_plot(vals["delta_c"], vals["omega_p"], vals["omega_c"], spontaneous_32_318, spontaneous_21_318, 
                       vals["lwp"], vals["lwc"], vals["dmin"], vals["dmax"], vals["steps"], gauss, kp_318, kc_318, 
                       vals["density"], d12_318, vals["sl"], vals["T"], vals["alpha"], vals["pd"], vals["cd"], tt)
            self.save_dialog()
        
    def population(self):
        vals = self.get_params()
        for param, val in vals.items():
            if vals[param] == "":
                vals[param] = 0
            else:
                vals[param] = float(val)
        vals["dmin"] = int(vals["dmin"])
        vals["dmax"] = int(vals["dmax"])
        vals["steps"] = int(vals["steps"])
        
        if self.system_choice.currentIndex() == 0:
            if self.input_type.currentIndex() == 0:
                vals["Ip"] = func_Ip_413(vals["pp"], vals["pd"])
                vals["Ic"] = func_Ic_413(vals["cp"], vals["cd"])
                vals["omega_p"] = func_omega_p_413(vals["Ip"])
                vals["omega_c"] = func_omega_c_413(vals["Ic"])
            if self.input_type.currentIndex() == 1:
                vals["omega_p"] = func_omega_p_413(vals["Ip"])
                vals["omega_c"] = func_omega_c_413(vals["Ic"])
        
        if self.system_choice.currentIndex() == 1:
            if self.input_type.currentIndex() == 0:
                vals["Ip"] = func_Ip_318(vals["pp"], vals["pd"])
                vals["Ic"] = func_Ic_318(vals["cp"], vals["cd"])
                vals["omega_p"] = func_omega_p_318(vals["Ip"])
                vals["omega_c"] = func_omega_c_318(vals["Ic"])
            if self.input_type.currentIndex() == 1:
                vals["omega_p"] = func_omega_p_318(vals["Ip"])
                vals["omega_c"] = func_omega_c_318(vals["Ic"])
        
        if self.doppler.isChecked():
            gauss = "Y"
        else:
            gauss = "N"

        if self.transit.isChecked():
        	tt = "Y"
        else:
        	tt = "N"

        if vals["pd"] == 0:
        	tt = "N"
        if vals["cd"] == 0:
        	tt = "N"
        
        self.showdialog()
        
        if self.system_choice.currentIndex() == 0:
            dlist, plist = pop_plot(self.state_number, vals["delta_c"], vals["omega_p"], vals["omega_c"], spontaneous_32_413, spontaneous_21_413, 
                   vals["lwp"], vals["lwc"], vals["dmin"], vals["dmax"], vals["steps"], gauss, 
                   vals["T"], kp_413, kc_413, vals["alpha"], vals["pd"], vals["cd"], tt)
            self.save_dialog()
            
        if self.system_choice.currentIndex() == 1:
            dlist, plist = pop_plot(self.state_number, vals["delta_c"], vals["omega_p"], vals["omega_c"], spontaneous_32_318, spontaneous_21_318, 
                   vals["lwp"], vals["lwc"], vals["dmin"], vals["dmax"], vals["steps"], gauss, 
                   vals["T"], kp_318, kc_318, vals["alpha"], vals["pd"], vals["cd"], tt)
            self.save_dialog()
    
    def load_csv(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.AnyFile)
        dlg.setNameFilter("csv (*.csv)")
        selected = dlg.exec()
        if selected:
            self.filename = dlg.selectedFiles()[0]
            dlg.close()
        else:
            dlg.close()
            return
        if self.filename == "":
            dlg.close()
            return
        
        with open(f"{self.filename}", "rt") as file: 
            reader = csv.reader(file, delimiter=',')
            for rows in reader:
                param = rows[0]
                val = rows[1]
                self.set_text(param, val)
            
    def save_csv(self):
        dlg = QFileDialog()
        self.filename = dlg.getSaveFileName(self, 'Save File')[0]
        if self.filename == "":
            dlg.close()
            return
        with open(f"{self.filename}", "w") as file:    
            write = csv.writer(file, delimiter=',')    
            for param, val in self.get_params().items():
                write.writerow([param, val])                        
        
    def grey(self, i):
        if i == 0:
            self.set_text("Ip", "")
            self.set_text("Ic", "")
            self.set_text("omega_p", "")
            self.set_text("omega_c", "")
            self.boxes["pp"].setReadOnly(False)
            self.boxes["cp"].setReadOnly(False)
            self.boxes["Ip"].setReadOnly(True)
            self.boxes["Ic"].setReadOnly(True)
            self.boxes["omega_p"].setReadOnly(True)
            self.boxes["omega_c"].setReadOnly(True)
        if i == 1:
            self.set_text("pp", "")
            self.set_text("cp", "")
            self.set_text("omega_p", "")
            self.set_text("omega_c", "")
            self.boxes["Ip"].setReadOnly(False)
            self.boxes["Ic"].setReadOnly(False)
            self.boxes["omega_p"].setReadOnly(True)
            self.boxes["omega_c"].setReadOnly(True)
        if i == 2:
            self.set_text("pp", "")
            self.set_text("cp", "")
            self.set_text("Ip", "")
            self.set_text("Ic", "")
            self.boxes["pp"].setReadOnly(True)
            self.boxes["cp"].setReadOnly(True)
            self.boxes["Ip"].setReadOnly(True)
            self.boxes["Ic"].setReadOnly(True)
            self.boxes["omega_p"].setReadOnly(False)
            self.boxes["omega_c"].setReadOnly(False)
            
    def showdialog(self):
        self.d = QDialog()
        self.dd = QComboBox(self.d)
        self.dd.move(100, 0)
        self.dd.addItems(["Ground", "Intermediate", "Rydberg"])
        self.dd.setCurrentIndex(0)
        self.b1 = QPushButton("ok",self.d)
        self.b1.move(110, 50)
        self.b1.clicked.connect(self.state)
        
        self.d.setWindowTitle("Choose State to Plot")
        self.d.setWindowModality(Qt.ApplicationModal)
        self.d.exec_()
        
    def state(self):
        if self.dd.currentIndex() == 0:
            self.state_number = "Ground"
        if self.dd.currentIndex() == 1:
            self.state_number = "Intermediate"
        if self.dd.currentIndex() == 2:
            self.state_number = "Rydberg"
        self.d.close()
        
    def save_dialog(self):
        box = QMessageBox.question(self, 'Save', 'Do you want to save data as a .csv?')
        
        
def main():        
    app = QApplication(sys.argv)
    window = UI()
    window.show()
    sys.exit(app.exec_())
    
if __name__ == "__main__":
    set_start_method("spawn")
    main()
    