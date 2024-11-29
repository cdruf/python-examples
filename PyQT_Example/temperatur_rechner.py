# -*- coding: utf-8 -*-
"""

"""

import sys

from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import QGridLayout, QLabel, QLineEdit, QToolBar, QApplication, QWidget, QMainWindow


def celsius_to_fahrenheit(celsius):
    return celsius * 9 / 5 + 32


def fahrenheit_to_celsius(fahrenheit):
    return (fahrenheit - 32) * 5 / 9


class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setWindowTitle("My Temperature Converter")

        layout = QGridLayout()

        self.lb_celsius = QLabel('Celsius')
        layout.addWidget(self.lb_celsius, 0, 0)
        self.tf_celsius = QLineEdit()
        self.tf_celsius.setValidator(QDoubleValidator())
        layout.addWidget(self.tf_celsius, 0, 1)

        self.lb_fahrenheit = QLabel('Fahrenheit')
        layout.addWidget(self.lb_fahrenheit, 1, 0)
        self.tf_fahrenheit = QLineEdit()
        self.tf_fahrenheit.setValidator(QDoubleValidator())
        layout.addWidget(self.tf_fahrenheit, 1, 1)

        self.tf_celsius.textEdited.connect(self.on_celsius_change)
        self.tf_fahrenheit.textEdited.connect(self.on_fahrenheit_change)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        toolbar = QToolBar("My main toolbar")
        self.addToolBar(toolbar)

    def on_celsius_change(self, s):
        if s != '':
            c = float(s)
            f = celsius_to_fahrenheit(c)
            print(f)
            self.tf_fahrenheit.setText(str(f))

    def on_fahrenheit_change(self, s):
        if s != '':
            f = float(s)
            c = fahrenheit_to_celsius(f)
            print(c)
            self.tf_celsius.setText(str(c))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()
    print("Finished")
