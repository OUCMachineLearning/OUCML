# -*- coding: utf-8 -*-

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import numpy as np

class GraphicsScene(QGraphicsScene):
    def __init__(self, mode_list, parent=None):
        QGraphicsScene.__init__(self, parent)
        self.modes = mode_list
        self.mouse_clicked = False
        self.prev_pt = None

        # self.masked_image = None

        # save the points
        self.mask_points = []
        self.sketch_points = []
        self.stroke_points = []

        # save the history of edit
        self.history = []

        # strokes color
        self.stk_color = None

    def reset(self):
        # save the points
        self.mask_points = []
        self.sketch_points = []
        self.stroke_points = []

        # save the history of edit
        self.history = []

        # strokes color
        self.stk_color = None

        self.prev_pt = None

    def mousePressEvent(self, event):
        self.mouse_clicked = True

    def mouseReleaseEvent(self, event):
        self.prev_pt = None
        self.mouse_clicked = False

    def mouseMoveEvent(self, event):
        if self.mouse_clicked:
            if self.modes[0] == 1:
                if self.prev_pt:
                    self.drawMask(self.prev_pt, event.scenePos())
                    pts = {}
                    pts['prev'] = (int(self.prev_pt.x()),int(self.prev_pt.y()))
                    pts['curr'] = (int(event.scenePos().x()),int(event.scenePos().y()))
                    self.mask_points.append(pts)
                    self.history.append(0)
                    self.prev_pt = event.scenePos()
                else:
                    self.prev_pt = event.scenePos()
            elif self.modes[1] == 1:
                if self.prev_pt:
                    self.drawSketch(self.prev_pt, event.scenePos())
                    pts = {}
                    pts['prev'] = (int(self.prev_pt.x()),int(self.prev_pt.y()))
                    pts['curr'] = (int(event.scenePos().x()),int(event.scenePos().y()))
                    self.sketch_points.append(pts)
                    self.history.append(1)
                    self.prev_pt = event.scenePos()
                else:
                    self.prev_pt = event.scenePos()
            elif self.modes[2] == 1:
                if self.prev_pt:
                    self.drawStroke(self.prev_pt, event.scenePos())
                    pts = {}
                    pts['prev'] = (int(self.prev_pt.x()),int(self.prev_pt.y()))
                    pts['curr'] = (int(event.scenePos().x()),int(event.scenePos().y()))
                    pts['color'] = self.stk_color
                    self.stroke_points.append(pts)
                    self.history.append(2)
                    self.prev_pt = event.scenePos()
                else:
                    self.prev_pt = event.scenePos()

    def drawMask(self, prev_pt, curr_pt):
        lineItem = QGraphicsLineItem(QLineF(prev_pt, curr_pt))
        lineItem.setPen(QPen(Qt.white, 12, Qt.SolidLine)) # rect
        self.addItem(lineItem)

    def drawSketch(self, prev_pt, curr_pt):
        lineItem = QGraphicsLineItem(QLineF(prev_pt, curr_pt))
        lineItem.setPen(QPen(Qt.black, 1, Qt.SolidLine)) # rect
        self.addItem(lineItem)

    def drawStroke(self, prev_pt, curr_pt):
        lineItem = QGraphicsLineItem(QLineF(prev_pt, curr_pt))
        lineItem.setPen(QPen(QColor(self.stk_color), 4, Qt.SolidLine)) # rect
        self.addItem(lineItem)

    def get_stk_color(self, color):
        self.stk_color = color

    def erase_prev_pt(self):
        self.prev_pt = None

    def reset_items(self):
        for i in range(len(self.items())):
            item = self.items()[0]
            self.removeItem(item)
        
    def undo(self):
        if len(self.items())>1:
            if len(self.items())>=9:
                for i in range(8):
                    item = self.items()[0]
                    self.removeItem(item)
                    if self.history[-1] == 0:
                        self.mask_points.pop()
                        self.history.pop()
                    elif self.history[-1] == 1:
                        self.sketch_points.pop()
                        self.history.pop()
                    elif self.history[-1] == 2:
                        self.stroke_points.pop()
                        self.history.pop()
                    elif self.history[-1] == 3:
                        self.history.pop()
            else:
                for i in range(len(self.items())-1):
                    item = self.items()[0]
                    self.removeItem(item)
                    if self.history[-1] == 0:
                        self.mask_points.pop()
                        self.history.pop()
                    elif self.history[-1] == 1:
                        self.sketch_points.pop()
                        self.history.pop()
                    elif self.history[-1] == 2:
                        self.stroke_points.pop()
                        self.history.pop()
                    elif self.history[-1] == 3:
                        self.history.pop()
