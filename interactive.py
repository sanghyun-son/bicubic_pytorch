import sys
import math
import argparse
import typing

import numpy as np
import imageio
from PIL import Image

import cv2
import torch

import core_warp
import utils

from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtGui import QPainter
from PyQt5.QtGui import QPen
from PyQt5.QtGui import QBrush
from PyQt5.QtCore import Qt


class Interactive(QMainWindow):

    def __init__(self, app: QApplication, img: str) -> None:
        super().__init__()
        self.setStyleSheet('background-color: gray;')
        self.margin = 300
        img = Image.open(img)
        #w, h = img.size
        #img = img.resize((2 * w, 2 * h), Image.NEAREST)
        self.img = np.array(img)
        self.img_tensor = utils.np2tensor(self.img).cuda()
        self.img_h = self.img.shape[0]
        self.img_w = self.img.shape[1]

        self.offset_h = self.margin
        self.offset_w = self.img_w + 2 * self.margin

        window_h = self.img_h + 2 * self.margin
        window_w = 2 * self.img_w + 3 * self.margin

        monitor_resolution = app.desktop().screenGeometry()
        screen_h = monitor_resolution.height()
        screen_w = monitor_resolution.width()

        screen_offset_h = (screen_h - window_h) // 2
        screen_offset_w = (screen_w - window_w) // 2

        self.setGeometry(screen_offset_w, screen_offset_h, window_w, window_h)
        self.reset_cps()
        self.line_order = ('tl', 'tr', 'br', 'bl')
        self.grab = None

        self.inter = cv2.INTER_CUBIC
        self.inter_idx = 2
        self.backend = 'opencv'
        self.update()
        return

    def reset_cps(self) -> None:
        self.cps = {
            'tl': (0, 0),
            'tr': (0, self.img_w - 1),
            'bl': (self.img_h - 1, 0),
            'br': (self.img_h - 1, self.img_w - 1),
        }
        return

    def keyPressEvent(self, e) -> None:
        if e.key() == Qt.Key_Escape:
            self.close()

        if e.key() == Qt.Key_I:
            self.inter_idx = (self.inter_idx + 1) % 3
            if self.inter_idx == 0:
                self.inter = cv2.INTER_NEAREST
            elif self.inter_idx == 1:
                self.inter = cv2.INTER_LINEAR
            else:
                self.inter = cv2.INTER_CUBIC
        elif e.key() == Qt.Key_M:
            if self.backend == 'opencv':
                self.backend = 'core'
            elif self.backend == 'core':
                self.backend = 'opencv'
        elif e.key() == Qt.Key_R:
            self.reset_cps()

        self.update()
        return

    def mousePressEvent(self, e) -> None:
        is_left = e.buttons() & Qt.LeftButton
        if is_left:
            threshold = 20
            min_dist = 987654321
            for key, val in self.cps.items():
                y, x = val
                dy = e.y() - y - self.offset_h
                dx = e.x() - x - self.offset_w
                dist = dy ** 2 + dx ** 2
                if dist < min_dist:
                    min_dist = dist
                    self.grab = key

            if min_dist > threshold ** 2:
                self.grab = None

        return

    def get_matrix(self) -> np.array:
        points_from = np.array([
            [0, 0],
            [self.img_w - 1, 0],
            [0, self.img_h - 1],
            [self.img_w - 1, self.img_h - 1],
        ]).astype(np.float32)
        points_to = np.array([
            [self.cps['tl'][1], self.cps['tl'][0]],
            [self.cps['tr'][1], self.cps['tr'][0]],
            [self.cps['bl'][1], self.cps['bl'][0]],
            [self.cps['br'][1], self.cps['br'][0]],
        ]).astype(np.float32)
        m = cv2.getPerspectiveTransform(points_from, points_to)
        return m

    def get_dimension(
            self,
            m: np.array) -> typing.Tuple[float, float, float, float]:

        '''

        '''

        '''
        What is a difference between corners and corner_points?
        corners:
            Actual corners of a rectangular image.
            Determine the image size.
        corner_points:
            The point coordinates.
            Determine the pixel position.
        '''
        corners = np.array([
            [-0.5, -0.5, self.img_w - 0.5, self.img_w - 0.5],
            [-0.5, self.img_h - 0.5, -0.5, self.img_h - 0.5],
            [1, 1, 1, 1],
        ])
        corners = np.matmul(m, corners)
        corners /= corners[-1, :]
        y_min = corners[1].min() + 0.5
        x_min = corners[0].min() + 0.5
        h_new = math.floor(corners[1].max() - y_min + 0.5)
        w_new = math.floor(corners[0].max() - x_min + 0.5)
        '''
        corner_points = np.array([
            [0, 0, self.img_w - 1, self.img_w - 1],
            [0, self.img_h - 1, 0, self.img_h - 1],
            [1, 1, 1, 1],
        ])
        corner_points = np.matmul(m, corner_points)
        corner_points /= corner_points[-1, :]
        y_min = corner_points[1].min()
        x_min = corner_points[0].min()
        h_new = math.floor(corner_points[1].max() - y_min)
        w_new = math.floor(corner_points[0].max() - x_min)
        '''
        return y_min, x_min, h_new, w_new

    def mouseMoveEvent(self, e) -> None:
        if self.grab is not None:
            y_old, x_old = self.cps[self.grab]
            y_new = e.y() - self.offset_h
            x_new = e.x() - self.offset_w
            self.cps[self.grab] = (y_new, x_new)

            is_convex = True
            #cross = None
            for i, pos in enumerate(self.line_order):
                y1, x1 = self.cps[pos]
                y2, x2 = self.cps[self.line_order[(i + 1) % 4]]
                y3, x3 = self.cps[self.line_order[(i + 2) % 4]]
                dx1 = x2 - x1
                dy1 = y2 - y1
                dx2 = x3 - x2
                dy2 = y3 - y2
                cross_new = dx1 * dy2 - dy1 * dx2
                if cross_new < 6000:
                    is_convex = False
                    break

            if not is_convex:
                self.cps[self.grab] = (y_old, x_old)

        self.update()
        return

    def mouseReleaseEvent(self, e) -> None:
        if self.grab is not None:
            self.grab = None

        return

    def paintEvent(self, e) -> None:
        if self.inter == cv2.INTER_NEAREST:
            inter_method = 'Nearest'
        elif self.inter == cv2.INTER_LINEAR:
            inter_method = 'Bilinear'
        elif self.inter == cv2.INTER_CUBIC:
            inter_method = 'Bicubic'

        self.setWindowTitle(
            'Interpolation: {} / backend: {}'.format(inter_method, self.backend)
        )

        qimg = QImage(
            self.img,
            self.img_w,
            self.img_h,
            3 * self.img_w,
            QImage.Format_RGB888,
        )
        qpix = QPixmap(qimg)

        qp = QPainter()
        qp.begin(self)
        qp.drawPixmap(self.margin, self.margin, self.img_w, self.img_h, qpix)

        m = self.get_matrix()
        y_min, x_min, h_new, w_new = self.get_dimension(m)
        mc = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
        m = np.matmul(mc, m)
        if self.backend == 'opencv':
            warp = cv2.warpPerspective(
                self.img, m, (w_new, h_new), flags=self.inter,
            )
        elif self.backend == 'core':
            warp = core_warp.warp(
                self.img_tensor,
                torch.Tensor(m),
                sizes=(h_new, w_new),
                kernel=inter_method.lower(),
                fill_value=0.5
            )
            warp = utils.tensor2np(warp)

        qimg_warp = QImage(warp, w_new, h_new, 3 * w_new, QImage.Format_RGB888)
        qpix_warp = QPixmap(qimg_warp)
        qp.drawPixmap(
            self.offset_w + x_min,
            self.offset_h + y_min,
            w_new,
            h_new,
            qpix_warp,
        )
        '''
        for i, pos in enumerate(self.line_order):
            j = (i + 1) % 4
            y, x = self.cps[pos]
            y = y + self.offset_h
            x = x + self.offset_w
            y_next, x_next = self.cps[self.line_order[j]]
            y_next = y_next + self.offset_h
            x_next = x_next + self.offset_w
            qp.drawLine(x, y, x_next, y_next)
        '''
        center_y = self.offset_h + self.img_h // 2
        center_x = self.offset_w + self.img_w // 2

        pen_blue = QPen(Qt.blue, 5)
        pen_white = QPen(Qt.white, 10)
        text_size = 20
        #brush = QBrush(Qt.red, Qt.SolidPattern)
        #qp.setBrush(brush)
        for key, val in self.cps.items():
            y, x = val
            y = y + self.offset_h
            x = x + self.offset_w
            qp.setPen(pen_blue)
            #qp.drawEllipse(x, y, 3, 3)
            qp.drawPoint(x, y)
            qp.setPen(pen_white)
            dy = y - center_y
            dx = x - center_x
            dl = math.sqrt(dy ** 2 + dx ** 2) / 10
            qp.drawText(
                x + (dx / dl) - text_size // 2,
                y + (dy / dl) - text_size // 2,
                text_size,
                text_size,
                int(Qt.AlignCenter),
                key,
            )

        qp.end()
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, default='example/butterfly_corners.png')
    parser.add_argument('--full', action='store_true')
    cfg = parser.parse_args()

    app = QApplication(sys.argv)
    sess = Interactive(app, cfg.img)

    if cfg.full:
        sess.showFullScreen()
    else:
        sess.show()

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()