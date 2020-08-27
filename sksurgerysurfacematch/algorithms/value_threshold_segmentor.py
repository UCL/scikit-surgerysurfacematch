# -*- coding: utf-8 -*-

""" Dummy segmentor, just to test the framework. """

import numpy as np
import cv2
import sksurgerysurfacematch.interfaces.video_segmentor as vs


class ValueThresholdSegmentor(vs.VideoSegmentor):
    """
    Dummy segmentor, to test the framework. Simply converts BGR to HSV,
    extracts the value channel, and applies a threshold between [0-255].

    It's not really useful for anything other than testing the interface.
    """
    def __init__(self, threshold=127):
        super().__init__()
        self.threshold = threshold

    def segment(self, image: np.ndarray):
        """
        Converts image from BGR to HSV and thresholds the Value channel.

        :param image: image, BGR
        :return: image, same size as input, 1 channel, uchar, [0-255].
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        v_c = hsv[:, :, 2]
        return (v_c > self.threshold) * 255
