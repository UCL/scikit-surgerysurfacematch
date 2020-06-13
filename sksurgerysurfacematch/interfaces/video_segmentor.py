# -*- coding: utf-8 -*-

""" Base class (pure virtual interface) for classes to do video segmentation """

import numpy as np


class VideoSegmentor:
    """
    Base class for classes that can segment a video image into a binary mask.
    For example, a deep network that can produce a mask of background=0,
    foreground=255.
    """
    def segment(self, image: np.ndarray):
        """
        A derived class must implement this.

        :param image: image, BGR
        :return: image, same size as input, 1 channel, uchar, [0-255].
        """
        raise NotImplementedError("Derived classes should implement this.")
