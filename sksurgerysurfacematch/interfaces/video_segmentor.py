# -*- coding: utf-8 -*-

""" Base class (pure virtual interface) for classes to do video segmentation """

import numpy as np


class VideoSegmentor:

    def segment(self,
                image: np.ndarray
                ):
        """
        A derived class must implement this.

        :param image: image, BGR
        :return: image, same size as input, 1 channel, uchar, [0-255].
        """
        raise NotImplementedError("Derived classes should implement this.")
