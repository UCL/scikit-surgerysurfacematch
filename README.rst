scikit-surgerysurfacematch
===============================

.. image:: https://github.com/UCL/scikit-surgerysurfacematch /raw/master/project-icon.png
   :height: 128px
   :width: 128px
   :target: https://github.com/UCL/scikit-surgerysurfacematch
   :alt: Logo

.. image:: https://github.com/UCL/scikit-surgerysurfacematch/workflows/.github/workflows/ci.yml/badge.svg
   :target: https://github.com/UCL/scikit-surgerysurfacematch/actions
   :alt: GitHub Actions CI status

.. image:: https://coveralls.io/repos/github/UCL/scikit-surgerysurfacematch/badge.svg?branch=master&service=github
   :target: https://coveralls.io/github/UCL/scikit-surgerysurfacematch?branch=master
   :alt: Coveralls coverage status

.. image:: https://readthedocs.org/projects/scikit-surgerysurfacematch/badge/?version=latest
    :target: http://scikit-surgerysurfacematch.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status



Author: Matt Clarkson

scikit-surgerysurfacematch is part of the `SNAPPY`_ software project, developed at the `Wellcome EPSRC Centre for Interventional and Surgical Sciences`_, part of `University College London (UCL)`_.

scikit-surgerysurfacematch supports Python 3.6 - 3.8

scikit-surgerysurfacematch contains algorithms that are useful in stereo reconstruction from video images, and matching to a pre-operative 3D model, represented as a point cloud.

Features
--------

* Base classes (pure virtual interfaces), for video segmentation, stereo reconstruction, rigid registration / pose estimation. See ```sksurgerysurfacematch/algorithms```
* A base class to handle rectification properly, and the right coordinate transformation, to save you the trouble.
* Stereo reconstruction classes based on Stoyanov MICCAI 2010, and OpenCV SGBM reconstruction, using above interface, and both allowing for optional masking.
* Rigid registration using PCL's ICP implementation, which is wrapped in scikit-surgerypclcpp
* A pipeline to combine the above, segment a video pair, do reconstruction, and register to a 3D model, where each part can then be swapped with different implementations of the above.


Developing
----------

Cloning
^^^^^^^

You can clone the repository using the following command:

::

    git clone https://github.com/UCL/scikit-surgerysurfacematch


Running tests
^^^^^^^^^^^^^
Pytest is used for running unit tests:
::

    pip install pytest
    python -m pytest


Linting
^^^^^^^

This code conforms to the PEP8 standard. Pylint can be used to analyse the code:

::

    pip install pylint
    pylint --rcfile=tests/pylintrc sksurgerysurfacematch


Installing
----------

You can pip install directly from the repository as follows:

::

    pip install git+https://github.com/UCL/scikit-surgerysurfacematch



Contributing
^^^^^^^^^^^^

Please see the `contributing guidelines`_.


Useful links
^^^^^^^^^^^^

* `Source code repository`_
* `Documentation`_


Licensing and copyright
-----------------------

Copyright 2020 University College London.
scikit-surgerysurfacematch is released under the BSD-3 license. Please see the `license file`_ for details.


Acknowledgements
----------------

Supported by `Wellcome`_ and `EPSRC`_.


.. _`Wellcome EPSRC Centre for Interventional and Surgical Sciences`: http://www.ucl.ac.uk/weiss
.. _`source code repository`: https://github.com/UCL/scikit-surgerysurfacematch
.. _`Documentation`: https://scikit-surgerysurfacematch.readthedocs.io
.. _`SNAPPY`: https://weisslab.cs.ucl.ac.uk/WEISS/PlatformManagement/SNAPPY/wikis/home
.. _`University College London (UCL)`: http://www.ucl.ac.uk/
.. _`Wellcome`: https://wellcome.ac.uk/
.. _`EPSRC`: https://www.epsrc.ac.uk/
.. _`contributing guidelines`: https://github.com/UCL/scikit-surgerysurfacematch/blob/master/CONTRIBUTING.rst
.. _`license file`: https://github.com/UCL/scikit-surgerysurfacematch/blob/master/LICENSE

