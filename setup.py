# coding=utf-8
"""
Setup for scikit-surgerysurfacematch
"""

from setuptools import setup, find_packages
import versioneer

# Get the long description
with open('README.rst') as f:
    long_description = f.read()

setup(
    name='scikit-surgerysurfacematch',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description='Tools for reconstructing and matching surfaces represented as point clouds.',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    url='https://github.com/UCL/scikit-surgerysurfacematch',
    author='Matt Clarkson',
    author_email='m.clarkson@ucl.ac.uk',
    license='BSD-3 license',
    classifiers=[
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Developers',
        'Intended Audience :: Healthcare Industry',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',


        'License :: OSI Approved :: BSD License',


        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',

        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
    ],

    keywords='medical imaging',

    packages=find_packages(
        exclude=[
            'doc',
            'tests',
        ]
    ),

    install_requires=[
        'numpy>=1.11',
        'ipykernel',
        'nbsphinx',
        'scikit-surgeryimage',
        'scikit-surgerycalibration',
        'scikit-surgeryopencvcpp',
        'scikit-surgerypclcpp>=0.3.0'
    ],

    entry_points={
        'console_scripts': [
            'sksurgerysurfacematch=sksurgerysurfacematch.__main__:main',
        ],
    },
)
