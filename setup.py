#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import find_packages, setup

setup(
    version='0.1.0',
    name='openpredict',
    license='MIT License',
    description='An API to compute and serve predictions of biomedical concepts associations via OpenAPI for the NCATS Translator project',
    author='Vincent Emonet',
    author_email='vincent.emonet@maastrichtuniversity.nl',
    url='https://github.com/MaastrichtU-IDS/translator-openpredict',
    packages=find_packages(),
    # packages=find_packages(include=['openpredict']),
    # package_dir={'openpredict': 'openpredict'},
    package_data={'': ['openapi.yml', 'data/models/*', 'data/ontology/*', 'data/models/*', 
        'data/features/*', 'data/input/*', 'data/sparql/*', 'data/resources/*', 'data/*.ttl',
        'tests/queries/*', 'data/embedding/*']},
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'openpredict=openpredict.__main__:main',
        ],
    },

    python_requires='>=3.8',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    install_requires=open("requirements.txt", "r").readlines(),
    tests_require=['pytest==5.2.0', 'reasoner-validator>=2.1.0'],
    setup_requires=[],
    # setup_requires=['pytest-runner'],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8"
    ]
)
