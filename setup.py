from setuptools import setup
import re


with open('ReadMe.md', 'r') as f:
    long_desc = f.read()
    # cut out the build status banner
    long_desc = re.sub('[![Build Status](.*?)](.*?)\n\n', '', long_desc)

setup(
    name='pytenet',
    version='1.0',
    description='Tensor network library for quantum simulations',
    long_description=long_desc,
    license='BSD 2-Clause',
    author='Christian B. Mendl',
    author_email='christian.b.mendl@gmail.com',
    url='https://github.com/cmendl/pytenet',
    packages=['pytenet'],
    install_requires=[
        'numpy>=1.9',
        'scipy>=1.0.0',
    ],
    extras_require={
        'test': ['unittest']
    },
    python_requires='>=2.7'
)
