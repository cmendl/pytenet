from setuptools import setup


with open('ReadMe.rst', 'r') as f:
    # skip the build status banner
    for n in range(4):
        next(f)
    long_desc = f.read()

setup(
    name='pytenet',
    version='1.0',
    description='Tensor network library for quantum simulations',
    long_description=long_desc,
    long_description_content_type='text/x-rst',
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
