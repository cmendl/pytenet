from setuptools import setup


with open('README.rst', 'r') as f:
    # skip the banners
    lines = f.readlines()[6:]
    long_desc = ''.join(lines)

setup(
    name='pytenet',
    version='1.2.0',
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
    python_requires='>=3.6'
)
