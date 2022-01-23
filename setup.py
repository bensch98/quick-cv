from setuptools import setup

with open('README.md', 'r') as fh:
  long_description = fh.read()

CLASSIFIERS = [
  'Programming Language :: Python :: 3',
  'License :: OSI Approved :: MIT License'
]

INSTALL_REQUIREMENTS = [
  'opencv-python',
  'numpy'
]

setup(name='quick-cv',
      version='0.0.1',
      description='Quick way to test out computer vision functions and practices with camera, e.g. webcam',
      author='Benedikt Scheffler',
      author_email='scheffler.benedikt@gmail.com',
      license='MIT',
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages=['quick-cv'],
      classifiers=CLASSIFIERS,
      install_requirements=INSTALL_REQUIREMENTS,
      python_require='>=3.9')
