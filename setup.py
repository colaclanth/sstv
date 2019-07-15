from setuptools import setup

with open("README.md", "r") as readme:
    long_description = readme.read()

setup(name="sstv",
      version="0.1",
      description="SSTV audio file decoder",
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="http://github.com/colaclanth/sstv",
      author="colaclanth",
      author_email="nomail@c0l.me",
      license="GPLv3",
      packages=["sstv"],
      keywords="sstv decode",
      entry_points={"console_scripts": ["sstv=sstv.__main__:main"]},
      install_requires=[
          'numpy',
          'Pillow',
          'PySoundFile',
          'scipy'
      ],
      classifiers=[
          'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
          'Programming Language :: Python :: 3',
      ])
