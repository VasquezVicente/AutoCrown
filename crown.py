##Import modules
import rasterio
from rasterio.plot import show
from rasterio import plot as rasterplot
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import mapping
from osgeo import gdal

##project manually deliniated crowns for BCI 2020 august
fp = r'aerial_imagery/2020_08_01_BCI_50ha.tif'
img = rasterio.open(fp)
shapefile = gpd.read_file("crown_shp/Crowns_2020_08_01_MergedWithPlotData.shp")
img.crs
shapefile.crs
shapefile = shapefile.to_crs('epsg:32617')
fig, ax=plt.subplots()
rasterplot.show(img, ax=ax)
shapefile.plot(facecolor="none",ax=ax)
plt.show()

##predict crowns with detectron model, possible overfittin since they mention 50ha in tutorial
##requirements of torch (pytorch) which does not currently support python 3.11, employed interpreter 3.9 . 
# $py -3.9 -m pip install torch, then $pip install git+https://github.com/PatBall1/detectree2.git
#detectree2 has many dependencies and conflicts between versions that i havent been able to resolve
#further examination of the package and its dependencies say linux is neccesary for detectron2 or macos

#installing  linux might be better

import torch
import gdal

from setuptools import find_packages, setup

setup(
    name="detectree2",
    version="0.0.1",
    author="James G. C. Ball",
    author_email="ball.jgc@gmail.com",
    description="Detectree packaging",
    url="https://github.com/PatBall1/detectree2",
    # package_dir={"": "detectree2"},
    packages=find_packages(),
    test_suite="detectree2.tests.test_all.suite",
    install_requires=[
        "pyyaml==5.1",
        "GDAL>=1.11",
        "proj",
        "geos",
        "pypng",
        "pygeos",
        "geopandas",
        "rasterio>=3.1.0",
        "fiona",
        "pycrs",
        "descartes",
        "detectron2@git+https://github.com/facebookresearch/detectron2.git",
    ],
)
python setup.py build_ext -I<C:\Users\P_pol\AppData\Local\Programs\Python\pkgs\libgdal-3.0.2-ha1b3edf_1\Library\include> -lgdal_i -L<C:\Users\P_pol\AppData\Local\Programs\Python\Library\lib> install --gdalversion 3.0.2

$ pip install --no-use-pep517 --global-option -I<C:\Users\P_pol\AppData\Local\Programs\Python\pkgs\libgdal-3.0.2-ha1b3edf_1\Library\include> -lgdal_i -L<C:\Users\P_pol\AppData\Local\Programs\Python\Library\lib> .
pip install --no-use-pep517 --global-option -I<C:\Users\P_pol\AppData\Local\Programs\Python\pkgs\libgdal-3.0.2-ha1b3edf_1\Library\include> -lgdal_i -L<C:\Users\P_pol\AppData\Local\Programs\Python\Library\lib>