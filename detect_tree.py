#import the function, find the code and create the class
from detectron2.engine.defaults import DefaultPredictor
from scripts.tiling2 import tile_data
from scripts.train import MyTrainer, setup_cfg
from scripts.predict import predict_on_data
from scripts.project import project_to_geojson, stitch_crowns, clean_crowns

import rasterio
#import image
img= "aerial_imagery/2020_08_01_BCI_50ha.tif"
site_path= "C:/Users/P_pol/repo/AutoCrown"
tiles_path=site_path + "/tiles/"
data=rasterio.open(img)
## Tile the images
buffer = 30
tile_width = 40
tile_height = 40
tile_data(data, tiles_path, buffer, tile_width, tile_height, dtype_bool = True)
## import the model

tree = "./230103_randresize_full.pth"
cfg = setup_cfg(update_model=tree)
predict_on_data(tiles_path, DefaultPredictor(cfg))

project_to_geojson(tiles_path, tiles_path + "predictions/", tiles_path + "predictions_geo/")

crowns = stitch_crowns(tiles_path + "predictions_geo/", 1)
crowns = crowns[crowns.is_valid]
crowns = clean_crowns(crowns, 0.6) 
crowns.to_file(site_path + "/crowns_out.gpkg")