library(sf)
library(rgdal)
library(raster)
install.packages("readTIFF")

shp_2020<-readOGR("crown_shp/Crowns_2020_08_01_MergedWithPlotData.shp")
plot(shp_2020,add=TRUE)

bci_2020<- raster::raster("aerial_imagery/2020_08_01_BCI_50ha.tif")
plot(bci_2020)

values(bci_2020)<-ifelse(values(bci_2020)==250,NA,values(bci_2020))
