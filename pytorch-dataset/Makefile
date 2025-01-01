.PHONY: download-food101 download-food101-full

FOOD101_SMALL_ZIP_PATH:=pizza_steak_sushi.zip
FOOD101_FULL_ARCHIVE_PATH:=food-101.tar.gz
DATA_PATH:=data/
FOOD101_SMALL_DATA_PATH:=$(DATA_PATH)/pizza_steak_sushi
FOOD101_FULL_DATA_PATH:=$(DATA_PATH)/food101

download-food101:
	/bin/test -f $(FOOD101_SMALL_ZIP_PATH) || wget wget -O $(FOOD101_SMALL_ZIP_PATH) https://github.com/mrdbourke/pytorch-deep-learning/raw/refs/heads/main/data/pizza_steak_sushi.zip
	/bin/test -d $(FOOD101_SMALL_DATA_PATH) || mkdir -p $(FOOD101_SMALL_DATA_PATH)
	/bin/test -d $(FOOD101_SMALL_DATA_PATH)/train || unzip $(FOOD101_SMALL_ZIP_PATH) -d $(FOOD101_SMALL_DATA_PATH)

download-food101-full:
	/bin/test -f $(FOOD101_FULL_ARCHIVE_PATH) || wget -O $(FOOD101_FULL_ARCHIVE_PATH) http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
	/bin/test -d $(FOOD101_FULL_DATA_PATH) || mkdir -p $(FOOD101_FULL_DATA_PATH)
	/bin/test -d $(FOOD101_FULL_DATA_PATH)/images || tar xvzf $(FOOD101_FULL_ARCHIVE_PATH)  --strip-components=1 -C $(FOOD101_FULL_DATA_PATH)
