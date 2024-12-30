.PHONY: download-food101

FOOD101_SMALL_ZIP_PATH:=pizza_steak_sushi.zip
DATA_PATH:=data/
FOOD101_SMALL_DATA_PATH:=$(DATA_PATH)/pizza_steak_sushi

download-food101:
	/bin/test -f $(FOOD101_SMALL_ZIP_PATH) || wget wget -O $(FOOD101_SMALL_ZIP_PATH) https://github.com/mrdbourke/pytorch-deep-learning/raw/refs/heads/main/data/pizza_steak_sushi.zip
	/bin/test -d $(FOOD101_SMALL_DATA_PATH) || mkdir -p $(FOOD101_SMALL_DATA_PATH)
	/bin/test -d $(FOOD101_SMALL_DATA_PATH)/train || unzip $(FOOD101_SMALL_ZIP_PATH) -d $(FOOD101_SMALL_DATA_PATH)
