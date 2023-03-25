# GTAV -> Cityscapes ra, deeplabv3+, 5%
python train.py -cfg configs/gtav/deeplabv3plus_r101_RA.yaml OUTPUT_DIR results/v3plus_gtav_ra_5.0_precent

# GTAV -> Cityscapes ra, deeplabv2, 2.2%
python train.py -cfg configs/gtav/deeplabv2_r101_RA.yaml OUTPUT_DIR results/v2_gtav_ra_2.2_precent

# GTAV -> Cityscapes pa, deeplabv2, 40 pixels
python train.py -cfg configs/gtav/deeplabv2_r101_PA.yaml OUTPUT_DIR results/v2_gtav_pa_40_pixel

# [source-free scenario] GTAV -> Cityscapes ra, deeplabv2, 2.2%
python train_source.py -cfg configs/gtav/deeplabv2_r101_src.yaml OUTPUT_DIR results/source_free/gtav
python train_source_free.py -cfg configs/gtav/deeplabv2_r101_RA_source_free.yaml OUTPUT_DIR results/v2_gtav_ra_2.2_precent_source_free resume results/source_free/gtav/model_iter020000.pth
