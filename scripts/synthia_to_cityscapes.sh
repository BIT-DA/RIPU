# SYNTHIA -> Cityscapes ra, deeplabv3+, 5%
python train.py -cfg configs/synthia/deeplabv3plus_r101_RA.yaml OUTPUT_DIR results/v3plus_synthia_ra_5.0_precent

# SYNTHIA -> Cityscapes ra, deeplabv2, 2.2%
python train.py -cfg configs/synthia/deeplabv2_r101_RA.yaml OUTPUT_DIR results/v2_synthia_ra_2.2_precent

# SYNTHIA -> Cityscapes pa, deeplabv2, 40 pixels
python train.py -cfg configs/synthia/deeplabv2_r101_PA.yaml OUTPUT_DIR results/v2_synthia_pa_40_pixels


# [source-free scenario] SYNTHIA -> Cityscapes ra, deeplabv2, 2.2%
python train_source.py -cfg configs/synthia/deeplabv2_r101_src.yaml OUTPUT_DIR results/source_free/synthia
python train_source_free.py -cfg configs/synthia/deeplabv2_r101_RA_source_free.yaml OUTPUT_DIR results/v2_synthia_ra_2.2_precent_source_free resume results/source_free/synthia/model_iter020000.pth