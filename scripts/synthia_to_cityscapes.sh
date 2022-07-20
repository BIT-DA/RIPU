# SYNTHIA -> Cityscapes ra, deeplabv3+, 5%
python train.py -cfg configs/synthia/deeplabv3plus_r101_RA.yaml OUTPUT_DIR results/v3plus_synthia_ra_5.0_precent

# SYNTHIA -> Cityscapes ra, deeplabv2, 2.2%
python train.py -cfg configs/synthia/deeplabv2_r101_RA.yaml OUTPUT_DIR results/v2_synthia_ra_2.2_precent

# SYNTHIA -> Cityscapes pa, deeplabv2, 40 pixels
python train.py -cfg configs/synthia/deeplabv2_r101_PA.yaml OUTPUT_DIR results/v2_synthia_pa_40_pixels


