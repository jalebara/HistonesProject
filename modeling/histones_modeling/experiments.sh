#!/bin/bash

# Default Tests
# Train with a single experiment -- Test with the rest
python embc2024.py -t 20221214_HP_HBS_CTH -t 20230929_2_BSA_Hapt_2kPEG \
    -t 20230525_Hapt_2kPeg_H4 \
    -t 20230526_Hapt_24_H4 \
    -t 0230518_H4_Rod \
    -t 20230518_H4_2kPEG \
    -t 20240318_Histones_BSA_Test_set \
    -t 20240318_Histones_BSA_1 \
    -t 0240318_Histones_BSA_3 \
    -t 0240318_Histones_BSA_2 \
    --run_name  2KPEG_NoPolish_Train 
python embc2024.py -t 20221214_HP_HBS_CTH -t 20230929_2_BSA_Hapt_2kPEG \
    -t 20230519_H2kPEG_H4_NoPolish -t 20240318_Histones_BSA_Test_set \
    -t 20230526_Hapt_24_H4 \
    -t 0230518_H4_Rod \
    -t 20230518_H4_2kPEG \
    -t 20240318_Histones_BSA_1 \
    -t 0240318_Histones_BSA_3 \
    -t 0240318_Histones_BSA_2 \
    --run_name  2KPEG_Hapt_Train 
python embc2024.py -t 20221214_HP_HBS_CTH  -t 20230929_2_BSA_Hapt_2kPEG \
    -t 20230519_H2kPEG_H4_NoPolish -t 20240318_Histones_BSA_Test_set \
    -t 20230525_Hapt_2kPeg_H4 \
    -t 0230518_H4_Rod \
    -t 20230518_H4_2kPEG \
    -t 20240318_Histones_BSA_1 \
    -t 0240318_Histones_BSA_3 \
    -t 0240318_Histones_BSA_2 \
    --run_name  24Hr_2KPEG_Train 
python embc2024.py -t 20221214_HP_HBS_CTH  -t 20230929_2_BSA_Hapt_2kPEG \
    -t 20230519_H2kPEG_H4_NoPolish -t 20240318_Histones_BSA_Test_set \
    -t 20230525_Hapt_2kPeg_H4 \
    -t 20230526_Hapt_24_H4 \
    -t 20230518_H4_2kPEG \
    -t 20240318_Histones_BSA_1 \
    -t 0240318_Histones_BSA_3 \
    -t 0240318_Histones_BSA_2 \
    --run_name  Rod_Train 
python embc2024.py -t 20221214_HP_HBS_CTH -t 20230929_2_BSA_Hapt_2kPEG \
    -t 20230519_H2kPEG_H4_NoPolish -t 20240318_Histones_BSA_Test_set \
    -t 20230525_Hapt_2kPeg_H4 \
    -t 20230526_Hapt_24_H4 \
    -t 0230518_H4_Rod \
    -t 20240318_Histones_BSA_1 \
    -t 0240318_Histones_BSA_3 \
    -t 0240318_Histones_BSA_2 \
    --run_name  2KPEG_Train 
python embc2024.py -t 20221214_HP_HBS_CTH -t 20230929_2_BSA_Hapt_2kPEG \
    -t 20230519_H2kPEG_H4_NoPolish -t 20240318_Histones_BSA_Test_set \
    -t 20230525_Hapt_2kPeg_H4 \
    -t 20230526_Hapt_24_H4 \
    -t 0230518_H4_Rod \
    -t 20230518_H4_2kPEG \
    --run_name  Mixed_Media_Train 

# Test with a single experiment -- Train with the rest
python embc2024.py -t 20221214_HP_HBS_CTH -t 20230929_2_BSA_Hapt_2kPEG \
    -t 20230519_H2kPEG_H4_NoPolish  \
    --run_name  2KPEG_NoPolish_Test
python embc2024.py -t 20221214_HP_HBS_CTH -t 20230929_2_BSA_Hapt_2kPEG \
    -t 20230525_Hapt_2kPeg_H4 \
    --run_name  2KPEG_Hapt_Test 
python embc2024.py -t 20221214_HP_HBS_CTH  -t 20230929_2_BSA_Hapt_2kPEG \
    -t 20230526_Hapt_24_H4 \
    --run_name  24Hr_2KPEG_Test
python embc2024.py -t 20221214_HP_HBS_CTH  -t 20230929_2_BSA_Hapt_2kPEG \
    -t 0230518_H4_Rod \
    --run_name  Rod_Test
python embc2024.py -t 20221214_HP_HBS_CTH -t 20230929_2_BSA_Hapt_2kPEG \
    -t 20230518_H4_2kPEG \
    --run_name  2KPEG_Test 
python embc2024.py -t 20221214_HP_HBS_CTH -t 20230929_2_BSA_Hapt_2kPEG \
    -t 20240318_Histones_BSA_Test_set \
    -t 20240318_Histones_BSA_1 \
    -t 0240318_Histones_BSA_3 \
    -t 0240318_Histones_BSA_2 \
    --run_name  Mixed_Media_Test


## Magnitude Controlled Experiments

# Train with a single experiment -- Test with the rest
python embc2024.py -t 20221214_HP_HBS_CTH -t 20230929_2_BSA_Hapt_2kPEG \
    -t 20230525_Hapt_2kPeg_H4 -t 20240318_Histones_BSA_Test_set\
    -t 20230526_Hapt_24_H4 \
    -t 0230518_H4_Rod \
    -t 20230518_H4_2kPEG \
    -t 20240318_Histones_BSA_1 \
    -t 0240318_Histones_BSA_3 \
    -t 0240318_Histones_BSA_2 \
    --flatten \
    --run_name  2KPEG_NoPolish_Train_Mag_Control
python embc2024.py -t 20221214_HP_HBS_CTH -t 20230929_2_BSA_Hapt_2kPEG \
    -t 20230519_H2kPEG_H4_NoPolish -t 20240318_Histones_BSA_Test_set \
    -t 20230526_Hapt_24_H4 \
    -t 0230518_H4_Rod \
    -t 20230518_H4_2kPEG \
    -t 20240318_Histones_BSA_1 \
    -t 0240318_Histones_BSA_3 \
    -t 0240318_Histones_BSA_2 \
    --flatten \
    --run_name  2KPEG_Hapt_Train_Mag_Control
python embc2024.py -t 20221214_HP_HBS_CTH  -t 20230929_2_BSA_Hapt_2kPEG \
    -t 20230519_H2kPEG_H4_NoPolish -t 20240318_Histones_BSA_Test_set \
    -t 20230525_Hapt_2kPeg_H4 \
    -t 0230518_H4_Rod \
    -t 20230518_H4_2kPEG \
    -t 20240318_Histones_BSA_1 \
    -t 0240318_Histones_BSA_3 \
    -t 0240318_Histones_BSA_2 \
    --flatten \
    --run_name  24Hr_2KPEG_Train_Mag_Control
python embc2024.py -t 20221214_HP_HBS_CTH  -t 20230929_2_BSA_Hapt_2kPEG \
    -t 20230519_H2kPEG_H4_NoPolish -t 20240318_Histones_BSA_Test_set \
    -t 20230525_Hapt_2kPeg_H4 \
    -t 20230526_Hapt_24_H4 \
    -t 20230518_H4_2kPEG \
    -t 20240318_Histones_BSA_1 \
    -t 0240318_Histones_BSA_3 \
    -t 0240318_Histones_BSA_2 \
    --flatten \
    --run_name  Rod_Train_Mag_Control
python embc2024.py -t 20221214_HP_HBS_CTH -t 20230929_2_BSA_Hapt_2kPEG \
    -t 20230519_H2kPEG_H4_NoPolish -t 20240318_Histones_BSA_Test_set \
    -t 20230525_Hapt_2kPeg_H4 \
    -t 20230526_Hapt_24_H4 \
    -t 0230518_H4_Rod \
    -t 20240318_Histones_BSA_1 \
    -t 0240318_Histones_BSA_3 \
    -t 0240318_Histones_BSA_2 \
    --flatten \
    --run_name  2KPEG_Train_Mag_Control
python embc2024.py -t 20221214_HP_HBS_CTH -t 20230929_2_BSA_Hapt_2kPEG \
    -t 20230519_H2kPEG_H4_NoPolish -t 20240318_Histones_BSA_Test_set \
    -t 20230525_Hapt_2kPeg_H4 \
    -t 20230526_Hapt_24_H4 \
    -t 0230518_H4_Rod \
    -t 20230518_H4_2kPEG \
    --run_name  Mixed_Media_Train_Mag_Control

# Test with a single experiment -- Train with the rest
python embc2024.py -t 20221214_HP_HBS_CTH -t 20230929_2_BSA_Hapt_2kPEG \
    -t 20230519_H2kPEG_H4_NoPolish  \
    --flatten \
    --run_name  2KPEG_NoPolish_Test_Mag_Control
python embc2024.py -t 20221214_HP_HBS_CTH -t 20230929_2_BSA_Hapt_2kPEG \
    -t 20230525_Hapt_2kPeg_H4   \
    --flatten \
    --run_name  2KPEG_Hapt_Test_Mag_Control
python embc2024.py -t 20221214_HP_HBS_CTH  -t 20230929_2_BSA_Hapt_2kPEG \
    -t 20230526_Hapt_24_H4  \
    --flatten \
    --run_name  24Hr_2KPEG_Test_Mag_Control
python embc2024.py -t 20221214_HP_HBS_CTH  -t 20230929_2_BSA_Hapt_2kPEG \
    -t 0230518_H4_Rod  \
    --flatten \
    --run_name  Rod_Test_Mag_Control
python embc2024.py -t 20221214_HP_HBS_CTH -t 20230929_2_BSA_Hapt_2kPEG \
    -t 20230518_H4_2kPEG -t 20240318_Histones_BSA_Test_set \
    --flatten \
    --run_name  2KPEG_Test_Mag_Control
python embc2024.py -t 20221214_HP_HBS_CTH -t 20230929_2_BSA_Hapt_2kPEG \
    -t 20240318_Histones_BSA_Test_set \
    -t 20240318_Histones_BSA_1 \
    -t 0240318_Histones_BSA_3 \
    -t 0240318_Histones_BSA_2 \
    --flatten \
    --run_name  Mixed_Media_Test_Mag_Control

### Downsampled Experiments
# Train with a single experiment -- Test with the rest
python embc2024.py --downsample 2 -t 20221214_HP_HBS_CTH -t 20230929_2_BSA_Hapt_2kPEG \
    -t 20230525_Hapt_2kPeg_H4 \
    -t 20230526_Hapt_24_H4 \
    -t 0230518_H4_Rod \
    -t 20230518_H4_2kPEG \
    -t 20240318_Histones_BSA_Test_set \
    -t 20240318_Histones_BSA_1 \
    -t 0240318_Histones_BSA_3 \
    -t 0240318_Histones_BSA_2 \
    --run_name Downsampled_2KPEG_NoPolish_Train 
python embc2024.py --downsample 2 -t 20221214_HP_HBS_CTH -t 20230929_2_BSA_Hapt_2kPEG \
    -t 20230519_H2kPEG_H4_NoPolish -t 20240318_Histones_BSA_Test_set \
    -t 20230526_Hapt_24_H4 \
    -t 0230518_H4_Rod \
    -t 20230518_H4_2kPEG \
    -t 20240318_Histones_BSA_1 \
    -t 0240318_Histones_BSA_3 \
    -t 0240318_Histones_BSA_2 \
    --run_name Downsampled_2KPEG_Hapt_Train 
python embc2024.py --downsample 2 -t 20221214_HP_HBS_CTH  -t 20230929_2_BSA_Hapt_2kPEG \
    -t 20230519_H2kPEG_H4_NoPolish -t 20240318_Histones_BSA_Test_set \
    -t 20230525_Hapt_2kPeg_H4 \
    -t 0230518_H4_Rod \
    -t 20230518_H4_2kPEG \
    -t 20240318_Histones_BSA_1 \
    -t 0240318_Histones_BSA_3 \
    -t 0240318_Histones_BSA_2 \
    --run_name Downsampled_24Hr_2KPEG_Train 
python embc2024.py --downsample 2 -t 20221214_HP_HBS_CTH  -t 20230929_2_BSA_Hapt_2kPEG \
    -t 20230519_H2kPEG_H4_NoPolish -t 20240318_Histones_BSA_Test_set \
    -t 20230525_Hapt_2kPeg_H4 \
    -t 20230526_Hapt_24_H4 \
    -t 20230518_H4_2kPEG \
    -t 20240318_Histones_BSA_1 \
    -t 0240318_Histones_BSA_3 \
    -t 0240318_Histones_BSA_2 \
    --run_name Downsampled_Rod_Train 
python embc2024.py --downsample 2 -t 20221214_HP_HBS_CTH -t 20230929_2_BSA_Hapt_2kPEG \
    -t 20230519_H2kPEG_H4_NoPolish -t 20240318_Histones_BSA_Test_set \
    -t 20230525_Hapt_2kPeg_H4 \
    -t 20230526_Hapt_24_H4 \
    -t 0230518_H4_Rod \
    -t 20240318_Histones_BSA_1 \
    -t 0240318_Histones_BSA_3 \
    -t 0240318_Histones_BSA_2 \
    --run_name Downsampled_2KPEG_Train 
python embc2024.py  --downsample 2 -t 20221214_HP_HBS_CTH -t 20230929_2_BSA_Hapt_2kPEG \
    -t 20230519_H2kPEG_H4_NoPolish -t 20240318_Histones_BSA_Test_set \
    -t 20230525_Hapt_2kPeg_H4 \
    -t 20230526_Hapt_24_H4 \
    -t 0230518_H4_Rod \
    -t 20230518_H4_2kPEG \
    --run_name  Downsampled_Mixed_Media_Train 

# Test with a single experiment -- Train with the rest
python embc2024.py --downsample 2 -t 20221214_HP_HBS_CTH -t 20230929_2_BSA_Hapt_2kPEG \
    -t 20230519_H2kPEG_H4_NoPolish \
    --run_name Downsampled_2KPEG_NoPolish_Test
python embc2024.py --downsample 2 -t 20221214_HP_HBS_CTH -t 20230929_2_BSA_Hapt_2kPEG \
    -t 20230525_Hapt_2kPeg_H4  \
    --run_name Downsampled_2KPEG_Hapt_Test 
python embc2024.py --downsample 2 -t 20221214_HP_HBS_CTH  -t 20230929_2_BSA_Hapt_2kPEG \
    -t 20230526_Hapt_24_H4  \
    --run_name Downsampled_24Hr_2KPEG_Test
python embc2024.py --downsample 2 -t 20221214_HP_HBS_CTH  -t 20230929_2_BSA_Hapt_2kPEG \
    -t 0230518_H4_Rod  \
    --run_name Downsampled_Rod_Test
python embc2024.py --downsample 2 -t 20221214_HP_HBS_CTH -t 20230929_2_BSA_Hapt_2kPEG \
    -t 20230518_H4_2kPEG  \
    --run_name Downsampled_2KPEG_Test 
python embc2024.py --downsample 2  -t 20221214_HP_HBS_CTH -t 20230929_2_BSA_Hapt_2kPEG \
    -t 20240318_Histones_BSA_Test_set \
    -t 20240318_Histones_BSA_1 \
    -t 0240318_Histones_BSA_3 \
    -t 0240318_Histones_BSA_2 \
    --run_name  Downsampled_Mixed_Media_Test


## Magnitude Controlled and Downsampled Experiments

# Train with a single experiment -- Test with the rest
python embc2024.py --downsample 2 -t 20221214_HP_HBS_CTH -t 20230929_2_BSA_Hapt_2kPEG \
    -t 20230525_Hapt_2kPeg_H4 \
    -t 20230526_Hapt_24_H4 \
    -t 0230518_H4_Rod \
    -t 20230518_H4_2kPEG \
    -t 20240318_Histones_BSA_Test_set \
    -t 20240318_Histones_BSA_1 \
    -t 0240318_Histones_BSA_3 \
    -t 0240318_Histones_BSA_2 \
    --flatten \
    --run_name Downsampled_2KPEG_NoPolish_Train_Mag_Control
python embc2024.py --downsample 2 -t 20221214_HP_HBS_CTH -t 20230929_2_BSA_Hapt_2kPEG \
    -t 20230519_H2kPEG_H4_NoPolish -t 20240318_Histones_BSA_Test_set \
    -t 20230526_Hapt_24_H4 \
    -t 0230518_H4_Rod \
    -t 20230518_H4_2kPEG \
    -t 20240318_Histones_BSA_1 \
    -t 0240318_Histones_BSA_3 \
    -t 0240318_Histones_BSA_2 \
    --flatten \
    --run_name Downsampled_2KPEG_Hapt_Train_Mag_Control
python embc2024.py --downsample 2 -t 20221214_HP_HBS_CTH  -t 20230929_2_BSA_Hapt_2kPEG \
    -t 20230519_H2kPEG_H4_NoPolish -t 20240318_Histones_BSA_Test_set \
    -t 20230525_Hapt_2kPeg_H4 \
    -t 0230518_H4_Rod \
    -t 20230518_H4_2kPEG \
    -t 20240318_Histones_BSA_1 \
    -t 0240318_Histones_BSA_3 \
    -t 0240318_Histones_BSA_2 \
    --flatten \
    --run_name Downsampled_24Hr_2KPEG_Train_Mag_Control
python embc2024.py --downsample 2 -t 20221214_HP_HBS_CTH  -t 20230929_2_BSA_Hapt_2kPEG \
    -t 20230519_H2kPEG_H4_NoPolish -t 20240318_Histones_BSA_Test_set \
    -t 20230525_Hapt_2kPeg_H4 \
    -t 20230526_Hapt_24_H4 \
    -t 20230518_H4_2kPEG \
    -t 20240318_Histones_BSA_1 \
    -t 0240318_Histones_BSA_3 \
    -t 0240318_Histones_BSA_2 \
    --flatten \
    --run_name Downsampled_Rod_Train_Mag_Control
python embc2024.py --downsample 2 -t 20221214_HP_HBS_CTH -t 20230929_2_BSA_Hapt_2kPEG \
    -t 20230519_H2kPEG_H4_NoPolish -t 20240318_Histones_BSA_Test_set \
    -t 20230525_Hapt_2kPeg_H4 \
    -t 20230526_Hapt_24_H4 \
    -t 0230518_H4_Rod \
    -t 20240318_Histones_BSA_1 \
    -t 0240318_Histones_BSA_3 \
    -t 0240318_Histones_BSA_2 \
    --flatten \
    --run_name Downsampled_2KPEG_Train_Mag_Control
python embc2024.py  --downsample 2 -t 20221214_HP_HBS_CTH -t 20230929_2_BSA_Hapt_2kPEG \
    -t 20230519_H2kPEG_H4_NoPolish -t 20240318_Histones_BSA_Test_set \
    -t 20230525_Hapt_2kPeg_H4 \
    -t 20230526_Hapt_24_H4 \
    -t 0230518_H4_Rod \
    -t 20230518_H4_2kPEG \
    --flatten \
    --run_name  Downsampled_Mixed_Media_Train_Mag_Control

# Test with a single experiment -- Train with the rest
python embc2024.py --downsample 2 -t 20221214_HP_HBS_CTH -t 20230929_2_BSA_Hapt_2kPEG \
    -t 20230519_H2kPEG_H4_NoPolish \
    --flatten \
    --run_name Downsampled_2KPEG_NoPolish_Test
python embc2024.py --downsample 2 -t 20221214_HP_HBS_CTH -t 20230929_2_BSA_Hapt_2kPEG \
    -t 20230525_Hapt_2kPeg_H4  \
    --flatten \
    --run_name Downsampled_2KPEG_Hapt_Test 
python embc2024.py --downsample 2 -t 20221214_HP_HBS_CTH  -t 20230929_2_BSA_Hapt_2kPEG \
    -t 20230526_Hapt_24_H4  \
    --flatten \
    --run_name Downsampled_24Hr_2KPEG_Test
python embc2024.py --downsample 2 -t 20221214_HP_HBS_CTH  -t 20230929_2_BSA_Hapt_2kPEG \
    -t 0230518_H4_Rod  \
    --flatten \
    --run_name Downsampled_Rod_Test
python embc2024.py --downsample 2 -t 20221214_HP_HBS_CTH -t 20230929_2_BSA_Hapt_2kPEG \
    -t 20230518_H4_2kPEG  \
    --flatten \
    --run_name Downsampled_2KPEG_Test 
python embc2024.py --downsample 2  -t 20221214_HP_HBS_CTH -t 20230929_2_BSA_Hapt_2kPEG \
    -t 20240318_Histones_BSA_Test_set \
    -t 20240318_Histones_BSA_1 \
    -t 0240318_Histones_BSA_3 \
    -t 0240318_Histones_BSA_2 \
    --flatten \
    --run_name  Downsampled_Mixed_Media_Test_Mag_Control

