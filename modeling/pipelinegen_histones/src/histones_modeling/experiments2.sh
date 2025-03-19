#!/bin/bash
## Single experiment training
python embc2024.py -t 20221214_HP_HBS_CTH -t 20230929_2_BSA_Hapt_2kPEG \
    -t 20230525_Hapt_2kPeg_H4 \
    -t 20240108_CTH_Hapt_2kPEG_24hr \
    -t 20230526_Hapt_24_H4 \
    -t 0230518_H4_Rod \
    -t 20230518_H4_2kPEG \
    --run_name  2KPEG_NoPolish_Train 
python embc2024.py -t 20221214_HP_HBS_CTH -t 20230929_2_BSA_Hapt_2kPEG \
    -t 20230519_H2kPEG_H4_NoPolish \
    -t 20240108_CTH_Hapt_2kPEG_24hr \
    -t 20230526_Hapt_24_H4 \
    -t 0230518_H4_Rod \
    -t 20230518_H4_2kPEG \
    --run_name  2KPEG_Hapt_Train 
python embc2024.py -t 20221214_HP_HBS_CTH  -t 20230929_2_BSA_Hapt_2kPEG \
    -t 20230518_CTH_2kPEG \
    -t 20230929_1_BSA_Hapt_2kPEG \
    -t 20230519_H2kPEG_H4_NoPolish \
    -t 20230525_Hapt_2kPeg_H4 \
    -t 0230518_H4_Rod \
    -t 20230518_H4_2kPEG \
    --run_name  24Hr_2KPEG_Train 
python embc2024.py -t 20221214_HP_HBS_CTH -t 20230929_2_BSA_Hapt_2kPEG \
    -t 20230519_H2kPEG_H4_NoPolish \
    -t 20240108_CTH_Hapt_2kPEG_24hr \
    -t 20230525_Hapt_2kPeg_H4 \
    -t 20230526_Hapt_24_H4 \
    -t 0230518_H4_Rod \
    --run_name  2KPEG_Train 
