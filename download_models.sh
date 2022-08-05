#!/bin/bash

set -euo pipefail

## links from https://learnopencv.com/super-resolution-in-opencv/
# move into models/ dir

curl -O  https://raw.githubusercontent.com/Saafke/EDSR_Tensorflow/master/models/EDSR_x3.pb

curl -O https://raw.githubusercontent.com/Saafke/EDSR_Tensorflow/master/models/EDSR_x2.pb
curl -O https://raw.githubusercontent.com/Saafke/EDSR_Tensorflow/master/models/EDSR_x3.pb
curl -O https://raw.githubusercontent.com/Saafke/EDSR_Tensorflow/master/models/EDSR_x4.pb

curl -O https://raw.githubusercontent.com/fannymonori/TF-ESPCN/master/export/ESPCN_x2.pb
curl -O https://raw.githubusercontent.com/fannymonori/TF-ESPCN/master/export/ESPCN_x3.pb
curl -O https://raw.githubusercontent.com/fannymonori/TF-ESPCN/master/export/ESPCN_x4.pb

# curl -O https://raw.githubusercontent.com/Saafke/FSRCNN_Tensorflow/master/models/FSRCNN-small_x2.pb
# curl -O https://raw.githubusercontent.com/Saafke/FSRCNN_Tensorflow/master/models/FSRCNN-small_x3.pb
# curl -O https://raw.githubusercontent.com/Saafke/FSRCNN_Tensorflow/master/models/FSRCNN-small_x4.pb
curl -O https://raw.githubusercontent.com/Saafke/FSRCNN_Tensorflow/master/models/FSRCNN_x2.pb
curl -O https://raw.githubusercontent.com/Saafke/FSRCNN_Tensorflow/master/models/FSRCNN_x3.pb
curl -O https://raw.githubusercontent.com/Saafke/FSRCNN_Tensorflow/master/models/FSRCNN_x4.pb

curl -O https://raw.githubusercontent.com/fannymonori/TF-LapSRN/master/export/LapSRN_x2.pb
curl -O https://raw.githubusercontent.com/fannymonori/TF-LapSRN/master/export/LapSRN_x4.pb
curl -O https://raw.githubusercontent.com/fannymonori/TF-LapSRN/master/export/LapSRN_x8.pb
