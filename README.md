# Common Carotid Artery Segmentation

## Introduction

Cardiovascular disease (CVD) stands as the leading cause of mortality globally, as outlined by the World Health Organization (WHO, 2009). Factors contributing to CVD include smoking, obesity, hypertension, and an imbalanced serum lipid profile. Accurate diagnosis and evaluation of therapy and clinical data are imperative for assessing cerebrovascular and cardiovascular pathologies.

In recent years, carotid atherosclerosis has garnered significant attention among researchers due to its association with a high risk of thrombosis generation and subsequent cerebral emboli, which can lead to fatal outcomes. Precise segmentation of the common carotid artery (CCA) is crucial for understanding its geometry, aiding in the assessment and management of carotid atherosclerosis.

## Dataset
The dataset has been taken from [Mendeley.data](https://data.mendeley.com/datasets/d4xt63mgjm/1) published on November 2022 by Agata Momot. The dataset comprises ultrasound images of the common carotid artery. It includes images from 11 subjects, with each subject undergoing examinations on both left and right sides. The images total 1100, with each subject having 100 images. Additionally, the dataset contains expert masks created by a technician and validated by an expert.

## Training Results

The best performance in common carotid artery segmentation was achieved using a variant of the Transformer model called Patchers, published in 2022. The Patchers model has demonstrated remarkable efficacy in accurately delineating the CCA.
#### Metrics


| Model Name            |  Dice Score | IoU Index | Tversky Score |
|-----------------------|-------------|-----------|---------------|
|    UNet               |    0.9054        |  0.8556         |       0.9279        |
| [IterNet](https://github.com/conscienceli/IterNet)|      0.8496       |    0.7461        |    0.8982           |
| [MetaPolyp](https://github.com/huyquoctrinh/MetaPolyp-CBMS2023)|  0.9332            |    0.8753        |  0.9264              |
|Transformer - Patchers - without patchification |   0.9314  |  0.877          |       0.936 |
|Transformer - Patchers -  with patchification|   95.59  |     91.86       |         95.06           |

## Test Images
<img src="image1.jpg" alt="Image 1" width="100" height="100"> <img src="image2.jpg" alt="Image 2" width="100" height="100"> <img src="image3.jpg" alt="Image 3" width="100" height="100">



## Usage

This repository provides resources for common carotid artery segmentation, including code implementations, datasets, and model architectures. Researchers and practitioners in the field of medical imaging and cardiovascular health can leverage these resources to enhance their understanding of carotid atherosclerosis and develop more effective diagnostic and therapeutic strategies.

## Note

This project beats most scores acheived on the Common Carotid Artery Segmentation, and thus could be used as a benchmark metrics. And for inquiries, collaborations, or further information, please contact at samadnajm.sn@gmail.com or atbinmogharabin@gmail.com

## Acknowledgments

We would like to express our gratitude to the contributors and researchers who have made this project possible. Their dedication and efforts have contributed to advancing the field of common carotid artery segmentation and cardiovascular health.



## License

This project is licensed under the MIT License.

---
