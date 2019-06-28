# Master's Thesis:  Domain Adaptation in Wireless Capsule Endoscopy Diagnosis

## Author: Èlia Ficapal Vila

This project is carried out collaboratively with the company CorporateHealth. The medical images are under a confidentially agreement and thus will not be uploaded.  


### Abstract
Convolutional neural networks have been proved to reach excellent results in image classification, but also to require large amounts of data. Unfortunately, there are plenty of domains and applications where the availability of labelled datasets is limited. For instance, privacy issues and the small amount of tests performed leads to low availability of data in many medical image analysis problems. This obstacle is faced by a current project developed in collaboration with the company Corporate Health, which aims to make diagnosis out of the frames obtained from the Wireless Capsule Endoscopy (WCE) procedure.

This thesis will focus on finding a way to train models that generalise well to unseen target domains. In particular, images from different sources will be used for training and testing. Triplet loss is the secret weapon that will play a key role in improving the results obtained from the domain adaptation experiments.

The achievements of this work cover both training an algorithm that classifies accurately enough the images from the target domain, as well as proving that triplet loss is indeed helpful for domain adaptation in our specific scenario.


### This repository
The link of the GitHub repository is: https://github.com/eliaficapalvila/Domain-Adaptation-in-Wireless-Capsule-Endoscopy-Diagnosis.git

The code included is divided in the following folders:
* images: contains the images used in the notebooks.
* model: includes three .py files which are used to read the data, build and run the models and visualise the results.
* res_experiments: contains the results obtained evaluating the models.
* resnet_TL: contains the ResNet implementation and all the necessary functions to use the triplet loss.

The library SenseTheFlow developed by Guillem Pascual will be also used.

#### Notebooks
The following notebooks the visualisation of the results that are included in this thesis. The first two notebooks are experiments using the ResNet with Cross-Entropy Loss and ResNet with Triplet Loss, respectively, trained and evaluated using images from PillCam SB2. The last to notebooks are experiments using the same models but trained using SB2 images and (or not) SB3 images, and tested using SB3 images.

[ResNet with Cross-Entropy Loss model trained and tested with PillCam SB2 images](1_CE_onSB2_experiment.ipynb)

[ResNet with Triplet Loss model trained and tested with PillCam SB2 images](2_TL_onSB2_experiment.ipynb)

[ResNet with Cross-Entropy Loss model trained using SB2 images and (or not) SB3 images and tested on SB3](3_CE_onSB3_experiment.ipynb)

[ResNet with Triplet Loss model trained using SB2 images and (or not) SB3 images and tested on SB3](4_TL_onSB3_experiment.ipynb)
