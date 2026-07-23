# Project_CIDL

## DESCRIPTION	
	Ifrit — a CNN-based forest fire detection system that recognizes wildfires from images, aimed at low-resource / embedded deployment. 
	Project for Computational Intelligence and Deep Learning (Master's in AI and Data Engineering, University of Pisa).

	**Goal**: 
	Build a custom convolutional network for binary Fire / Non-Fire classification that is both accurate and small enough to run on constrained hardware (e.g. Jetson Nano, Google Coral), where a false negative (missed fire) is treated as far more costly than a false positive.

	**Dataset cleaning (a core contribution)**
	The starting Kaggle dataset (~5,050 images) was noisy and only loosely related to the actual problem. I performed an in-depth, manual curation: removing 391 corrupted files, then discarding 779 images incongruent with forest-fire detection (snow-capped mountains, cities, beaches, fireworks, city fires, satellite/rendered images, etc.), each categorized with an explicit rationale. I also analyzed the per-class RGB color distribution to check for and mitigate the obvious bias (the network learning "lots of red = fire"), verifying the distribution stayed balanced after cleaning. Resizing used AREA interpolation, chosen after empirically trading ~30% extra preprocessing time for higher image quality.

	**Model design & optimization**: 
	Starting from AlexNet and GoogLeNet as baselines, I designed a custom network family, IfritNet (v1–v4), iteratively evolving the architecture — introducing inception modules and 1×1 convolutions — to shrink the network while preserving or improving accuracy. Hyperparameters were tuned rigorously: 10-fold cross-validation over every (batch size, early-stopping patience) pair, then Keras Hyperband tuning for learning rate and dropout.

	**Results**: 
	The final IfritNet v4 reaches 98.17% test accuracy with 0.65% false negatives using only ~74k parameters and a 1.22 MB model — about 99.7% fewer parameters than AlexNet (which reached ~95% accuracy at ~189 MB), while being both more accurate and drastically lighter.

	**Tech stack**: 
	Python, TensorFlow/Keras, OpenCV, Keras Tuner. Trained across an RTX 3060, a GTX 1080, and Google Colab (T4), with a comparative analysis of each.

## DATASET
	The dataset used for the application can be found on kaggle at the following link: 
	https://www.kaggle.com/datasets/mohnishsaiprasad/forest-fire-images

## Project Structured
The folder contains:
 - 3 python files: Fire_cls_GUI (main file, GUI), AlexNet_class, GoogLeNet class, IfritNet class (my own CNN)
 - Dataset: folder containing the dataset images used for training and testing networks in the project. This folder is divided into:
	– old ds: contains the original dataset except for the corrupted images that have been removed;
	– new ds: contains the dataset after the image analysis and selection phase, it does not contain the images that I found not congruent with the problem addressed.
 - Model: folder containing the saved model of the CNN trained during the project. This folder is divided into:
	– train hdf5: Contains checkpoint saves of models made during network training phases and other test saves;
	– best model: contains the saves of the best models.
 - result test CNN: folder containing a record of all network training done for the project. It is divided into subfolders, one for each type of CNN seen in the project. 
   There is data related to workouts done with GTX 1080, RTX 3060, and Google Colab. The training reports are both in notepad form (text lines) and saved images (accuracy and loss trends and confusion matrix).
 - Others folder: for future implementations, utilitys or CNN experiments in Python
 - CIDL_Documentation: documentation regarding the application and the analysis carried out for the choice of model and parameters.

## Developer's notes
	The work related to the university examination has been done and the project is completed. 
	There may be updates or improvements to the project in the future, but nothing is planned for now.

## Developers:
	- Alessandro Diana
