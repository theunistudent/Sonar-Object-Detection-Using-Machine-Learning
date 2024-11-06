This code was used for the compleation of my under gratudate thesis on underwater object detection, using multibeam sonar and machine learning. 
The code has been made public incase someone finds it usefull for their own project. The models are built using PyTorch and attempt to train multiclass binary classifiers.

Unfortunatly due to File size constraints the data used for training could not be included
in the repo. Some of the the training data wased from the Shipwreck-Dataset:

https://www.lirmm.fr/shipwreck-dataset/
Nicolas Pecheux, Vincent Creuze, Frédéric Comby et Olivier Tempier, “Self-Calibration of a Sonar–Vision System for Underwater Vehicles: A New Method and a Dataset”, Sensors, 23(3), 1700, Feb. 3, 2023.

A link to the primary data and thesis paper will be added soon.


Requirments:
matplotlib
numpy
pandas
Pillow
opencv-python
torch
torchvision
scikit-learn


Model Training
The model training process is built on PyTorch, selected for its flexibility, community support, and user-friendly interface.

DataLoader Design
Custom Dataset Classes: The SonarImages class processes still images, while SonarVideo manages video frames, with both classes applying data transformations (grayscale conversion, normalization, random flips, and Gaussian noise) to enhance model generalization.
Combined Dataset: CombinedSonarDataset merges SonarImages and SonarVideo into a single dataset for more efficient training, allowing random data selection and multi-threading. 
Training Loop
The training loop uses a DataLoader to fetch random data batches. Models with sequence dependencies (e.g., those with LSTM layers) are supplied with frame sequences. Binary cross-entropy loss and Adam optimizer are used to backpropagate errors.


Metrics
Performance metrics recorded at each epoch and stored in CSV files.

CNN Models and Architecture
Initial models were binary classifiers for detecting man-made objects. Each CNN block in the architecture applies convolution, batch normalization, ReLU activation, and dropout layers. Multiple configurations were tested (e.g., 1-layer, 4-layer, 5-layer, and 18-layer CNNs).

Unsupervised Models
A stretch goal involved using a U-Net architecture for unsupervised segmentation to detect anomalies in images. Although time constraints prevented full implementation, this approach remains an area of interest.
