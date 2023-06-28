# Skin-Lesion-Detection-for-Skin-Cancer-using-CNN
A CNN model which predicts the 7 types of skin lesions, some of which are related to cancer. It uses HAMM10000 dataset

DATA COLLECTION :- 
HAM10000 is a dataset collection containing more than 10000 images relating to skin lesions of 7 different kinds. These lesions are indicators or symptoms of specific types of skin cancer or a benign skin related disease like sun-burns. 
Dataset was released by Harvard University in Harvard Dataverse. There are options to add our own data to the collection. It is released to public in interest of people coming up with better algorithms to detect skin cancer. HAM stands for Human against Machine, which is the whole point of making ML/DL models to cross them against humans and find a better way of diagnosis of fatal diseases.
The 7 different classes of skin lesions are:-
1.	 Actinic keratoses and intraepithelial carcinoma / Bowen's disease 
2.	Basal cell Carcinoma (bcc)
3.	Benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses (bkl) (not cancers)
4.	dermatofibroma (df)
5.	melanoma (mel)
6.	Melanocytic nevi (nv) 
7.	Vascular lesions (angiomas, angiokeratomas)

DATA PREPROCESSING :- 
I accessed the pixel of each data in grayscale already available from the csv file, in Kaggle. Each Image is a 28X28 size which gives a total of 784 pixels from pixel0000 to pixel0783. These belong to 7 different classes. Highest is class 4 or Melanoma (mel) at 6705 examples, lowest is class or dermatofibroma(df) at115 . My goal is to create a dataset that is balanced in both cases. Best way to under-sample the majority class and oversample the minority class to meet at a junction.
Method Used :- 
SMOTE-Tomeklink :- The SMOTE-Tomeklink method is a combination of two techniques used for addressing imbalanced datasets in machine learning: Synthetic Minority Over-sampling Technique (SMOTE) and Tomek links. Imbalanced datasets refer to situations where the classes or categories of the target variable are not represented equally, leading to a skewed distribution. This can pose challenges for machine learning algorithms, as they tend to perform poorly on underrepresented classes.
Apply SMOTE: The Synthetic Minority Over-sampling Technique (SMOTE) is applied to oversample the minority class. SMOTE works by creating synthetic samples in the feature space of the minority class. It selects a minority class instance and finds its k nearest neighbors. It then generates synthetic samples by interpolating between the selected instance and its neighbors. This process is repeated until the desired balance between the minority and majority classes is achieved. Calculate Tomek Links: After applying SMOTE and oversampling the minority class, Tomek links are calculated. Tomek links are pairs of instances from different classes that are closest to each other. These instances represent potentially misclassified or overlapping samples. Tomek links can be detected by computing the distance matrix between all instances and identifying pairs that satisfy the Tomek link criteria. A Tomek link exists between two instances if they belong to different classes and are each other's nearest neighbors. 
Remove Majority Class Instances: In the SMOTE-Tomeklink method, the majority class instances involved in Tomek links are removed. By removing these instances, the overlapping regions or potentially misclassified samples are eliminated, resulting in better separation between the minority and majority classes.
I used the method ‘not majority’ for tomeklinks which applies the method on all the classes except majority there by reducing data loss.
Standardization :- After the dataset was balanced I standardised the feature pixels to remove any problems arising from outliers, as in this case outliers are important to medical diagnosis.

CNN ARCHITECTURE :- 
The CNN architecture consists of 4 Convolutional layers and 4 Dense layers, with input being a array of shape (None, 28,28,1) and output being a array of 7 floating numbers representing the probability of an image belonging to one of the 7 classes.
Convolutional Layers : - They are arranged with filters from (16 – 128) with each filter size of (2,2) each padding such that we get same dimension as input image in each filter and strides of size (1,1). Each Convolution layer is followed by a max-pooling layer and a batch normalization layer. Max-pooling layer has a size of (1,1). Batch Normalization after each layer (even in dense) ensures that there is uniformity in each layer outputs. 
Dense Layers :- Model also has 4 dense layers ranging from (128-7) units in each layer in decreasing order. Each dense layer has ReLU activation function and a dropout of 20% to compensate for model overfitting. Each layer output is also batch-normalized. 
The output layer has 7 neurons and a Softmax activation function to determine the probability of each Image belonging to which Class.
Model is compiled with Adam as the optimizer and Sparse Categorical Cross-entropy as the loss function. I used accuracy for metrics. 

MODEL TRAINING :-
I trained the model for 100 epochs with Early-Stopping method. 
Early-Stopping is used when we need to stop the model training at its optimum value.
I used Early stopping to monitor the validation loss, Why validation loss?
So that we can stop at the best weights, where our model gives out a lowest output. I set the patience to 10, which means that if for 10 epochs the validation loss doesn’t reduce by certain amount then the Model training stops returning the best weights in the executed epochs.
A validation split of 20-30% is optimum for the training I went with 20% and a test size of 25%.
MODEL VALIDATION:-
Testing and Validation go hand in hand, they can be interchangeable. In both cases we use an unseen dataset on the model to test its accuracy.
At a test size of 25% my model provided an accuracy of 91% ~ 90.932%
From the Classification report we can conclude that precision or TP classification is well above the acceptance level for all classes. Only class 4 is shy at 82%. 


CONCLUSION :-
For a dataset highly imbalanced as HAM10000, the focus of achieving a broad spectrum classification at 91% accuracy is an optimum goal.
I tried to improve the accuracy using sample weights and class weights method to focus more on the classes that were being ignored, but all the methods reduced the accuracy due to higher attention given to dataset of higher classes and the imbalance between the classes being so vast (6705 as highest and 115 as lowest). Hence I arrived at SMOTE-Tomeklink Method which gave the most accurate model.
I also used GAN method, but as it contains 10000 images is in the lower amount for the data needed, when I resampled the data all classes got 6075 samples each. For GAN it is very less. Which led to me doing more epochs while training GAN, another constraint I faced for GAN was the training time which exceeded my GPU memory.
Hence I finally used the regular CNN architecture. RNN and LSTM are not the best as they are focused on time series of a data, and I did not use a time series data.



