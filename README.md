# Speech-Accent-Recognition
A deep learning model is developed which can predict the native country on the basis of the spoken english accent


1. Overview:

	Using audio samples from [The Speech Accent Archive] (http://accent.gmu.edu/), we wanted to show that a deep neural network can classify the countries of english residence of a speaker.

2. Dependencies:

  	* Python 3.5 (https://www.python.org/download/releases/2.7/)
 	  * Keras (https://keras.io/)
  	* Numpy (http://www.numpy.org/)
  	* BeautifulSoup (https://www.crummy.com/software/BeautifulSoup/)
   	* Pydub (https://github.com/jiaaro/pydub)
  	* Sklearn (http://scikit-learn.org/stable/)
  	* Librosa (http://librosa.github.io/librosa/)

3. Data:

	We started with the data from The Speech Accent Archive, a collection of more than 2400 audio samples from people for over 177 countries speaking the same English paragraph. 	The paragraph contains most of the consonants, vowels, and clusters of standard American English.

4. Model:

•	Converted wav audio files into Mel Frequency Cepstral Coefficients graph.

•	The MFCC was fed into a 2-Dimensional Convolutional Neural Network (CNN) to predict the native language class.

5. Challenges & Solutions:

•	Computationally expensive: Uses only native english origin for a smaller subset of 645 speakers

• Small dataset: MFCCs were sliced into smaller segments. These smaller segments were fed into the neural network where predictions were made. Using an ensembling method, a majority vote was taken to predict the native language class.


6. Running Model:
  
  ├── src   
        ├── accuracy.py
        ├── fromwebsite.py
        ├── getaudio.py
        ├── getsplit.py
        ├── trainmodel.py
  ├── models  
       ├── cnn_model138.h5
  ├── logs  
       ├── events.out.tfevents.1506987325.ip-172-31-47-225
  └── audio

Note- Run all the python files as described below on the terminal


1. Run getaudio.py to download audio files to the audio directory. All audio files listed in bio_metadata.csv will be downloaded. Use nativeenglish.csv for this model

Command: `python getaudio.py nativeenglish.csv` 

###### To filter audio samples to feed into the CNN:

1. Edit the filter_df method in getsplit.py
    * This will filter audio files from the csv when calling trainmodel.py

###### To make predictions on audio files:

1. Run trainmodel.py to train the CNN

Command: `python trainmodel.py nativeenglish.csv milestonemodel`

  * Running trainmodel.py will save the trained model as milestonemodel.h5 in the model directory and output the results to the console.
  * This script will also save a TensorBoard log file into the logs directory.

7. Performance

	With the inconsistent labels and smaller set of 645 speakers, the model was able to predict the correct country of english residence at around 73% accuracy when given a test sample from the 645 native english speakers.

Epoch 00062: early stopping
WARNING:tensorflow:From /content/drive/My Drive/CS230_SAR/src/accuracy.py:12: Sequential.predict_classes (from tensorflow.python.keras.engine.sequential) is deprecated and will be removed after 2021-01-01.
Instructions for updating:
Please use instead:* `np.argmax(model.predict(x), axis=-1)`,   if your model does multi-class classification   (e.g. if it uses a `softmax` last-layer activation).* `(model.predict(x) > 0.5).astype("int32")`,   if your model does binary classification   (e.g. if it uses a `sigmoid` last-layer activation).
Training samples: Counter({"['usa']": 328, "['uk']": 34, "['canada']": 33, "['australia']": 30, "['ireland']": 11, "['uk', 'usa']": 9, "['new zealand']": 5, "['usa', 'uk']": 4, "['canada', 'usa']": 3, "['uk,usa']": 3, "['usa', 'canada']": 3, "['south africa']": 3, "['singapore']": 3, "['guyana', 'usa']": 2, "['australia', 'usa']": 2, "['jamaica', 'usa']": 2, "['philippines,canada']": 1, "['uk', 'new zealand']": 1, "['New Zealand', 'Australia']": 1, "['antigua and barbuda', 'usa']": 1, "['belize', 'usa']": 1, "['ghana', 'usa']": 1, "['jamaica', 'uk']": 1, "['australia', 'singapore']": 1, "['scotland', 'usa', 'sierra leone']": 1, "['northern ireland']": 1, "['australia', 'hong kong', 'usa']": 1, "['uk', 'canada']": 1, "['liberia', 'usa']": 1, "['the bahamas']": 1, "['uk', 'singapore']": 1, "['USA']": 1, "['uk', 'australia']": 1, "['papua new guinea,uk,usa']": 1, "['nigeria', 'usa']": 1, "['scotland', 'usa']": 1, "['us virgin islands']": 1, "['singapore,usa']": 1, "['jamaica']": 1, "['barbados']": 1, "['uk,canada']": 1, "['fiji', 'usa']": 1, "['canada', 'uk']": 1, "['panama', 'usa']": 1, "['australia', 'uk', 'usa']": 1, "['uk', 'australia', 'usa']": 1, "['philippines']": 1, "['usa', 'new zealand']": 1, "['uk', 'ireland', 'usa']": 1, "['australia', 'canada']": 1, "['india', 'usa']": 1, "['isle of man']": 1, "['jamaica,usa']": 1, "['australia,usa']": 1, "['canada', 'usa', 'uk', 'cayman islands']": 1, "['canada', 'ireland']": 1, "['ausstralia']": 1})
Testing samples: Counter({"['usa']": 94, "['uk']": 10, "['canada']": 8, "['australia']": 5, "['south africa']": 1, "['new zealand']": 1, "['australia', 'usa']": 1, "['jamaica']": 1, "['wales', 'australia']": 1, "['ireland']": 1, "['new zealand', 'ireland', 'uk', 'usa']": 1, "['uk', 'canada', 'usa']": 1, "['canada', 'usa']": 1, "['singapore']": 1, "['trinidad,usa']": 1, "['scotland']": 1})
Accuracy to beat: 0.7286821705426356
Confusion matrix of total samples:
 [ 0  8  0  0  0  0  0  1  0  0  1  0  0  1  1  0  1  0  0  0  0  0  0  0
  0  0  1  0  1  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  1  0  0  0
  0  0  5  0  0  1  1  0 10  0  0  0 94  1]
Confusion matrix:
 [[ 0  0  0 ...  0  0  0]
 [ 0  0  0 ...  0  8  0]
 [ 0  0  0 ...  0  0  0]
 ...
 [ 0  0  0 ...  0  0  0]
 [ 0  0  0 ...  0 94  0]
 [ 0  0  0 ...  0  1  0]]
Accuracy: 0.7286821705426356


