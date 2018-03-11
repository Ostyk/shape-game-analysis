# shape-game-analysis
This is a Deep learning project I did as part of a recruitment process for Unit9.


### What is this repository for? ###
* 1.0
* Prove the feasibility of creating a shape drawing game for kids

### How do I get set up? ###

* Python 3.6.3
* Numpy,pandas,keras, matplotlib (all in the latest version as of 25/01/2018)


### My approach ###
The first thing that struck me the most was how on earth should I get the data? 
I needed to prove a point and reassure the board of Unit 9 that creating such a game would be feasible.
At first I thought of traditional Machine learning techniques. However, this is an image classification task and since I do not have all the time in the world, I decided to go with Deep Learning, namely CNNs.

These are some of the key points I thought could possibly influence the board's decision.

* Metrics such as accuracy
* Development time
* Whether kids would actually download such an app

### Guide if you want to test the entire repo ###
* First, delete the folder data and run the file 'shape_generator'
* Then train the model by running CNN_Model, where you can adjust the abtch size, epochs and other aspects of DL.
* Time to train will vary based on computing power
* Refer to present.py and test_model.py to see the end result of the model
* The three .png files are only used in test_model.py, feel free to change them to see the prediction outcomes

### End result ###
If you aren't a tech savy type of person then please refer to test_model.py to easily see how the model works 

However, if you are than please do check out the repo and escially the file called 'present.py' which will provide you with some intresting insights about the model.
