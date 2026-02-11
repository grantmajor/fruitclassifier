# Fruit Classifier

Fruit Classifier is a convolutional neural network (CNN) trained on the [fruits-360](https://www.kaggle.com/datasets/moltean/fruits) dataset.

Docker Repo: https://hub.docker.com/repository/docker/grantmajor/fruitclassifier/general

## Project Specifics

<details>
  <summary>Data Augmentation</summary>
  Mimicking common data augmentation practices to improve generalization and reduce overfitting, we apply a series of transformations to each training image. First we resize the image to 64x64, followed by randomly flipping approximately 50% of images along the horizontal axis. Next, we apply a random rotation transformation that rotates each training image by approximately 0-10 degrees. Finally, we apply a color jitter transformation that slightly alters the brightness, contrast, and saturation of the images to simulate variability in lighting conditions. After these transformations are complete, we convert the image into a 3 channel tensor and send it to the CNN.
</details>   

<details>
  <summary>Training</summary>
  The CNN was trained using a cross-entropy loss function with label smoothing applied. AdamW was selected as the optimizer due to its quick convergence and wide-spread adoption throughout the deep learning community. A learning rate of 0.001 was selected and a cosine annealing scheduler was used to improve convergence rate and generalization. Fruit Classifier was trained for 50 total epochs with a patience of 6; its lowest loss was seen at epoch 47. The model took approximately 1 hour and 45 minutes to train on an NVIDIA GTX 1080 with a final total of 1,636,737 parameters.
</details>   


<details>
  <summary>Validation</summary> 
  The validation loop was fairly standard, with multiple variables being used to track relevant metrics like macro-f1, per-class accuracy, top-k accuracy, and loss. These variables allowed the creation of the plots seen on the Model Metrics page. A confusion matrix was also generated, however, due to the large number of classes, its effectiveness is minimized.
</details>   

<details>
  <summary>Backend & Frontend</summary> 
  To allow for users to easily interact with the CNN, a front-end and back-end were created using HTML/CSS/JavaScript and Flask respectively. The back-end loads the model checkpoint from the project folder and creates a new FruitCNN object. Three unique webpages were created, with the predictions webpage featuring two separate routes. Upon running the application, the user is brought to a landing page that features a navigation bar to allow for swapping between webpages. After swapping to a different webpage, the corresponding HTML template is returned from Flask which, in turn, displays the new webpage. 
  The front-end is a typical implementation of HTML/CSS/JavaScript for a Flask API. A CSS file is used to improve webpage aesthetics. 
</details>   


## Getting Started

### Installing
First, clone the repository using Git
```
git clone https://github.com/grantmajor/fruitclassifier.git
```
Now, download the necessary libraries using Python's pip command.
```
pip install -r requirements.txt
```

### Executing program
We can execute the program using the app.py file:
```
python app.py
```

## Author

Grant Major

