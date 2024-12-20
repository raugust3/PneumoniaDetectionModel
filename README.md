# PneumoniaDetectionModel
Course Project for Introduction to Machine Learning (DS3000) at Western University.

Project Members (Group 10):
- Richard Augustine (raugust3@uwo.ca)
- Hasan Kaware (hkaware@uwo.ca)
- Ritwick Vemula (rvemula4@uwo.ca)
- Saifeldin Hafez (ncadmann@uwo.ca)
- Nathanael Cadman-Neu (shafez2@uwo.ca)

Instructions for Running the Code

1 - **Download the Pre-trained Weights**
- Open the file weight_link.doc in this repository.
- Use the link inside the document to download the pre-trained model weights.
- Save the downloaded file as pneumonia_detection_model.pth in the same directory as the repository files.

``model_path="pneumonia_detection_model.pth"``

2 - **Install Dependencies**

Run the following command in your terminal:
pip install torch torchvision scikit-learn matplotlib seaborn pillow

3 - **Execute test.py**
To test the model or make predictions, execute the test.py file:

``python test.py``

The script will:
- Load the pre-trained weights.
- Evaluate the model on the test dataset (if configured).
- Predict labels and confidence scores for images.

4 - **View Prediction**
By default, the script uses a sample image (img.png) provided in the repository for single-image prediction.
``image_path = "img.png"``

To test with a different image:
- Update the image_path variable in the test.py script with the path to your desired image:
``image_path = "your_image_name.png"``
- Make sure the new image is in the same directory as the script, or provide a full path to the image.

Note: Each time the model is trained, the resulting curves may vary due to the random initialization and stochastic optimization inherent in the training process. Additionally, if the main script detects an existing pre-trained model, it may load this model, resulting in completely different outputs that do not reflect the current training process. For reference, a supplementary video has been included in the repository to demonstrate how the metrics were obtained.

