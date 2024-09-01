# F1-Driver-Classification

![f1 first](https://github.com/user-attachments/assets/fb432fdf-dadf-4018-9723-b44bcd6e0ce6)

![f1 second](https://github.com/user-attachments/assets/10871fd3-83ef-44cd-8c5e-f56473c31c74)

The images of 5 drivers ( Lewis Hamilton, Max Verstappen, Charles Leclerc, Alex Albon, Zhou Guanyu) were imported through data scraping, and Python libraries such asOpenCV, TensorFlow, numpy, and matplotlib were used for preprocessing and normalization.Then a Convolutional Neural Network (CNN) model was built to analyze and predict outcomes based on the image data.

Next, A Python Flask server was developed to integrate the trained CNN model, enabling it to handle HTTP requests. This server processed incoming requests efficiently, providing  predictions based on the input data.

Finally, the UI was designed using HTML, CSS, and JavaScript where users are to upload a driver’s image, which was then processed by the Flask server. The system returns the input image, the predicted driver’s name, and the associated probabilities.


