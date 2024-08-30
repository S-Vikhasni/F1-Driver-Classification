import os
import sys
import io
from flask import Flask, render_template, request, url_for
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import json

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

#template folder for html files, static folder for css files
app = Flask(__name__, template_folder=r'client', static_folder=r'client')


#load saved model
model_path = r'server\artifacts\saved_model.keras'
model = load_model(model_path)

#load saved dictionary
class_dict_path = r'server\artifacts\class_dictionary.json'

#function to read saved dictionary
def load_class_dictionary():
    try:
        with open(class_dict_path, 'r', encoding='utf-8') as f:
            class_dict = json.load(f)
        return class_dict
    except (UnicodeDecodeError, IOError) as e:
        print(f"Error loading class dictionary: {e}")
        raise

#create dictionary inverse
class_dict = load_class_dictionary()
class_dict_inv = {v: k for k, v in class_dict.items()}

#save predicted images to client folder
target_img = os.path.join(os.getcwd(), r'client')  
ALLOWED_EXT = {'jpeg', 'jpg', 'bmp', 'png'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

#resize image
def read_image(filename):
    img = load_img(filename, target_size=(256, 256))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    return x

@app.route('/')
def index_view():
    return render_template('index.html')

#save uploaded files
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        if request.method == 'POST':
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = file.filename
                file_path = os.path.join(target_img, filename)
                file.save(file_path)

                #prepare saved image
                img = read_image(file_path)
                class_prediction = model.predict(img)
                classes_x = np.argmax(class_prediction, axis=1)

                #use innverse dictionary to get driver name from number
                driver_index = classes_x[0]  
                driver = class_dict_inv.get(driver_index, "Unknown")  

                #send prediction to predict.html
                user_image = filename  

                return render_template('predict.html', driver=driver, prob=class_prediction.tolist(), user_image=user_image)
            else:
                return "Unable to read the file. Please check the file extension."
    except Exception as e:
        return f"An error occurred: {str(e)}"


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=8001)
