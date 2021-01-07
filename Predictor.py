from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from flask import Flask, render_template , request , jsonify
from PIL import Image
import os , io , sys 
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage

app = Flask(__name__)

UPLOAD_FOLDER = '/home/solibot/Documents/CropCure/User_Queries/'

@app.route('/tea/classify' , methods=['GET','POST'])
def classification():
    
    if request.method == 'POST':
    
        disease_img = request.files['image']
        tper = disease_img.content_type
        print(tper)
        disease_img.save(os.path.join(UPLOAD_FOLDER, secure_filename(disease_img.filename)))

    class_names = ['Tea Blight', 'Tea Red Spot', 'Tea Red Scab']

    model = load_model('neural_network_model.h5')

    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    img_filename = secure_filename(disease_img.filename)
    img_dir = UPLOAD_FOLDER + img_filename
    img = load_img(img_dir, target_size=(150, 150))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img, axis=0)

    classes = np.argmax(model.predict(img_array), axis = -1)

    predicted_disease = [class_names[i] for i in classes]

    print("Predicted Tea Disease: ", predicted_disease[0])

    return jsonify({'response':predicted_disease[0]})


@app.route('/home' , methods=['GET','POST'])
def home():
	resp = 'Hey there! I can help you classify any kind of diseases on your crop. \nPlease click a picture of the leaf and send it right away!'
	return jsonify({'response':resp})

if __name__ == '__main__':
    app.run(host='0.0.0.0')