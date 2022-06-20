from flask import *
import os
from werkzeug.utils import secure_filename
from keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__)

# Classes of trafic signs
classes = { 0:'Limita de viteza (20km/h)',
            1:'Limita de viteza (30km/h)',
            2:'Limita de viteza (50km/h)',
            3:'Limita de viteza (60km/h)',
            4:'Limita de viteza (70km/h)',
            5:'Limita de viteza (80km/h)',
            6:'Sfaritul zonei cu limita 80km/h',
            7:'Limita de viteza (100km/h)',
            8:'Limita de viteza (120km/h)',
            9:'Depasirea interzisa',
            10:'Depasirea interzisa a autovehiculelor cu peste 3.5 tone',
            11:'Prioritate',
            12:'Drum cu prioritate',
            13:'Cedeaza trecerea',
            14:'Stop',
            15:'Accesul interzis vehiculelor',
            16:'Accesul interzis vehiculelor cu peste 3.5 tone',
            17:'Intrarea interzisa',
            18:'Atentie sporita',
            19:'Curba periculoasa la stanga',
            20:'Curba periculoasa la dreapta',
            21:'Curba dubla',
            22:'Drum cu denivelari',
            23:'Drum alunecos',
            24:'Drumul se ingusteaza pe dreapta',
            25:'Drum in lucru',
            26:'Semafor',
            27:'Pietoni',
            28:'Copii care traversează',
            29:'Traversarea bicicletelor',
            30:'Atenție la gheață/zăpadă',
            31:'Atentie animale',
            32:'Sfarsit restrictii',
            33:'Virare la dreapta',
            34:'Virare la stanga',
            35:'Doar înainte',
            36:'In fata sau la dreapa',
            37:'In fata sau la stanga',
            38:'La dreapta',
            39:'La stanga',
            40:'Sens giratoriu',
            41:'Sfarsitul zonei unde depasirea e interzisa',
            42:'Sfarsitul interzicerii trecerii vehiculelor de peste 3.5 tone' }

def image_processing(img):
    model = load_model('./model/model_tsr.h5')
    data=[]
    image = Image.open(img)
    image = image.resize((30,30))
    data.append(np.array(image))
    X_test=np.array(data)
    Y_pred = model.predict_classes(X_test)
    return Y_pred

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        file_path = secure_filename(f.filename)
        f.save(file_path)
        # Make prediction
        result = image_processing(file_path)
        s = [str(i) for i in result]
        a = int("".join(s))
        result = "Semnul de circulatie este: " +classes[a]
        os.remove(file_path)
        return result
    return None

if __name__ == '__main__':
    app.run(debug=True)