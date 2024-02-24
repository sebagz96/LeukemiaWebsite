from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import pickle
from datetime import datetime
import random
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

# Configuracion de la base de datos
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///results.db'
db                                      = SQLAlchemy(app)
migrate                                 = Migrate(app, db)
app.config['TEMPLATES_AUTO_RELOAD']     = True

# Configuracion de la sesión
SESSION_TYPE        = 'SqlAlchemySessionInterface'
app.secret_key      ='acH86BA51O'

# Categorias a clasificar por los modelos
Categories = ['hem', 'all']
Categories_leukemia = ['ALL', 'CML', "CLL", "AML"]

# Cargar modelos entrenados con datasets obtenidos
model = pickle.load(open('img_model.pkl', 'rb')) # ESTE ES EL MODELO que clasifica HEM O ALL
modelo_leukemia = pickle.load(open('leukemia_model.pkl', 'rb'))

# Cargar el modelo que clasifica los tipos de leucemia linfoblastica aguda
modelo_cargado = tf.keras.models.load_model('modelokaggle1.h5')  # Este es el modelo que clasifica pre, pro, early, etc.
input_size = (224, 224)

# Clase Resultados
class Result(db.Model):
    id                      = db.Column(db.Integer, primary_key=True)
    identification          = db.Column(db.Integer)
    patient_first_name      = db.Column(db.String(100))
    patient_last_name       = db.Column(db.String(100))
    image_path              = db.Column(db.String(200))
    classification_result   = db.Column(db.String(50))
    registration_date       = db.Column(db.DateTime, default=datetime.utcnow)
    
# Clase Usuario
class User(db.Model):
    id              = db.Column(db.Integer, primary_key = True)
    username        = db.Column(db.String(200))
    password        = db.Column(db.String(100))
    email           = db.Column(db.String(100))

# Borrar cache al recargar la pagina para evitar reload
@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response

# Funcion para llamar al modelo encargado de detectar patrones benignos o malignos de LLA
def classify_image_with_model(image_path): # HEM O ALL
    img = imread(image_path)
    target_image_size = (100, 100, 3)
    img_resize = resize(img, target_image_size)
    img_array = img_resize.flatten().reshape(1, -1)

    probability = model.predict_proba(img_array)
    predicted_category = Categories[model.predict(img_array)[0]]
    return predicted_category

# Preprocesamiento y deteccion de etapas de la LLA
def cargar_y_preprocesar_imagen(ruta_imagen):
    img = image.load_img(ruta_imagen, target_size=input_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.vgg19.preprocess_input(img_array)
    return img_array

# Funcion que devuelve el resultado de deteccion por etapas
def clasificar_imagen(ruta_imagen): # PRE PRO EARLY O BENIGN
    img_array = cargar_y_preprocesar_imagen(ruta_imagen)
    prediction = modelo_cargado.predict(img_array)
    predicted_class = np.argmax(prediction)
    code = {0: "LLA - Benigno", 1: "LLA - Etapa Temprana", 2: "LLA - Pre-B", 3: "LLA - Pro-B"}
    predicted_class_name = code[predicted_class]
    return predicted_class_name


def classify_leukemia_image_with_model(image_path):
    img = imread(image_path)
    target_image_size = (100, 100, 3)
    img_resize = resize(img, target_image_size)
    img_array = img_resize.flatten().reshape(1, -1)

    probability = modelo_leukemia.predict_proba(img_array)
    predicted_category = Categories_leukemia[model.predict(img_array)[0]]
    return predicted_category

# Ruta base a la pagina principal
@app.route('/')
def landing_page():
    return render_template('landing.html')

# Ruta a la pagina de Login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        
        username = request.form['username']
        password = request.form['password']

        print(username, password)

        queryResult = User.query.filter_by(username=User.username)
        
        print(queryResult.username)

        if not queryResult:
            flash("Usuario no registrado!")
            return render_template('login.html')
        else:
            
            if username == queryResult.username:
                if queryResult and check_password_hash(queryResult.password, password):
                    session['user'] = username
                    return redirect(url_for('database'))
                else:
                    flash("Contraseña inválida!")
                    return redirect(url_for('login'))
            else:
                flash("Usuario incorrecto!")
                return redirect(url_for('login'))
            
    return render_template('login.html')

@app.route('/upload_leukemia', methods=['GET', 'POST'])
def upload_leukemia():
    if request.method == 'POST':
    
    # Proceso para crear un nombre aleatorio a la imagen
        letras          = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        numeros         = "0123456789"
        unir            = f'{letras}{numeros}'
        longitud        = 20
        nombreLista     = random.sample(unir, longitud)
        nombreImagen    = "".join(nombreLista)

        file                    = request.files['image']           
        filename                = secure_filename(file.filename)
        # Extension de la imagen
        extension               = os.path.splitext(filename)[1]
        nuevoFile               = f'{nombreImagen}{extension}'

        image_path              = os.path.join('static/uploads/', nuevoFile)
        file.save(image_path)

        result = classify_leukemia_image_with_model(image_path)

        current_date = datetime.utcnow()
        os.remove(image_path)
        new_result = Result(
            patient_first_name=request.form['first_name'],
            patient_last_name=request.form['last_name'],
            image_path=image_path,
            classification_result=result,
            registration_date=current_date
        )
        db.session.add(new_result)
        db.session.commit()

        if result in ['ALL', 'CML', "CLL", "AML"]:
            return redirect(url_for('result_leukemia', result=result))
        else:
            return redirect(url_for('upload_leukemia', result=result))
    return render_template("upload_leukemia.html")

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
       
        # Proceso para crear un nombre aleatorio a la imagen
        letras          = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        numeros         = "0123456789"
        unir            = f'{letras}{numeros}'
        longitud        = 20
        nombreLista     = random.sample(unir, longitud)
        nombreImagen    = "".join(nombreLista)

        file                    = request.files['image']           
        filename                = secure_filename(file.filename)
        # Extension de la imagen
        extension               = os.path.splitext(filename)[1]
        nuevoFile               = f'{nombreImagen}{extension}'

        image_path              = os.path.join('static/uploads/', nuevoFile)
        file.save(image_path)        
            
    
        result = classify_image_with_model(image_path)

        os.remove(image_path)
        new_result = Result(
            identification = request.form['identification'],
            patient_first_name=request.form['first_name'],
            patient_last_name=request.form['last_name'],
            image_path=image_path,
            classification_result=result
        )
        db.session.add(new_result)
        db.session.commit()

        if result in ['hem', 'all']:
            return redirect(url_for('result_hem_all', result=result))
        else:
            return redirect(url_for('upload_diagnostico', result=result))
    if 'user' in session:
        return render_template("upload_data.html")
    else:
        return render_template('404.html')

@app.route('/result_diagnostico', methods=['GET'])
def show_result_diagnostico():
    result = request.args.get('result')
    if 'user' in session:
        flash("Registro agregado satisfactoriamente!")
        return render_template('result_diagnostico.html', result=result)
    else:
        return render_template('404.html')

@app.route('/result_hem_all', methods=['GET'])
def result_hem_all():
    if 'user' in session:
        result = request.args.get('result')
        return render_template('result_hem_all.html', result=result)
    else:
        return render_template('404.html')


@app.route('/result_leukemia', methods=['GET'])
def result_leukemia():
    result = request.args.get('result')
    return render_template('result_leukemia.html', result=result)

@app.route('/database', methods=['GET'])
def database():
    if 'user' in session:
        results = Result.query.all()
        return render_template('database.html', results=results)
    else:
        return render_template('404.html')

@app.route('/upload_diagnostico', methods=['GET', 'POST'])
def upload_diagnostico():

    # Proceso para crear un nombre aleatorio a la imagen
    letras          = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    numeros         = "0123456789"
    unir            = f'{letras}{numeros}'
    longitud        = 20
    nombreLista     = random.sample(unir, longitud)
    nombreImagen    = "".join(nombreLista)
    

    if request.method == 'POST':
            
            file                    = request.files['image']           
            filename                = secure_filename(file.filename)
            # Extension de la imagen
            extension               = os.path.splitext(filename)[1]
            nuevoFile               = f'{nombreImagen}{extension}'

            image_path              = os.path.join('static/uploads/', nuevoFile)
            file.save(image_path)

            result_diagnostico      = clasificar_imagen(image_path)

           
            new_result = Result(
                identification = request.form['identification'],
                patient_first_name=request.form['first_name'],
                patient_last_name=request.form['last_name'],
                image_path=image_path,
                classification_result=result_diagnostico
            )
            db.session.add(new_result)
            db.session.commit()

            if result_diagnostico:
                
                return redirect(url_for('show_result_diagnostico', result=result_diagnostico))
            else:
                return redirect(url_for('upload_diagnostico', result=result_diagnostico))

    if 'user' in session:
        return render_template("upload_data_diagnostico.html")
    else:
        return render_template('404.html')
    
# Direccion a vista para ver resultados
@app.route('/info-paciente/<int:id>', methods = ['GET', 'POST'])
def vistaRegistro(id):
    if 'user' in session:
        if request.method == 'GET':
            resultData = Result.query.get(id)
            if resultData:
                return render_template('vistaRegistro.html', dataInfo = resultData)
            else:
                return render_template('404.html')
    else:
        return render_template('404.html')

# Eliminar registros
@app.route('/eliminar-registro/<int:id>', methods=['GET', 'POST'])
def eliminarRegistro(id):
    if 'user' in session:
        if request.method == 'GET':
            resultData = Result.query.get(id)           
            db.session.delete(resultData)
            db.session.commit()

            flash("Registro Eliminado!")
            
            return redirect(url_for('database'))

    else:
        return render_template('404.html')
    
@app.route('/component', methods=['GET', 'POST'])
def component():
    results = Result.query.all()
    return render_template('component.html', results = results)

@app.route('/registro', methods = ['GET', 'POST'])
def nuevoRegistro():
    if request.method == 'POST':

        username    = request.form['username']
        password    = request.form['password']
        email       = request.form['email']


        #Verificar si existe el registro
        user_result = User.query.get(email)

        if not user_result:

            hashed_password = generate_password_hash(password)
            newUser = User(
                username = username,
                password = hashed_password,
                email = email
            )

            db.session.add(newUser)
            db.session.commit()

            flash("Usuario agregado con éxito!")
            return render_template('login.html')
        
        else:
            flash("Este usuario ya esta registrado!")
            return render_template('register.html')
        
    return render_template('register.html')
    
@app.route('/logout')
def logout():
    session.pop('user', None)
    return render_template('landing.html')

if __name__ == '__main__':
    app.run(debug=True)