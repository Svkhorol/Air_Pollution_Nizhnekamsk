import os
import pickle

from flask import Flask, render_template, request, flash
from flask_bootstrap import Bootstrap


app = Flask(__name__, template_folder='templates')
bootstrap = Bootstrap(app)

ALLOWED_EXTENSIONS = {'xlsx'}
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['POST', 'GET'])
def main():
    """Функция для рендеринга главной страницы приложения.
    """

    # Обработка GET-запроса
    if request.method == 'GET':
        return render_template('index.html')

    # Обработка POST-запроса
    if request.method == 'POST':

        files = request.files.getlist('file')
        for file in files:
            if file and allowed_file(file.filename):
                filename = file.filename.replace(' ', '_')
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)

        # Обработка файлов

        # Удаление файлов
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            if filename != '.gitkeep':
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                os.remove(file_path)

        return render_template('result.html')


if __name__ == '__main__':
    app.run()
