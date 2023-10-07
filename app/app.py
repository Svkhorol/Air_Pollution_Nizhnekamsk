import pickle

from flask import Flask, render_template, request
#from flask_bootstrap import Bootstrap


app = Flask(__name__, template_folder='templates')
#bootstrap = Bootstrap(app)


@app.route('/', methods=['POST', 'GET'])
def main():
    """Функция для рендеринга главной страницы приложения.
    """

    # Обработка GET-запроса
    if request.method == 'GET':
        return render_template('index.html')

    # Обработка POST-запроса
    if request.method == 'POST':
        pass


if __name__ == '__main__':
    app.run()
