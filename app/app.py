import pickle

from flask import Flask, render_template, request


app = Flask(__name__,
            template_folder='templates',
            )


@app.route('/', methods=['POST', 'GET'])
def main():
    """Функция для рендеринга главной страницы приложения.
    """

    # Обработка GET-запроса
    if request.method == 'GET':
        return render_template('main_map.html')

    # Обработка POST-запроса
    if request.method == 'POST':
        pass


if __name__ == '__main__':
    app.run()
