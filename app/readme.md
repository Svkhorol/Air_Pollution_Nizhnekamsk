### Архитектура приложения


### Инструкция по запуску приложения на локальном сервере

- Требуется Python версии не ниже 3.10  
  
- Клонировать репозиторий:
```bash
git clone https://github.com/Svkhorol/Air_Pollution_Nizhnekamsk.git
```  

- Перейти в папку с проектом, создать и активировать в ней виртуальное окружение:  
```bash
cd Air_Pollution_Nizhnekamsk
python -m venv venv
source venv/Scripts/activate
```

- В папке Air_Pollution_Nizhnekamsk перейти в папку app и установить зависимости из файла requirements.txt:  
```bash
cd app
pip install -r requirements.txt 
```
 
- В папке app запустить исполняемый файл:  
```bash
python app.py 
```
- В терминале перейти по ссылке http://127.0.0.1:5000 на локальный сервер
