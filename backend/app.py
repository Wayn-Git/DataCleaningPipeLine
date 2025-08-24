from flask import Flask
from routes.clean_api import clean_bp

app = Flask(__name__)
app.register_blueprint(clean_bp)

if __name__ == "__main__":
    app.run(debug = True)