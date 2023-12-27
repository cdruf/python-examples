from flask import Flask

app = Flask(__name__)  # Web Server Gateway Interface (WSGI) instance / application


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

# Start the server with `flask --app flask_postman_example run`.
# The server should run on `http://127.0.0.1:5000`.
# Open postman.
# Send a request to `http://127.0.0.1:5000` and check the result.
