import pydantic
from flask import Flask, request
from pydantic import BaseModel

from flask_blueprint_example import simple_page

app = Flask(__name__)  # Web Server Gateway Interface (WSGI) instance / application
app.register_blueprint(simple_page)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


# Start the server with `flask --app flask_postman_example run`.
# The server should run on `http://127.0.0.1:5000`.
# Open postman.
# Send a request to `http://127.0.0.1:5000` and check the result.

def my_function(number: int) -> str:
    return f'The number is {number}'


@app.route("/my_endpoint/<int:number>")
def my_endpoint(number):
    """Called for example with `http://127.0.0.1:5000/my_endpoint/12`"""
    print(type(number))
    return my_function(number)


class QueryModelForMyFunction(BaseModel):
    name: str


@pydantic.validate_arguments
def my_function_2(data: QueryModelForMyFunction):
    result = f'The name is {data.name}'
    return result


@app.route("/my_endpoint_2", methods=['POST'])
def my_endpoint_2():
    request_body = request.json
    try:
        result = my_function_2(request_body)
    except pydantic.ValidationError as e:
        return {
            "message": "Data validation error",
            "errors": e.errors()
        }, 400

    return {"message": "Here you have it", "payload": result}
