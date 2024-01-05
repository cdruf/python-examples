from flask import Blueprint

simple_page = Blueprint('simple_page', __name__)


@simple_page.route('/my_blueprint')
def my_blueprint_function():
    return "my_blueprint_function"
