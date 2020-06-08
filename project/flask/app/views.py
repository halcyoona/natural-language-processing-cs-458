from app import app
from pusher import Pusher
from flask import render_template, request


pusher = Pusher(
    app_id='1014384',
    key='b33b6b39041e051213d3',
    secret='bf9ded3a771709c92b97',
    cluster='ap2', 
    ssl=True
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/orders', methods=['POST'])
def order():
    data = request.form
    pusher.trigger(u'order', u'place', {
        u'units': data['units']
    })
    return "units logged"

@app.route('/message', methods=['POST'])
def message():
    data = request.form
    pusher.trigger(u'message', u'send', {
        u'name': data['name'],
        u'message': data['message']
    })
    return "message sent"

@app.route('/customer', methods=['POST'])
def customer():
    data = request.form
    pusher.trigger(u'customer', u'add', {
        u'name': data['name'],
        u'position': data['position'],
        u'office': data['office'],
        u'age': data['age'],
        u'salary': data['salary'],
    })
    return "customer added"
