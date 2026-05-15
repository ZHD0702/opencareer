from flask import Flask, render_template
import os

app = Flask(__name__)

@app.route('/')
def index():
    return '<h1>Flask is working!</h1><p>If you see this, Flask is configured correctly.</p>'

if __name__ == '__main__':
    print("Testing Flask...")
    print("Templates folder:", os.path.exists('templates'))
    print("Static folder:", os.path.exists('static'))
    app.run(host='0.0.0.0', port=5002, debug=False)
