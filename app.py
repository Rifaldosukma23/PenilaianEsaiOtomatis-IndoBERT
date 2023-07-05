import os
import sys
from flask import Flask, render_template, request
from lib.predict import calculate_score

sys.path.insert(1, os.path.abspath(os.path.join(os.getcwd())))

app = Flask(__name__)
app.config["SECRET_KEY"] = 'rifaldo sukma s.'


@app.route("/", methods=["POST", "GET"])
@app.route("/dashboard", methods=["POST", "GET"])
def index():
    score = ""
    if request.method == "POST":
        question = request.form["question"]
        kunci_jawaban = request.form["answer_key"]
        jawaban_siswa = request.form.getlist("answer_students")

        score = calculate_score(question, kunci_jawaban, jawaban_siswa)

    return render_template("/index.html", score=score)


if __name__ == '__main__':
    app.run(host=os.getenv("API_HOST"), port=os.getenv("API_PORT"), debug=True)