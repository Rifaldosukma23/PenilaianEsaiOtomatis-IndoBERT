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
    if request.method == 'POST':
        question = request.form['question']
        kunci_jawaban = request.form['kunci_jawaban']
        jawaban_siswa_list = request.form.getlist('jawaban_siswa')

        results = []
        for jawaban_siswa in jawaban_siswa_list:
            result = calculate_score(question, kunci_jawaban, jawaban_siswa)
            results.append(result)

        return render_template('result.html', scores=results, jawaban_siswa_list=jawaban_siswa_list, question=question, kunci_jawaban=kunci_jawaban, enumerate=enumerate)

    return render_template('index.html')

@app.route("/result", methods=["POST", "GET"])
def result():
    if request.method == 'POST':
        question = request.form['question']
        kunci_jawaban = request.form['kunci_jawaban']
        jawaban_siswa_list = request.form.getlist('jawaban_siswa')

        results = []
        for jawaban_siswa in jawaban_siswa_list:
            result = calculate_score(question, kunci_jawaban, jawaban_siswa)
            results.append(result)

        return render_template('result.html', scores=results, jawaban_siswa_list=jawaban_siswa_list, question=question, kunci_jawaban=kunci_jawaban, enumerate=enumerate)

    return render_template('result.html')

if __name__ == '__main__':
     app.run()