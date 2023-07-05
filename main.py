from flask import Flask, render_template, request
from lib.predict import calculate_score

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def score():
    if request.method == 'POST':
        question = request.form['question']
        kunci_jawaban = request.form['kunci_jawaban']
        jawaban_siswa_list = request.form.getlist('jawaban_siswa')

        results = []
        for jawaban_siswa in jawaban_siswa_list:
            result = calculate_score(question, kunci_jawaban, jawaban_siswa)
            results.append(result)

        return render_template('result.html', scores=results, jawaban_siswa_list=jawaban_siswa_list, question=question, kunci_jawaban=kunci_jawaban, enumerate=enumerate)

    return render_template('test/index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000, debug=True)
