import flask
from flask import Flask, request, render_template
from sklearn.externals import joblib
import numpy as np
from scipy import misc

app = Flask(__name__)

# 메인 페이지 라우팅
# 기본 url 요청 시 기본 index.html로 이동 (렌더링)
@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')

# 데이터 예측 처리
@app.route('/predict',methods=['POST'])
    # 핵심 메소드!
    # 클라이언트에서 숫자 이미지 파일 업로드 -> 편집 (이미지 알파값 제거 등) -> 문자열 변환
    # -> 숫자 예측 -> 예측 숫자값을 index.html로 리턴 -> 렌더링
def make_prediction():
    if request.method == 'POST':
        # 업로드 파일 처리 분기
        file = request.files['image']
        if not file: return render_template('index.html', label="No Files")

        # 이미지 픽셀 정보 읽기
        # 알파 채널 값 제거 후 1차원 reshape
        img = misc.imread(file)
        img = img[:, :, :3]
        img = img.reshape(1, -1)

        # 입력 받은 이미지 예측
        prediction = model.predict(img)

        # 예측 값을 1차원 배열로부터 확인 가능한 문자열로 변환
        label = str(np.squeeze(prediction))

        # 숫자가 10일 경우 0으로 처리
        if label == 10: label = '0'

        # 결과 리턴
        return render_template('index.html', label=label)

if __name__ == '__main__':
    # 모델 로드
    # ml/model.py 선 실행 후 생성
    model = joblib.load('./model/model.pkl')
        # 기존 학습 모델 로드
    # flask 서버 가동 == flask 서비스 시작
    app.run(host='0.0.0.0', port=8000, debug=True)