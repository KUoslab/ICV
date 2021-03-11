from flask import Flask, render_template
import flask_restful
import tensorflow as tf
import numpy as np
# 데이터를 읽어들이고
((train_data, train_label), (eval_data, eval_label)) = tf.keras.datasets.mnist.load_data()
eval_data = eval_data/np.float32(255)
eval_data = eval_data.reshape(10000, 28, 28, 1)
# 저장해 두었던 모델을 읽어들인 후
model_dir = "/tmp/tfkeras_mnist"
new_model = tf.keras.experimental.load_from_saved_model(model_dir)
new_model.summary()
#그래프를 생성하고
new_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Flask Restful API로 읽어들일 APP을 지정.
app = Flask(__name__)
api = flask_restful.Api(app)
# Flask가 사용할 리소스는 Test 클래스. 
# get 함수가 HTTP Get으로 결과를 읽어들임
class Test(flask_restful.Resource):
    def get(self):
        random_idx = np.random.choice(eval_data.shape[0])
        random_idx
        test_data = eval_data[random_idx].reshape(1, 28, 28, 1)
        res = new_model.predict(test_data)
return {
            'predict': np.argmax(res).tolist(),
            'answer': eval_label[random_idx].tolist()
            }
# Test 클래스를 리소스로 추가. 두번째 인자는 파일의 위치. 
# 우리는 ~/venv/tf_mnist 현재 디렉토리에서 읽을 것이므로 '/'     
api.add_resource(Test, '/')
# 사용하는 포트는 5000번
if __name__ == "__main__":              
    app.run(host="0.0.0.0", port=5000)

# 서버 수행 후 (python3 flask-server-medium.py)
# 브라우저로 테스트 : 