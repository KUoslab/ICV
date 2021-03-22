from model.ner_model import NERModel
from model.config import Config
from model.utils import align_data


def get_model_api():
    """Returns lambda function for api"""
    # 1. initialize model once and for all and reload weights
    config = Config()
    model  = NERModel(config)
    model.build()
    model.restore_session("results/crf/model.weights/")

    def model_api(input_data):
        # 2. process input with simple tokenization and no punctuation
        punc = [",", "?", ".", ":", ";", "!", "(", ")", "[", "]"]
        s = "".join(c for c in input_data if c not in punc)
        words_raw = s.strip().split(" ")
        # 3. call model predict function
        preds = model.predict(words_raw)
        # 4. process the output
        output_data = align_data({"input": words_raw, "output": preds})
        # 5. return the output for the api
        return output_data

    return model_api
    
model_api = get_model_api()
response = model_api(request)