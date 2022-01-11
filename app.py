
from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import re
import string
import tensorflow as tf 

import pickle
from timeit import default_timer as timer  
import flask
app = Flask(__name__)

###############################################################################

tokenizer = pickle.load(open('tokenizer.pkl','rb'))   

def text_cleaning(text):
  '''' takes input as raw  text  and removes hyperlinks,Numbers,Angular Brackets,Square Brackets,'\n' character,**** by ABUSE,wordpuntuation '''
  text = str(text).lower()
  text = re.sub('https?://\S+|www\.\S+', '', text)  #Removing hyperlinks
  text=re.sub('\S*\d\S*',' ',text) #Removing Numbers
  text=re.sub('<.*?>+',' ',text)   #Removing Angular Brackets
  text=re.sub('\[.*?\]',' ',text)  #Removing Square Brackets
  text=re.sub('\n',' ',text)       #Removing '\n' character 
  text=re.sub('\*+','<ABUSE>',text) #Replacing **** by ABUSE word

  text = "".join([i for i in text if i not in string.punctuation]) # Removing puntuation 
  return text


################################################################################

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/index')
def index():
    return flask.render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
  ''' function takes twee-text and tweet-sentiment as input and outputs phrase in the text which supports sentiment '''
  start = timer()

  to_predict_list = request.form.to_dict()
  tweet_data = to_predict_list['tweet_text']
  stweet = list(tweet_data.split(','))
  tweet          = text_cleaning(stweet[0])
  encoder        = tokenizer.encode_plus(tweet,stweet[1],add_special_tokens=True,max_length=92,
                              return_attention_mask=True,pad_to_max_length=True,return_tensors='tf',verbose=False)
  input_ids      = encoder['input_ids'] 
  attention_mask = encoder['attention_mask']
  interpreter = tf.lite.Interpreter(model_path="converted_quant_model.tflite")
  interpreter.allocate_tensors()
  # Get input and output tensors.
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  # Test model on some input data.
  input_shape = input_details[0]['shape']

  interpreter.set_tensor(input_details[1]['index'], input_ids)
  interpreter.set_tensor(input_details[0]['index'], attention_mask)
  interpreter.invoke()
  start_logits = interpreter.get_tensor(output_details[0]['index'])
  end_logits  = interpreter.get_tensor(output_details[1]['index'])

  a = np.argmax(start_logits)
  b = np.argmax(end_logits)
  text1 = " "+" ".join(tweet.split())               # pred_answer
  enc = tokenizer.encode(text1)
  pred_text = tokenizer.decode(enc[a:b+1]).replace('[SEP]','')
  end = timer()
  print('total time : ',end - start)
  return jsonify({'sentiment_supporting_phrase': pred_text,'time':end-start})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)


