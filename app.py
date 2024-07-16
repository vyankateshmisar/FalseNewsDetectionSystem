from textblob import TextBlob
from flask_cors import CORS
import validators
import urllib
import newspaper
from newspaper import Article, Config
import pickle
import numpy as np
import pandas as pd
# TextBlob(sentence).sentiment

from flask import Flask, jsonify, request
app = Flask(__name__)
CORS(app)

#Receiving the input url from the user and using Web Scrapping to extract the news content
@app.route('/predict',methods=['GET','POST'])
def predict():
    data = request.get_json()
    url = data['sentence']
    url = url[5:]
    url = urllib.parse.unquote(url)
    validate = validators.url(url)

    if validate == True:
        user_agent = request.headers.get('User-Agent')
        config = Config()
        config.browser_user_agent = user_agent

        try:
            article = Article(str(url))
            article.download()
            article.parse()
            parsed = article.text

            if parsed:  
                b = TextBlob(parsed)
                lang = b.detect_language()

                if lang == "en":
                    article.nlp()
                    news_title = article.title
                    news = article.text
                    news_html = article.html

                    if news:
                        news_to_predict = pd.Series(np.array([news]))

                        cleaner = pickle.load(open('TfidfVectorizer-new.sav', 'rb'))
                        model = pickle.load(open('DL.sav', 'rb'))

                        cleaned_text = cleaner.transform(news_to_predict)
                        pred = model.predict(cleaned_text)
                        pred_outcome = format(pred[0])
                        if (pred_outcome == "0"):
                            outcome = "True"
                        else:
                            if (pred_outcome == "REAL"):
                                outcome = "True"
                            else:
                                outcome = "False"
                        
                        return jsonify({"sentiment": outcome})
                    else:
                        error = 'Invalid URL! Please try again'
                        return jsonify({"sentiment": error})
                else:
                    language_error = "We currently do not support this language"
                    return jsonify({"sentiment": language_error})
            else:
                error = 'Invalid news article! Please try again'
                return jsonify({"sentiment": error})
        except newspaper.article.ArticleException:
            error = 'We currently do not support this Website! Please try again'
            return jsonify({"sentiment": error})
        
    else:
        error = 'Please Enter a Valid News Site URL', 'danger'
        return jsonify({"sentiment": error})


@app.route('/', methods=['GET'])
def hello():
    return jsonify({"response":"This is Sentiment Application"})

if __name__=="__main__":
    app.run(host="0.0.0.0", threaded=True, port=5000)