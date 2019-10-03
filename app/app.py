import numpy as np
from flask import Flask, render_template, jsonify
from flask import request
import json
import pp 
from time import sleep

app = Flask(__name__)

def sigmoid(z):

    """
    Calcula a função sigmoidal  
    """
    z = 1/(1+np.exp(-z))
    
    return z

def predicao(Theta1, Theta2, Xval):

    m = Xval.shape[0] # número de amostras
    num_labels = Theta2.shape[0]
    
    p = np.zeros(m)

    a1 = np.hstack( [np.ones([m,1]), Xval] )
    h1 = sigmoid( np.dot(a1, Theta1.T) )

    a2 = np.hstack( [np.ones([m,1]), h1] ) 
    h2 = sigmoid( np.dot(a2, Theta2.T) )

    print("h2 = ", h2)

    Ypred = np.argmax(h2,axis=1)
            
    return Ypred

@app.route('/', methods=['GET', 'POST'])
def review():
	return render_template('review.html')	


@app.route('/request', methods=['GET', 'POST'])
def request_review():

	if request.method == 'POST':

		category = request.form.get('category')
		print("categoria = " + category)
		review_text = request.form.get('review')

		title = request.form.get('reviewTitle')

		if title:
			review_text += title
			
		with open('data/' + category +'/vocabulary.txt', 'r') as in_file:
			vocabulary = in_file.read().split('\n')

		vocabulary = vocabulary[:-1] #remove last \n

		for i in range(len(vocabulary)):
			if vocabulary[i][0] == "(" and vocabulary[i][-1] == ")":
				elements = vocabulary[i][1:-1].split(",")
				vocabulary[i] = tuple(elements)

		Theta1 = np.load('data/' + category+'/Theta1.npy')

		Theta2 = np.load('data/' + category+'/Theta2.npy')

		sample = pp.bow(vocabulary, review_text)

		classe = predicao(Theta1, Theta2, np.asmatrix(sample))

		classe = classe[0][0]

		print("classe ", int(classe))

		sleep(0.5)

		return jsonify({'classe': int(classe)})


if __name__ == '__main__':
    app.run(debug=True)