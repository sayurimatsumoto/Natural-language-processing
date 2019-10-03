import numpy as np
import nltk
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer 
from nltk.util import ngrams # function for making ngrams



NEGATION = ["not", "no", "nothing", "never"]

def remove_not_alpha(words):
    
    only_alpha = [word.replace("'", "o") for word in words if word.isalpha() or (word == "n't")]
    
    return only_alpha
    
def remove_stop_words(words):
    
    stop_words = stopwords.words('english')
        
    without_sw = [word for word in words if word not in stop_words or word in NEGATION]
    
    return without_sw

def stemming(words):
    
    stemmer = PorterStemmer() 
   
    stemming_words = [ stemmer.stem(word) for word in words]
    
    return stemming_words


def pos_tag(words, noun):
        
    pt_words = []
       
    i = 0
    
    adj = False
    
    for token, pos in nltk.pos_tag(words):
        
        if (token =='very'):
        	print(pos)

        #if flag is true, get nouns
        if noun and pos[0] == 'N':
            pt_words.append(token)
        
        #get only verbs, adverbs and adjectives
        if pos[0] == 'V' or pos[0] == 'J' or pos[0] == 'R': 
            pt_words.append(token)
        
    return pt_words
    
#modifiquei apenas para testar argumentos
def clear_text(words, noun):
        
    words = remove_not_alpha(words)

    words = remove_stop_words(words)

    words = stemming(pos_tag(words, noun))
        
    return words
         

def handle_negation(words):
           
    with_negation = []
    
    for i in range(len(words)):
        
        if words[i] in NEGATION and i+1 < len(words):
            with_negation.append((words[i], words[i+1]))
            i += 1
        else:
            with_negation.append(words[i])
            
    return with_negation

def bow(vocabulary, review_text):

	noun = True
	       
	review_text = nltk.word_tokenize(review_text)
	review_text = clear_text(review_text, noun)
	review_text = handle_negation(review_text)

	sample = np.zeros(len(vocabulary), dtype="int")

	for word in review_text:
		if word in vocabulary:
			i = vocabulary.index(word)
			sample[i] += 1


	return sample
