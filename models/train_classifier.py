import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import pickle
import gzip


import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])


from nltk.corpus import wordnet as wn
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline, FeatureUnion
#, FeatureUnion, make_pipeline, make_union
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
#, make_scorer, accuracy_score, jaccard_score, hamming_loss
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin



tag_map = {
        'CC':None, # coordin. conjunction (and, but, or)  
        'CD':wn.NOUN, # cardinal number (one, two)             
        'DT':None, # determiner (a, the)                    
        'EX':wn.ADV, # existential ‘there’ (there)           
        'FW':None, # foreign word (mea culpa)             
        'IN':wn.ADV, # preposition/sub-conj (of, in, by)   
        'JJ':wn.ADJ, # adjective (yellow)                  
        'JJR':wn.ADJ, # adj., comparative (bigger)          
        'JJS':wn.ADJ, # adj., superlative (wildest)           
        'LS':None, # list item marker (1, 2, One)          
        'MD':None, # modal (can, should)                    
        'NN':wn.NOUN, # noun, sing. or mass (llama)          
        'NNS':wn.NOUN, # noun, plural (llamas)                  
        'NNP':wn.NOUN, # proper noun, sing. (IBM)              
        'NNPS':wn.NOUN, # proper noun, plural (Carolinas)
        'PDT':wn.ADJ, # predeterminer (all, both)            
        'POS':None, # possessive ending (’s )               
        'PRP':None, # personal pronoun (I, you, he)     
        'PRP$':None, # possessive pronoun (your, one’s)    
        'RB':wn.ADV, # adverb (quickly, never)            
        'RBR':wn.ADV, # adverb, comparative (faster)        
        'RBS':wn.ADV, # adverb, superlative (fastest)     
        'RP':wn.ADJ, # particle (up, off)
        'SYM':None, # symbol (+,%, &)
        'TO':None, # “to” (to)
        'UH':None, # interjection (ah, oops)
        'VB':wn.VERB, # verb base form (eat)
        'VBD':wn.VERB, # verb past tense (ate)
        'VBG':wn.VERB, # verb gerund (eating)
        'VBN':wn.VERB, # verb past participle (eaten)
        'VBP':wn.VERB, # verb non-3sg pres (eat)
        'VBZ':wn.VERB, # verb 3sg pres (eats)
        'WDT':None, # wh-determiner (which, that)
        'WP':None, # wh-pronoun (what, who)
        'WP$':None, # possessive (wh- whose)
        'WRB':None, # wh-adverb (how, where)
        '$':None, #  dollar sign ($)
        '#':None, # pound sign (#)
        '“':None, # left quote (‘ or “)
        '”':None, # right quote (’ or ”)
        '(':None, # left parenthesis ([, (, {, <)
        ')':None, # right parenthesis (], ), }, >)
        ',':None, # comma (,)
        '.':None, # sentence-final punc (. ! ?)
        ':':None # mid-sentence punc (: ; ... – -)
    }

def load_data(database_filepath):

    '''loads data from database

        Inputs:
                db filepath
        Output:
                X: messages
                y: 36 categories dataframe
                categories_names: y column names


    '''
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql("SELECT * FROM DisasterResponse", engine)
    df.drop('child_alone', axis =1, inplace =True)
    df['related']= np.where(df['related']==2, 1, 0)
    #X = df['message']
    X = df['message']
    y = df.iloc[:,4:]
    category_names = df.columns[4:]
    return X, y, category_names
    
    


def tokenize(text):
    
    """
    Tokenize the text function
    
    Arguments:
        text -> Text message which needs to be tokenized
    Output:
        clean_tokens -> List of tokens extracted from the provided text
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    #remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())    
    tokens = word_tokenize(text)

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    # remove short words
    tokens = [token for token in tokens if len(token) > 2]
    
    # tag parts of speech
    pos_tokens = pos_tag(tokens) 

    clean_tokens = []
    for tok, pos in pos_tokens:
        try:
            if tag_map[pos] is not None:
                clean_tok = lemmatizer.lemmatize(tok, tag_map[pos]).lower().strip()
                clean_tokens.append(clean_tok)
            else:
                clean_tok = lemmatizer.lemmatize(tok).strip()
                clean_tokens.append(clean_tok)
        except KeyError:
            pass
    return clean_tokens


def build_model():
    ''' Builds model
        Inputs: 
            None
        Output: Model
            
    '''
    pipeline = Pipeline([
    ('features', FeatureUnion([
            
        ('text_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())
        ]))
    ])),
    
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
    'clf__estimator__min_samples_leaf': [1,2]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=2)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    ''' Predicts on houldout data and prints out classification report for each category  
        Inputs: 
            model: model as specified in build_model()
            X_test: holdout data with messages
            Y_test: holdout data with categories
            category_names: list of categories' names
        Output: None
            
    '''
    y_pred = model.predict(X_test)
    print(classification_report(Y_test.values, y_pred, target_names=Y_test.columns.values))

    


def save_model(model, model_filepath):
    ''' Saves model as gzip object
        Inputs: 
            model: model 
            model_filepath: path to save the model
        Output: None
            
    '''
    with gzip.open(model_filepath, 'wb') as gzipped_f:
    # Pickle the trained pipeline using the highest protocol available.
        pickled = pickle.dumps(model)
        gzipped_f.write(pickled)



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()