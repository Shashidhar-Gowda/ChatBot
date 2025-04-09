from tools import remove_punctuations , stemming

train = []
target = []

def create_vector(sent , words_list):
    vec = []
    clean_sent = remove_punctuations(sent)
    word_list = clean_sent.lower().split()
    stemming_list = stemming(word_list)
    for w in words_list:
        vec.append(1 if w in stemming_list else 0)
    return vec

def create_data(dt , words_list , intents_mapping):
    for d in dt["data"]:
        inte = d["intent"]
        queries = d["query"]
        for q in queries:
            vector = create_vector(q , words_list)
            train.append(vector)
            target.append(intents_mapping[inte])
    return train , target
