import pandas as pd
import numpy as np
import nltk
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
nltk.download('stopwords')
da_stop = set(nltk.corpus.stopwords.words('danish'))

sent_df = pd.read_csv("sentiment_analysis/2_headword_headword_polarity.csv", header=None)[[0, 4]]
sentiment_dict = {row[0]: row[4] for _, row in sent_df.iterrows()}

def get_tfs(text):
    return dict(nltk.probability.FreqDist(([lemmatizer.lemmatize(w) for word in text if (w:=word.lower()).isalpha()])))

# Use this function to find sentiments of node attributes
def get_sents(node_att):
    post_sents = dict()
    for post_id, text in tqdm(node_att.items()):
        tf = get_tfs(nltk.word_tokenize(text))
        if (denom := np.sum([tf[word] for word in tf.keys() if word in list(sentiment_dict.keys())])) != 0 :
            post_sents[post_id] = (
                np.sum([sentiment_dict[word] * tf[word] for word in tf.keys() if word in list(sentiment_dict.keys())]) / 
                denom
            )
        else:
            post_sents[post_id] = 0
    
    return post_sents

if __name__ == "__main__":
    import pickle; import networkx as nx
    with open('graph_analysis/Bipartite_G_5251_cutoff_v2.pkl', 'rb') as f:
        G = pickle.load(f)
        
    node_att = nx.get_node_attributes(G, 'bipartite')
    posts = [node for node in G.nodes if node_att[node]]
    n = nx.subgraph(G, posts)

    sents = get_sents(nx.get_node_attributes(n, 'text'))