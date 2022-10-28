import pickle
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

def stored_data(path):
    with open(path, "rb") as fIn:
        stored_data = pickle.load(fIn)
        stored_names = stored_data['names']
        stored_embeddigs = stored_data['embeddings']
    return stored_names, stored_embeddigs

def similar_companies(company):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'	
    model = SentenceTransformer('./weights/content/sbert_model', device=device)
    
    company_embedding = model.encode(company)
    
    stored_names, stored_embeddigs = stored_data('./weights/embeddings.pkl')
    
    cosine_companies = cosine_similarity(
        [company_embedding], 
        stored_embeddigs)[0]
    
    top_5_similar_companies = stored_names[np.argsort(cosine_companies)[-5:][::-1]]
    return top_5_similar_companies
    
if __name__ == "__main__":
    company_name = input('Enter company ')
    print(similar_companies(company_name))    
