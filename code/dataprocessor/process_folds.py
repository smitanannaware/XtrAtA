from dataloader import InitiativeExcelLoader
    
    
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd

embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')




dataloader = InitiativeExcelLoader()

def DFS(m, key, visit, fold, visit_m):
    visit_m.add(key)
    for val in set_m[key]:
        if visit[val] == 0:
            visit[val] = fold
            for mi in m[val].split(', '):
                if mi not in visit_m:
                    DFS(m, mi, visit, fold, visit_m)
dataset = pd.DataFrame(dataloader.dataset['test'])  
#dataset = dataloader.dataset['test']
dataset = dataset.loc[dataset['true_nc'] != 'None']
r = dataset['review'].tolist()
m = dataset['true_nc'].tolist()
#print('reviews:', r)
print('true_nc:', m)

set_m = {}

for index in range(len(m)):
    for item in m[index].split(', '):
        item = item.strip().lower()
        if item in set_m:
            set_m[item].add(index)
        else:
            set_m[item] = {index}

print(set_m)
# Corpus with example sentences
#print(set_m.keys())

#corpus_embeddings = embedder.encode(list(set_m.keys()))
corpus_embeddings = embedder.encode(m)

# Normalize the embeddings to unit length
corpus_embeddings = corpus_embeddings /  np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

# Perform kmean clustering
clustering_model = AgglomerativeClustering(n_clusters=11) #, affinity='cosine', linkage='average', distance_threshold=0.4)
clustering_model.fit(corpus_embeddings)
cluster_assignment = clustering_model.labels_

print(cluster_assignment)
key_cluster_map = {}
cluster_r_map = {}
for i, (item, cluster) in enumerate(zip(m, cluster_assignment)):
    if cluster in key_cluster_map:
        key_cluster_map[cluster].add(item)
        cluster_r_map[cluster].add(i)
    else:
        key_cluster_map[cluster] = {item}
        cluster_r_map[cluster] = {i}

print(key_cluster_map)
print(cluster_r_map)


# fold = 0
# visit = [0] * len(r)
# visit_m = set()
# for cluster in key_cluster_map.keys():
#     fold = fold+1
#     for key in key_cluster_map[cluster]:
#         if key not in visit_m:
#             DFS(m, key, visit, fold, visit_m)

#print(visit)
# final_map = {}
# for i, vis in enumerate(visit):
#     if vis in final_map:
#         final_map[vis].append(m[i])
#     else:
#         final_map[vis] = [m[i]]
        
#print(final_map)

# a_map = {}
# for cluster in key_cluster_map.keys():
#     for key in key_cluster_map[cluster]:
#         if cluster in a_map:
#             a_map[cluster].update(set_m[key])
#         else:
#             s = set()
#             s.update(set_m[key])
#             a_map[cluster] = s

# print(a_map)