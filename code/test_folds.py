from dataloader import InitiativeExcelLoader
    
    
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import os
import pickle

if __name__ == "__main__":
    '''
    path = 'nc_clusters.pickle'
    if os.path.isfile(path):
        with open(path, "rb") as f:
            print(pickle.load(f))

    '''
    embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    dataloader = InitiativeExcelLoader()

    dataset = pd.DataFrame(dataloader.dataset['test'])
    print(dataset.head()) 
    #dataset = dataloader.dataset['test']
    dataset_NC = dataset.loc[dataset['true_nc'] != 'None']
    r = dataset['review'].tolist()
    m = dataset['true_nc'].tolist()
    nc_oc = dataset['true_nc_oc'].tolist()
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

    #print(set_m)
    nc_list = list(set_m.keys())
    nc_list.remove('none')
    corpus_embeddings = embedder.encode(nc_list)

    # Normalize the embeddings to unit length
    corpus_embeddings = corpus_embeddings /  np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

    # Perform kmean clustering
    clustering_model = AgglomerativeClustering(n_clusters = None, distance_threshold=1.3) #, affinity='cosine', linkage='average', distance_threshold=0.4)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_

    print(cluster_assignment)
    max_key = max(cluster_assignment)
    cluster_assignment = np.append(cluster_assignment, max_key+1)

    #cluster_assignment = {0: {'cutest little piglet', 'baby alligator', 'rooster', 'peacocks', 'manatee', 'gators', 'clydesdales horses', 'anaconda', 'animals', 'pigs', 'goat', 'your pets', 'hounds', "their 'petting zoo'", 'local wildlife'}, 1: {'keno', 'computer football', 'madden 15', 'nfl blitz 99', 'computer nascar style racing', 'big connect 4', 'jenga', 'shuffleboard', 'life size beer pong', 'table and floor games', 'pool table', 'games', 'arcade games', 'board games', 'zombie dodgeball', 'ping pong', 'beach volleyball', 'badminton', 'quizo', 'trivia pursuit', 'virtual golf', 'mini trampoline', 'trivia night'}, 2: {'awesome skateboard deck'}, 3: {'casinos', 'lottery tickets/ scratchers', 'scratch off lottery machine'}, 4: {'moms', 'activists', 'hackers', 'anarchists'}, 5: {'drag show', 'balloon puppets', 'mark the balloon guy', 'stuffed animal/puppets', 'puppet show', 'hypnosis show'}, 6: {'local charity', 'outreach program', 'meal sponsorship program'}, 7: {'arizona greyhound rescue event', 'adoption', 'community scarf/blanket project', 'stray cat alliance'}, 8: {'hummingbirds', 'songbirds', 'parakeet paradise'}, 9: {'park woods', 'parklands', 'goose creek park'}, 10: {'journals', 'guidebook', 'used books', 'a guide on planetary leadership', 'basics of mechanical engineering'}, 11: {'book exchange area', 'paper studio'}, 12: {'their museum', 'collectibles', 'their extensive collection', 'red wine collection'}, 13: {'deck'}, 14: {'tank'}, 15: {'cellar room', 'lounge area', 'their apothecary area'}, 16: {'nice playroom', 'playground'}, 17: {'wifi', 'free wifi', 'internet'}, 18: {'ponchos'}, 19: {'henna artist', 'henna tattoo artist', 'local artist', 'local artists', 'fortune teller', 'stand-up comic'}, 20: {'professional babysitting service'}, 21: {'indy 500 photos', 'photo booth', 'polaroid'}, 22: {'custom associations', 'symbolism', 'culturally amazing german experience', 'history lesson'}, 23: {'canoeing', 'kayaking', 'paddle cruise', 'airboat ride'}, 24: {'beautiful garden', 'garden', 'organic gardeners', 'plants', 'adjacent garden'}, 25: {'10 dollar fresh day tour', 'their tour', 'amazing tour de bier', 'tour', 'free tour'}, 26: {'video', 'audiobook'}, 27: {'gift shop', 'bookstore/gift shop', 'cute indoor store'}, 28: {'greeting cards', 'gift tags', 'anniversary gifts', 'lip balms', 'shawl', 'their candles', 'candles', 'necklace', 'cute wallet', 'customized carmel sportswear'}, 29: {'gorgeous waves', 'shore'}, 30: {'open air beach view', 'sunset'}, 31: {'ava amphitheater', 'music venue', 'their nightclub', 'karaoke venue'}, 32: {'concert', 'entertainment', 'performances', 'small concerts', 'karaoke night', 'live music', 'best sounding band', 'live bands'}, 33: {'derek st. holmes'}, 34: {'nasty blues'}, 35: {'activities', 'hikes', 'river walk'}, 37: {'fireworks', 'fire', 'penns landing fireworks'}, 38: {'dolphin spotting', 'sunny fish', 'fishies'}, 39: {'water', 'dock area', 'dock', 'long public dock'}, 40: {'boaters', 'jet powered surfboard surfers', 'kite surfers'}, 41: {'cemetery', 'estate'}, 42: {'events', 'really good events', 'special event'}, 43: {'watercolors', 'great therapeutic sketching', 'art'}, 44: {'orange cat', 'cats'}, 45: {'circus', 'carnival'}, 46: {'bollywood movies'}, 47: {'different size diapers'}, 48: {'on site medium'}, 49: {'huge skater', 'his own skating company'}, 50: {'terrariums', 'beautiful pots'}, 51: {'sunbathers'}, 52: {'souvenir t-shirts', 'hats,"hi af" shirt'}, 53: {'free comic'}, 54: {'golf course'}, 55: {'philly community acupuncture'}, 56: {'movie good burger', 'good burger car'}, 57: {'free calendar'}, 58: {'pokestop', 'pokestops'}, 59: {'yoga', 'capoiera/zumba studio', 'dance floors'}, 60: {'glass'}, 61: {'coffee grounds'}, 62: {'tips'}, 63: {'entrance fee'}, 64: {'none'}}
    #print(cluster_assignment)
    nc_cluster_map = {}
    cluster_nc_map = {}
    nc_list.append('none')
    #print(nc_list)
    for item, cluster in zip(nc_list, cluster_assignment):
        if cluster in cluster_nc_map:
            cluster_nc_map[cluster].add(item)
        else:
            cluster_nc_map[cluster] = {item}
        nc_cluster_map[item] = cluster
        
    print(cluster_nc_map)
    print(nc_cluster_map)

    
    cluster_review_mapping = {}
    review_cluster_mapping = {}
    for cluster in cluster_nc_map.keys():
        for key in cluster_nc_map[cluster]:
            if cluster in cluster_review_mapping:
                cluster_review_mapping[cluster].update(set_m[key])
            else:
                s = set()
                s.update(set_m[key])
                cluster_review_mapping[cluster] = s
            for r in set_m[key]:
                if r in review_cluster_mapping:
                    review_cluster_mapping[r].add(cluster.item())
                else:
                    s = set()
                    s.add(cluster.item())
                    review_cluster_mapping[r] = s
            

    print(cluster_review_mapping)
    print(review_cluster_mapping)

    dataset['nc_clusters'] = dataset.index.to_series().map(review_cluster_mapping)
    dataset.to_excel('clustered_reviews.xlsx')

    clusters = {0: {'cutest little piglet', 'baby alligator', 'rooster', 'peacocks', 'manatee', 'gators', 'clydesdales horses', 'anaconda', 'animals', 'pigs', 'goat', 'your pets', 'hounds', "their 'petting zoo'", 'local wildlife'}, 1: {'keno', 'computer football', 'madden 15', 'nfl blitz 99', 'computer nascar style racing', 'big connect 4', 'jenga', 'shuffleboard', 'life size beer pong', 'table and floor games', 'pool table', 'games', 'arcade games', 'board games', 'zombie dodgeball', 'ping pong', 'beach volleyball', 'badminton', 'quizo', 'trivia pursuit', 'virtual golf', 'mini trampoline', 'trivia night'}, 2: {'awesome skateboard deck'}, 3: {'casinos', 'lottery tickets/ scratchers', 'scratch off lottery machine'}, 4: {'moms', 'activists', 'hackers', 'anarchists'}, 5: {'drag show', 'balloon puppets', 'mark the balloon guy', 'stuffed animal/puppets', 'puppet show', 'hypnosis show'}, 6: {'local charity', 'outreach program', 'meal sponsorship program'}, 7: {'arizona greyhound rescue event', 'adoption', 'community scarf/blanket project', 'stray cat alliance'}, 8: {'hummingbirds', 'songbirds', 'parakeet paradise'}, 9: {'park woods', 'parklands', 'goose creek park'}, 10: {'journals', 'guidebook', 'used books', 'a guide on planetary leadership', 'basics of mechanical engineering'}, 11: {'book exchange area', 'paper studio'}, 12: {'their museum', 'collectibles', 'their extensive collection', 'red wine collection'}, 13: {'deck'}, 14: {'tank'}, 15: {'cellar room', 'lounge area', 'their apothecary area'}, 16: {'nice playroom', 'playground'}, 17: {'wifi', 'free wifi', 'internet'}, 18: {'ponchos'}, 19: {'henna artist', 'henna tattoo artist', 'local artist', 'local artists', 'fortune teller', 'stand-up comic'}, 20: {'professional babysitting service'}, 21: {'indy 500 photos', 'photo booth', 'polaroid'}, 22: {'custom associations', 'symbolism', 'culturally amazing german experience', 'history lesson'}, 23: {'canoeing', 'kayaking', 'paddle cruise', 'airboat ride'}, 24: {'beautiful garden', 'garden', 'organic gardeners', 'plants', 'adjacent garden'}, 25: {'10 dollar fresh day tour', 'their tour', 'amazing tour de bier', 'tour', 'free tour'}, 26: {'video', 'audiobook'}, 27: {'gift shop', 'bookstore/gift shop', 'cute indoor store'}, 28: {'greeting cards', 'gift tags', 'anniversary gifts', 'lip balms', 'shawl', 'their candles', 'candles', 'necklace', 'cute wallet', 'customized carmel sportswear'}, 29: {'gorgeous waves', 'shore'}, 30: {'open air beach view', 'sunset'}, 31: {'ava amphitheater', 'music venue', 'their nightclub', 'karaoke venue'}, 32: {'concert', 'entertainment', 'performances', 'small concerts', 'karaoke night', 'live music', 'best sounding band', 'live bands'}, 33: {'derek st. holmes'}, 34: {'nasty blues'}, 35: {'activities', 'hikes', 'river walk'}, 37: {'fireworks', 'fire', 'penns landing fireworks'}, 38: {'dolphin spotting', 'sunny fish', 'fishies'}, 39: {'water', 'dock area', 'dock', 'long public dock'}, 40: {'boaters', 'jet powered surfboard surfers', 'kite surfers'}, 41: {'cemetery', 'estate'}, 42: {'events', 'really good events', 'special event'}, 43: {'watercolors', 'great therapeutic sketching', 'art'}, 44: {'orange cat', 'cats'}, 45: {'circus', 'carnival'}, 46: {'bollywood movies'}, 47: {'different size diapers'}, 48: {'on site medium'}, 49: {'huge skater', 'his own skating company'}, 50: {'terrariums', 'beautiful pots'}, 51: {'sunbathers'}, 52: {'souvenir t-shirts', 'hats,"hi af" shirt'}, 53: {'free comic'}, 54: {'golf course'}, 55: {'philly community acupuncture'}, 56: {'movie good burger', 'good burger car'}, 57: {'free calendar'}, 58: {'pokestop', 'pokestops'}, 59: {'yoga', 'capoiera/zumba studio', 'dance floors'}, 60: {'glass'}, 61: {'coffee grounds'}, 62: {'tips'}, 63: {'entrance fee'}, 64: {'none'}}
    