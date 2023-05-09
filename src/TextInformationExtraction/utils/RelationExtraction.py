import os
import numpy as np
import pandas as pd
import re

import itertools
import unidecode
import stanza
from stanza.models.common.doc import Document
import copy
import networkx as nx
import torch
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

class RelationExtraction() : 

    def __init__(self,pos,dict_tag):
        self.pos = pos
        self.dict_tag = dict_tag

    ### Liste de dictionnaire de relations de dépendances ###

    def relation_dependance(self,doc) : 
        liste_relation = []
        for i in range(0,len(doc.sentences)):
            sent = doc.sentences[i]
            liste_relation_sent = []
            for word in sent.words: 
                rel = {}
                if(word.deprel not in ['punct','cc','det',"root",'conj']):
                    liste_relation_sent.append({"texte1" : word.text, "texte2": sent.words[word.head-1].text,"id1" : word.id,"id2":sent.words[word.head-1].id,"relation": word.deprel})
                elif(word.deprel == "conj") : 
                    dep = "conj"
                    word_dep = word
                    
                    while(dep == "conj") : 
                        word_dep = sent.words[word_dep.head-1]
                        dep = word_dep.deprel
                    rel = {"texte1" : word.text, "texte2": sent.words[word_dep.head-1].text,"id1" : word.id,"id2":sent.words[word_dep.head-1].id,"relation": word_dep.deprel}
            
                    liste_relation_sent.append(rel)
            liste_relation.append(liste_relation_sent)
        return liste_relation
    
    
    

    ### CREATION GRAPHE DE RELATIONS ###

    def graph_dep(self,G,liste_relation):
        for i in range(0,len(liste_relation)) : 
            for rel in liste_relation[i]:
                if(str(rel['id2']-1) in list(self.dict_tag[i].keys())):
                    G.add_nodes_from([(rel["texte2"]+"_"+str(rel['id2'])+"_"+str(i),{"label": self.dict_tag[i][str(rel['id2']-1)],"id":rel['id2']})])
            
                else:
                    G.add_nodes_from([(rel["texte2"]+"_"+str(rel['id2'])+"_"+str(i),{"label": self.pos[i][rel['id2']-1]['upos'],"id":rel['id2']})])
                
                if(str(rel['id1']-1) in list(self.dict_tag[i].keys())):
                    G.add_nodes_from([(rel["texte1"]+"_"+str(rel['id1'])+"_"+str(i),{"label": self.dict_tag[i][str(rel['id1']-1)],"id":rel['id1']})])
            
                else:
                    G.add_nodes_from([(rel["texte1"]+"_"+str(rel['id1'])+"_"+str(i),{"label": self.pos[i][rel['id1']-1]['upos'],"id":rel['id1']})])
            
            
                G.add_edge(rel["texte2"]+"_"+str(rel['id2'])+"_"+str(i), rel["texte1"]+"_"+str(rel['id1'])+"_"+str(i),rel=rel['relation'])

        
        return G
    

    ### EXTRACTION PLUS COURT CHEMIN ### 
    def source_target_path(self,nodes1,nodes2,G,pretagged_doc,max_length,max_word):

        dict_pos_tot_source = []
        dict_word_tot_source = []
        dict_deprel_tot_source = []
        dict_pos_rel_tot = []
        dict_pos_rel_id_tot = []
        for n in nodes1 :
            for m in nodes2:
                if(nx.has_path(G, n, m)):
                    dict_pos = []
                    dict_pos.append(G.nodes[n]["label"])
                    dict_word_id= []
                    dict_word_id.append(int(n.split("_")[1]))
                    word_id_phrase = int(n.split("_")[2])
                    dict_word_rel = []
                    dict_word_rel.append(n.split("_")[0])
                    dict_deprel = []
                
                    short_path = nx.shortest_path(G, source=n, target=m)
                    if((len(short_path)<max_length) & (len(short_path)>1)):
                        for i in range(0,len(short_path)-1):
                            dict_word_id.append(int(short_path[i+1].split("_")[1]))
                            dict_word_rel.append(short_path[i+1].split("_")[0])
                            dict_pos.append(G.nodes[short_path[i+1]]['label'])
                            dict_deprel.append(G.get_edge_data(short_path[i],short_path[i+1])['rel'])
                        dict_word = [word.text for word in pretagged_doc.sentences[word_id_phrase].words  if word.id in range(min(dict_word_id)-3,max(dict_word_id)+3)]
                        length_word = len([word.text for word in pretagged_doc.sentences[word_id_phrase].words  if word.id in range(min(dict_word_id)+1,max(dict_word_id)-1)])
                        
                        if(length_word <=max_word):
                            dict_pos_tot_source.append(dict_pos)
                            dict_deprel_tot_source.append(dict_deprel)
                            dict_word_tot_source.append(dict_word)
                            dict_pos_rel_tot.append(((tuple(dict_pos),tuple(dict_deprel)),tuple(dict_word),tuple(dict_word_rel),tuple(dict_word_id),word_id_phrase,length_word,len(short_path)))
                            dict_pos_rel_id_tot.append((tuple(dict_word_id),word_id_phrase))

        return dict_pos_rel_tot, dict_pos_rel_id_tot, dict_pos_tot_source, dict_deprel_tot_source

    ### GROUPER MOTS ###
    
    def group_word(self, data,upos,ner,del_ner):
        liste_pos_final_group = copy.deepcopy(self.pos)
        liste_dict_tag_group = copy.deepcopy(self.dict_tag)
        for sent in range(0,len(self.pos)):
            j=0
            id_eg = data["ids"][0][0]
            for index,row in data.iterrows() : 
                if((row['id_phrase']==sent) & (row["ids"][0]<row["ids"][-1])):

                    if(id_eg==row["ids"][0]):
                        liste_pos_final_group[row["id_phrase"]][row["ids"][0]-1]["upos"] = upos
                        liste_pos_final_group[row["id_phrase"]][row["ids"][0]-1]["lemma"] =  " ".join([liste_pos_final_group[row["id_phrase"]][k-1]["lemma"] for k in range(row["ids"][0], row["ids"][-1]-j+1)])
                        liste_pos_final_group[row["id_phrase"]][row["ids"][0]-1]["text"] =  " ".join([liste_pos_final_group[row["id_phrase"]][k-1]["text"] for k in range(row["ids"][0], row["ids"][-1]-j+1)])

                        liste_dict_tag_group[row["id_phrase"]][str(row["ids"][0]-1-j)] = ner
                        if(del_ner) : 
                            del liste_dict_tag_group[row["id_phrase"]][str(row["ids"][-1]-1-j)]

                        key_eg = row["ids"][0]-1-j

                        k=0
                        for i in range(row["ids"][0],row["ids"][-1]-j):
                            del liste_pos_final_group[row["id_phrase"]][i-k]
                            j=j+1
                            k=k+1

                        key= [i for i in liste_dict_tag_group[row["id_phrase"]]]
                        for i in key:
                            if(int(i)>key_eg):
                                liste_dict_tag_group[row["id_phrase"]][str(int(i)-k)] = liste_dict_tag_group[row["id_phrase"]].pop(i)

                    else:
                        liste_pos_final_group[row["id_phrase"]][row["ids"][0]-1-j]["upos"] = upos
                        liste_pos_final_group[row["id_phrase"]][row["ids"][0]-1-j]["lemma"] =  " ".join([liste_pos_final_group[row["id_phrase"]][k-1]["lemma"] for k in range(row["ids"][0]-j, row["ids"][-1]-j+1)])
                        liste_pos_final_group[row["id_phrase"]][row["ids"][0]-1-j]["text"] =  " ".join([liste_pos_final_group[row["id_phrase"]][k-1]["text"] for k in range(row["ids"][0]-j, row["ids"][-1]-j+1)])

                        liste_dict_tag_group[row["id_phrase"]][str(row["ids"][0]-1-j)] = ner
                        
                        if(del_ner) :
                            del liste_dict_tag_group[row["id_phrase"]][str(row["ids"][-1]-1-j)]


                        key_eg = row["ids"][0]-1-j
                        k=0
                        for i in range(row["ids"][0]-j,row["ids"][-1]-j):
                            del liste_pos_final_group[row["id_phrase"]][i-k]
                            j=j+1
                            k=k+1

                        key = [i for i in liste_dict_tag_group[row["id_phrase"]]]
                        for i in key:
                            if(int(i)>key_eg):
                                liste_dict_tag_group[row["id_phrase"]][str(int(i)-k)] = liste_dict_tag_group[row["id_phrase"]].pop(i)


                    id_eg = row["ids"][0]
            for id_word in range(0,len(liste_pos_final_group[sent])):
                liste_pos_final_group[sent][id_word]['id'] = id_word+1

        return liste_pos_final_group, liste_dict_tag_group