import os
import numpy as np
import pandas as pd

import math
import unicodedata
from unidecode import unidecode


 
import stanza
from stanza.models.common.doc import Document

import copy
import networkx as nx


import utils.Preprocessing as preprocessing
import utils.DependencyParsing as dep
import utils.RelationExtraction as RelationExtraction


class AppRelation() : 

    def __init__(self,new_dep) :
        self.new_dep = new_dep
    
    
    def get_nodes(self,phrase,nlp):

        doc = self.new_dep.get_doc(phrase)
        lemma = self.new_dep.get_lemma(doc)
        word_list = self.new_dep.get_word_list(doc)
        text_sentences =  self.new_dep.sentence_text(doc)
        sent = " ".join(text_sentences[i] for i in range(0,len(text_sentences)))

        # Ignored unknown kwarg option direction ? 
        liste_pos_final, liste_pos_conf_final = self.new_dep.get_postag(sent,lemma,word_list)

        try :
            # Ignored unknown kwarg option direction ? 
            index_tag_final,text_ner_final = self.new_dep.get_nertag(sent)
            liste_pos_final_group, id_tag_tot, tag_tot = self.new_dep.group_nertag(liste_pos_final, index_tag_final,doc)

            #### APPLICATION DU DEPENDENCY PARSER ####
            pretagged_doc = Document(liste_pos_final_group)
            new_doc = nlp(pretagged_doc)
            dict_tag_id = [{str(id_tag_tot[i][j]):tag_tot[i][j] for j in range(0,len(id_tag_tot[i]))} for i in range(0,len(id_tag_tot)) ]

            ### CREATION D'UNE LISTE AVEC CHAQUE DEPENDANCE POUR CHAQUE MOT DE CHAQUE PHRASE ###
            G = nx.DiGraph()
            new_rel = RelationExtraction.RelationExtraction(liste_pos_final_group,dict_tag_id)
            liste_relation = new_rel.relation_dependance(new_doc)

            ### CREATION DU GRAPHE ###
           
            G = new_rel.graph_dep(G,liste_relation)

        except :
            print("Problème annonce - get_nodes")
    

        return pretagged_doc,G, liste_relation, new_rel,sent,index_tag_final



    def pred_relation(self,nodes1,nodes2,pretagged_doc,length_relation,modele,d2v_i,d2v_p,tab_rel_seq,path_seq_input,path_seq_output,pca,new_rel,G):
    
            dict_prediction = []
            relation_essai = new_rel.source_target_path(nodes1,nodes2,G,pretagged_doc,length_relation)[0]
            for rel in relation_essai : 
                relation_essai_decompose = self.new_sqn.decompose_relation(rel[0])
                data_vec = self.create_sqn2vec(relation_essai_decompose,d2v_i,d2v_p,tab_rel_seq,path_seq_input,path_seq_output)
                mots_embed = self.new_sqn.embed_sentence(rel[1])
                
                #df_relation = pd.concat([pd.DataFrame([data_vec]),pd.DataFrame([mots_embed],columns=["embed"+str(i) for i in range(0,3016)])],axis=1)
                df_relation = pd.concat([pd.DataFrame([data_vec]),pd.DataFrame(pca.transform(pd.DataFrame([mots_embed])),columns=["embed"+str(i) for i in range(0,64)])],axis=1)
                #df_relation =pd.DataFrame(pca.transform(pd.DataFrame([mots_embed])),columns=["embed"+str(i) for i in range(0,350)])
                #df_relation = pd.DataFrame([data_vec])
                dict_prediction.append({"mots_relation" : tuple(rel[2]),"ids" : tuple(rel[3]),"id_phrase":rel[4], "classe" : modele.predict(df_relation)[0], "prediction" : modele.predict_proba(df_relation)[0][modele.predict(df_relation)[0]]})
            df_prediction = pd.DataFrame(dict_prediction)
            return df_prediction

    def get_relation(self,nodes1,nodes2,pretagged_doc,max_length, max_word,new_rel,G):
    
            dict_prediction = []
            relation = new_rel.source_target_path(nodes1,nodes2,G,pretagged_doc,max_length,max_word)[0]
            for rel in relation : 
                dict_prediction.append({"mots_relation": tuple(rel[2]), "relation" : tuple(rel[0]), "ids" : tuple(rel[3]),"id_phrase":rel[4],"length_words_rel" : rel[5], "length_path" : rel[6] })
            df_prediction = pd.DataFrame(dict_prediction)
            return df_prediction