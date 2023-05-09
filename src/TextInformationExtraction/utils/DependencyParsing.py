import os
import numpy as np
import pandas as pd
import re

import flair
from flair.data import Sentence
from flair.models import SequenceTagger
from flair.data import Dictionary, Corpus
from flair.datasets import ColumnCorpus
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, TransformerWordEmbeddings,CharacterEmbeddings, FlairEmbeddings

import stanza
from stanza.models.common.doc import Document
import copy
import networkx as nx
import itertools

import torch
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import subprocess

import utils.Preprocessing as preprocessing

class DependancyParsing() : 

    def __init__(self, model_ner, model_pos, nlp, nlp_mwt):
        """
        model_ner = SequenceTagger.load('./model_flair_eg_rs/final-model.pt')
        model_pos = SequenceTagger.load('./model_pos_best/final-model.pt')
        nlp = stanza.Pipeline(lang='fr', processors='depparse', depparse_pretagged=True) 
        -> le paramètre depparse_pretagged permet de donner un texte déjà taggué et donc Stanza n'utilise pas son POS TAGGER (mais le notre de FLAIR)
        nlp_mwt = stanza.Pipeline(lang='fr', processors='tokenize,pos,mwt,lemma')
        -> LEMME
        """
        self.model_ner = model_ner
        self.model_pos = model_pos
        self.nlp = nlp 
        self.nlp_mwt = nlp_mwt



    def get_index_entity(self,span) :

        '''
        Fonction pour récupérer les index et les labels des entités
        ''' 
        index = []
        index_all = []
        entity_all = []
        for i in range(0,len(span)):
            index_token = []
            for j in range(0,len(span[i])) :
                index_token.append(int(str(span[i].tokens[j]).split(" ")[1])-1)
                index_all.append(int(str(span[i].tokens[j]).split(" ")[1])-1)
                entity_all.append(span[i].tag)
            index.append(index_token)
        return index,index_all,entity_all

    
    ### DECOUPAGE DU TEXTE ET LEMME AVEC LE MODELE DE STANZA POUR LE FRANCAIS ####

    def get_doc(self,text) :
        return self.nlp_mwt(preprocessing.Preprocessing.clean_text(text))


    def get_lemma(self,doc): 
        lemma = []
        for sent in doc.sentences:
            for word in sent.words:
                lemma.append(word.lemma)
        return lemma


    def get_word_list(self,doc): 
        word_list = []
        for sent in doc.sentences:
            for word in sent.words:
                word_list.append(word.text)
        return word_list

    #### DECOUPAGE DU TEXTE EN PHRASE ####

    def sentence_text(self,doc): 
        sentences_text = []
        for i in range(0,len(doc.sentences)) : 
            text= []
            for token in doc.sentences[i].tokens:
                text.append(" ".join([word.text for word in token.words]))
            sentences_text.append(" ".join(text))
            
        return sentences_text

    #### APPLICATION DU POS TAGGER DE FLAIR SUR CHAQUE PHRASE DU TEXTE ET CREATION DU SCHEMA DE DOCUMENT PRE TAGGUE -> liste_pos_final ####

    def get_postag(self,sent,lemma, word_list):
        
        liste_pos_final = []
        liste_pos_confidence_final = []
        id_lemme = 0
        
        sentence = Sentence(sent)
            
        self.model_pos.predict(sentence)
    
        for entity in range(0,len(sentence)) : 
            
            text = sentence[entity].text
            upos = sentence[entity].get_tag("upos").to_dict()["value"]
            confidence = sentence[entity].get_tag("upos").to_dict()["confidence"]
            
            if(word_list[id_lemme] != text):
                ## Résolution du problème si la tokenization différente entre lemme et Sentence(sent) ##
                lem1 = lemma[id_lemme][0:len(text)]
                lem2 = lemma[id_lemme][len(text)+1:]
                lem3 = lemma[id_lemme+1:]
                lemma[id_lemme] = lem1
                lemma[id_lemme+1] = lem2
                lemma[id_lemme+2:] = lem3
                    
                wrd1 = text
                wrd2 = sentence[entity+1].text
                wrd3 = word_list[id_lemme+1:]
                word_list[id_lemme] = wrd1
                word_list[id_lemme+1] = wrd2
                word_list[id_lemme+2:] = wrd3
        
            
            if((lemma[id_lemme-1].endswith(('er','ir','oir','re'))) &  (upos == "ADJ")):   
                liste_pos_final.append({'id':id_lemme,'text':text,'lemma': re.sub('s$|es$|e$', '', text),'upos':upos})
                liste_pos_confidence_final.append({'id':id_lemme,'text':text,'lemma': re.sub('s$|es$|e$', '', text),'upos':upos,"confidence":confidence})

            else:
                liste_pos_final.append({'id':id_lemme,'text':text,'lemma': lemma[id_lemme],'upos':upos})
                liste_pos_confidence_final.append({'id':id_lemme,'text':text,'lemma': lemma[id_lemme],'upos':upos,"confidence":confidence})

            
            id_lemme = id_lemme+1
            
        return liste_pos_final, liste_pos_confidence_final


    #### APPLICATION DU MODELE NER DE FLAIR POUR RETROUVER OU ET QUELS SONT LES EG ET TOPO ####

    def get_nertag(self,sent):

        index_final,index_all_final,entity_all_final = [],[],[]
        tag_ner_final = []
        text_ner_final = []
        confidence_score_final = []

        sentence = Sentence(sent)
        self.model_ner.predict(sentence)

        for entity in sentence.get_spans('ner'):
            text_ner_final.append(entity.text)
            tag_ner_final.append(entity.labels)
            confidence_score_final.append(entity.score)

        index,index_all,entity_all = self.get_index_entity(sentence.get_spans('ner'))

        index_tag_final = [{"id_ner" : index[j], "label" : tag_ner_final[j][0].value,"confidence" : confidence_score_final[j]} for j in range(0,(len(index)))]

        return index_tag_final,text_ner_final


        #### REGROUPEMENT DES TOPONYMES ET EG DANS liste_pos_final ####
    #### exemple : centre ville = 2 tokens au départ alors que c'est une seule EG donc on fusionne pour avoir 1 seul token ####

    def group_nertag(self,liste_pos_final, index_tag_final,doc):
        liste_pos_final_group = copy.deepcopy(liste_pos_final)
        id_tag_tot = []
        tag_tot = []
        to_del = []
        for index_ner in index_tag_final : 
            if(len(index_ner["id_ner"])>1):
                if(index_ner['label'] == "TOPONYME"):
                    liste_pos_final_group[index_ner["id_ner"][0]]["upos"] = "PROPN"
                    liste_pos_final_group[index_ner["id_ner"][0]]["lemma"] =  " ".join([liste_pos_final_group[j]["text"] for j in index_ner["id_ner"]])
                    liste_pos_final_group[index_ner["id_ner"][0]]["text"] =  " ".join([liste_pos_final_group[j]["text"] for j in index_ner["id_ner"]])

                else : 
                    liste_pos_final_group[index_ner["id_ner"][0]]["upos"] = "NOUN"
                    liste_pos_final_group[index_ner["id_ner"][0]]["lemma"] =  " ".join([liste_pos_final_group[j]["lemma"] for j in index_ner["id_ner"]])
                    liste_pos_final_group[index_ner["id_ner"][0]]["text"] =  " ".join([liste_pos_final_group[j]["text"] for j in index_ner["id_ner"]])

                to_del.append(index_ner['id_ner'][1:])

            elif(index_ner['label'] == "TOPONYME"):
                    liste_pos_final_group[index_ner["id_ner"][0]]["upos"] = "PROPN"
            else : 
                    liste_pos_final_group[index_ner["id_ner"][0]]["upos"] = "NOUN"

            id_tag_tot.append(index_ner["id_ner"][0])
            tag_tot.append(index_ner["label"])

        j=0
        id_delete = []
        for id_del in list(itertools.chain(*to_del)): 
            del liste_pos_final_group[id_del-j]
            j=j+1
            id_delete.append(id_del)

        k=0
        for id_ner in range(0,len(id_tag_tot)) : 

            if(k<len(id_delete)):
                while((id_tag_tot[id_ner]>=id_delete[k])):
                    k=k+1
                    if(k>=len(id_delete)):
                        break
            id_tag_tot[id_ner] = id_tag_tot[id_ner] - k

        for id_pos in range(0,len(liste_pos_final_group)) :
            liste_pos_final_group[id_pos]["id"] = id_pos+1

        ## RETROUVER PHRASE COMME DANS DOC ##
        i= 0
        phrase = [len(doc.sentences[j].words) for j in range(0,len(doc.sentences))]
        nb_count = 0
        nb_tot = 0
        tot_phrase = []
        for del_id in id_delete:
            if(del_id < phrase[i]+nb_tot): 
                nb_count +=1
            else : 
                tot_phrase.append(phrase[i]-nb_count)
                nb_tot = nb_tot + nb_count
                nb_count = 1
                i +=1
        tot_phrase.append(phrase[i]-nb_count)
        if(i<len(doc.sentences)):
            len_phrase = [len(doc.sentences[j].words) for j in range(i+1,len(doc.sentences))]
            tot_phrase.extend(len_phrase)

        ##BON FORMAT##
        liste_tot =[]
        nb = 0
        for i in range(0,len(tot_phrase)) : 
            liste_tot.append(liste_pos_final_group[nb:tot_phrase[i]+nb])
            nb = nb + tot_phrase[i]

        id_tag_final = [[] for i in range(0,len(tot_phrase))]
        for id_tag in id_tag_tot : 
            i=0
            nb = tot_phrase[i]
            while(id_tag >= nb):
                i +=1
                nb = nb + tot_phrase[i]
            if(i>=1):
                id_tag_final[i].append(id_tag-sum([tot_phrase[j] for j in range(0,i)]))
            else:
                id_tag_final[i].append(id_tag)
        
        
        tag_tot_final = [[] for i in range(0,len(tot_phrase))]
        nb = 0
        for i in range(0,len(id_tag_final)):
            tag_tot_final[i].extend(tag_tot[nb:nb+len(id_tag_final[i])])
            nb = nb+len(id_tag_final[i])
        
        liste_tot_final = []
        for i in range(0,len(liste_tot)):
            liste = []
            for j in range(0,len(liste_tot[i])):
                test = liste_tot[i][j]
                test["id"] = j+1
                liste.append(test)
            liste_tot_final.append(liste)
            
        return liste_tot_final, id_tag_final, tag_tot_final
