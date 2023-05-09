import numpy as np
import pandas as pd
from pymongo import MongoClient

import flair
from flair.models import SequenceTagger

import stanza
from stanza.models.common.doc import Document

import utils.Preprocessing as preprocessing
import utils.DependencyParsing as dep
import utils.RelationExtraction as RelationExtraction
import utils.AppRelation as AppRelation
import utils.ExtractAds as ExtractAds
import information_extraction.InfoExtraction as InfoExtraction

import pickle
import os
import sys

stanza.download('fr')


model_ner = SequenceTagger.load('./model_flair_eg_rs_correction/best-model.pt')
model_pos = SequenceTagger.load('./model_pos_best/final-model.pt')
nlp = stanza.Pipeline(lang='fr', processors='depparse', depparse_pretagged=True)
nlp_mwt = stanza.Pipeline(lang='fr', processors='tokenize,pos,mwt,lemma')

new_dep = dep.DependancyParsing(model_ner,model_pos,nlp,nlp_mwt)
new_app = AppRelation.AppRelation(new_dep)
new_extractor = InfoExtraction.InfoExtraction(new_app,nlp)


dfAds = pd.read_csv("../../dataset/initial_dataset.csv",sep=';')

nb_ads = len(dfAds)

#print(dfAds)

print("### EXTRACT INFORMATION FROM ADS ###")

df_finalKG = pd.DataFrame()
df_finalAnnot = pd.DataFrame()
for i in range(nb_ads) :
    phrase = dfAds["description"].iloc[i]
    try:
        infoAds = new_extractor.extract_information(phrase)

        #### RELATIONS ####
        data_attr_eg = new_extractor.reAttributes(infoAds.attributs_eg)
        data_attr_et = new_extractor.reAttributes(infoAds.attributs_et)
        data_composition= new_extractor.reComposition(infoAds,data_attr_eg)
        data_spatial= new_extractor.reSpatial(infoAds,data_composition,data_attr_eg)
        data_transport= new_extractor.reTransport(infoAds,data_composition,data_attr_eg)

        data_final = data_spatial.reset_index(drop=True)
        data_final = new_extractor.reGroupTransport(data_final, data_transport)
        data_final = new_extractor.reGroupEntity(data_final, data_composition,infoAds)
        data_final = new_extractor.reAddEntityIN(infoAds,data_final, data_composition)
        data_final = data_final.reset_index(drop=True)
        data_final = new_extractor.reAddAtributesET(data_final,data_attr_et)
        data_final = new_extractor.reAddAtributesEG(data_final,data_attr_eg)
        data_final = new_extractor.reDeleteSpatial(data_final)
        data_final = data_final.drop(["entity","id_et","id_entity","id_phrase","id_eg"],axis=1)
        data_final["idAds"] = dfAds["idAds"][i]

        df_finalKG = df_finalKG.append(data_final)

        ## ANNOTATIONS ##
        data_annot = pd.DataFrame({"sent":infoAds.sent,"tag":infoAds.index_tag_final})
        data_annot["idAds"] = dfAds["idAds"][i]

        df_finalAnnot = df_finalAnnot.append(data_annot)

    except :
        print('Problem with ads/token')

if df_finalKG.shape[0] > 0 :
    df_finalKG = df_finalKG.set_index('idAds')
    df_finalKG.to_csv(os.path.join("./results/data_final_extract.csv"),sep=";")

if df_finalAnnot.shape[0] > 0 :
    df_finalAnnot = df_finalAnnot.set_index('idAds')
    df_finalAnnot.to_csv(os.path.join("./results/data_final_annot.csv"),sep=";")