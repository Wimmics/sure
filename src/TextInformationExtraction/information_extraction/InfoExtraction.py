import os
import numpy as np

import pandas as pd
import stanza
from stanza.models.common.doc import Document


import networkx as nx


from bisect import bisect

import utils.Preprocessing as preprocessing
import utils.DependencyParsing as dep
import utils.RelationExtraction as RelationExtraction
import utils.AppRelation as AppRelation


class InfoAds():
    def __init__(self,attributs_eg,composition,subclass_eg,spatial,attributs_et,transport,df_nodesEG,df_nodesTOPONYME,df_nodesET,df_nodesMD,sent,index_tag_final) :
        self.attributs_eg = attributs_eg
        self.composition =  composition
        self.subclass_eg = subclass_eg
        self.spatial = spatial
        self.attributs_et = attributs_et
        self.transport  = transport 
        self.df_nodesEG = df_nodesEG
        self.df_nodesTOPONYME = df_nodesTOPONYME
        self.df_nodesET = df_nodesET
        self.df_nodesMD = df_nodesMD
        self.sent = sent
        self.index_tag_final = index_tag_final


class InfoExtraction() : 

    def __init__(self,new_app,nlp) :
        self.new_app = new_app
        self.nlp = nlp 


    def extract_information(self,phrase):
         try:
            pretagged_doc, G, liste_relation, new_rel,sent,index_tag_final = self.new_app.get_nodes(phrase,self.nlp)

            nodesEG = [x for x,y in G.nodes(data=True) if y['label']=="EG"]
            nodesTOPONYME = [x for x,y in G.nodes(data=True) if y['label']=="TOPONYME"]
            nodesET = [x for x,y in G.nodes(data=True) if y['label']=="ET"]
            nodesMD = [x for x,y in G.nodes(data=True) if y['label']=="MD"]
            nodesADJ = [x for x,y in G.nodes(data=True) if y['label']=="ADJ"]
            nodesNUM = [x for x,y in G.nodes(data=True) if y['label']=="NUM"]

            df_nodesEG = pd.DataFrame([[x.split("_")[0],int(x.split("_")[1]),int(x.split("_")[2])] for x in nodesEG], columns=["entity","id_entity","id_phrase"])
            df_nodesTOPONYME = pd.DataFrame([[x.split("_")[0],int(x.split("_")[1]),int(x.split("_")[2])] for x in nodesTOPONYME], columns=["entity","id_entity","id_phrase"])
            df_nodesET = pd.DataFrame([[x.split("_")[0],int(x.split("_")[1]),int(x.split("_")[2])] for x in nodesET], columns=["entity","id_entity","id_phrase"])
            df_nodesMD = pd.DataFrame([[x.split("_")[0],int(x.split("_")[1]),int(x.split("_")[2])] for x in nodesMD], columns=["entity","id_entity","id_phrase"])

            #### ATTRIBUTS et ET ####

            attributs_et = self.new_app.get_relation(nodesET,nodesADJ,pretagged_doc,3,3,new_rel, G).\
                        append(self.new_app.get_relation(nodesET,nodesNUM,pretagged_doc,3,3,new_rel, G))


            ##voir pour regrouper les fixed " à proximité de"
            

            #### ATTRIBUTS et EG ####
            attributs_eg = self.new_app.get_relation(nodesEG,nodesADJ,pretagged_doc,3,4,new_rel, G)

            attributs_eg = attributs_eg.append(self.new_app.get_relation(nodesEG,nodesNUM,pretagged_doc,3,0,new_rel, G)).\
                        append(self.new_app.get_relation(nodesNUM,nodesEG,pretagged_doc,3,0,new_rel, G))

            #### Regrouper ATTRIBUTS ET EG ####
            if(attributs_eg.empty==False) :  
                attributs_eg_group = pd.DataFrame()
                for index, row in attributs_eg.iterrows(): 
                    if(row["ids"][0] < row["ids"][1]) : 
                        attributs_eg_group = attributs_eg_group.append(row)
                attributs_eg_group.loc[:,['id_phrase','length_path','length_words_rel']] = attributs_eg_group.loc[:,['id_phrase','length_path','length_words_rel']].astype(int)

                if(attributs_eg_group.empty==False):
                    liste_pos_final_group, liste_dict_tag_group = new_rel.group_word(attributs_eg_group,"NOUN","EG",False)

                    pretagged_doc = Document(liste_pos_final_group)
                    doc = self.nlp(pretagged_doc)
                    new_rel = RelationExtraction.RelationExtraction(liste_pos_final_group,liste_dict_tag_group)

                    liste_relation = new_rel.relation_dependance(doc)
                    G = nx.DiGraph()
                    G = new_rel.graph_dep(G,liste_relation)

                    nodesEG = [x for x,y in G.nodes(data=True) if y['label']=="EG"]
                    nodesTOPONYME= [x for x,y in G.nodes(data=True) if y['label']=="TOPONYME"]
                    nodesET = [x for x,y in G.nodes(data=True) if y['label']=="ET"]
                    nodesMD = [x for x,y in G.nodes(data=True) if y['label']=="MD"]
                    nodesADJ = [x for x,y in G.nodes(data=True) if y['label']=="ADJ"]
                    nodesNUM = [x for x,y in G.nodes(data=True) if y['label']=="NUM"]

            #### Regrouper TOPONYME et EG ####
            composition = self.new_app.get_relation(nodesEG,nodesTOPONYME,pretagged_doc,4,2,new_rel, G)

            if(composition.empty==False):
                liste_pos_final_group, liste_dict_tag_group = new_rel.group_word(composition,"NOUN","EG",True)

                pretagged_doc = Document(liste_pos_final_group)
                doc = self.nlp(pretagged_doc)
                new_rel = RelationExtraction.RelationExtraction(liste_pos_final_group,liste_dict_tag_group)

                liste_relation = new_rel.relation_dependance(doc)
                G = nx.DiGraph()
                G = new_rel.graph_dep(G,liste_relation)

                nodesEG = [x for x,y in G.nodes(data=True) if y['label']=="EG"]
                nodesTOPONYME = [x for x,y in G.nodes(data=True) if y['label']=="TOPONYME"]
                nodesET = [x for x,y in G.nodes(data=True) if y['label']=="ET"]
                nodesMD =  [x for x,y in G.nodes(data=True) if y['label']=="MD"]
                nodesADJ = [x for x,y in G.nodes(data=True) if y['label']=="ADJ"]
                nodesNUM = [x for x,y in G.nodes(data=True) if y['label']=="NUM"]




            composition = composition.reset_index(drop=True)

            if(attributs_eg.empty==False):
                for index,row in composition.iterrows() :   
                    j_attr = attributs_eg_group[attributs_eg_group["id_phrase"] == row["id_phrase"]]
                    key = [row["ids"][-1] for index,row in j_attr.iterrows()]
                    liste_tuple = []
                    for i in range(0,len(row["ids"])) :
                        indice = bisect(key, row["ids"][i])
                        if(indice!=0):
                            liste_tuple.append(row["ids"][i] + len(j_attr))
                        else:
                            liste_tuple.append(row['ids'][i])
                    composition.at[index,"ids"] = tuple(liste_tuple)


            #### EG - appos - EG ####
            subclass_eg = self.new_app.get_relation(nodesEG,nodesEG,pretagged_doc,4,15,new_rel, G)

            #subclass_topo, rel_topo= new_app.get_relation(nodesEG,nodesTOPONYME,pretagged_doc,4,5,new_rel, G)
            #subclass_topo["id_annonce"] = i

            subclass_eg = subclass_eg.reset_index(drop=True)

            ## KEEP APPOS##
            for index,row in subclass_eg.iterrows() : 
                if(row["relation"][1][0]!="appos") :
                    subclass_eg = subclass_eg.drop(index)

            if(composition.empty==False):
                for index,row in subclass_eg.iterrows() :     
                    j_eg = composition[composition["id_phrase"]==row["id_phrase"]]
                    key = [row["ids"][-1] for index,row in j_eg.iterrows()]
                    liste_tuple = []
                    for i in range(0,len(row["ids"])) :
                        indice = bisect(key, row["ids"][i])
                        if(indice!=0):
                            liste_tuple.append(row["ids"][i] + len(j_eg))
                        else:
                            liste_tuple.append(row['ids'][i])
                    subclass_eg.at[index,"ids"] = tuple(liste_tuple)

            if(attributs_eg.empty==False):
                for index,row in subclass_eg.iterrows() :   
                    j_attr = attributs_eg_group[attributs_eg_group["id_phrase"] == row["id_phrase"]]
                    key = [row["ids"][-1] for index,row in j_attr.iterrows()]
                    liste_tuple = []
                    for i in range(0,len(row["ids"])) :
                        indice = bisect(key, row["ids"][i])
                        if(indice!=0):
                            liste_tuple.append(row["ids"][i] + len(j_attr))
                        else:
                            liste_tuple.append(row['ids'][i])
                    subclass_eg.at[index,"ids"] = tuple(liste_tuple)



        
            #### RELATIONS MODE_TRANSPORT  ####
            spatial = self.new_app.get_relation(nodesET,nodesEG,pretagged_doc,4,15,new_rel, G).\
                    append(self.new_app.get_relation(nodesEG,nodesET,pretagged_doc,4,15,new_rel, G)).\
                    append(self.new_app.get_relation(nodesET,nodesTOPONYME,pretagged_doc,4,15,new_rel, G)).\
                    append(self.new_app.get_relation(nodesTOPONYME,nodesET,pretagged_doc,4,15,new_rel, G))

            spatial = spatial.reset_index(drop=True)

            ###RELATION 3 -> KEEP ONLY ADP, VERB OR MD ? ####
            for index,row in spatial.iterrows() : 
                if(((row["length_path"]==3) & (row["relation"][0][1] not in ["ADP","VERB","MD","ADV","EG"]))) :
                    spatial = spatial.drop(index)

            if(composition.empty==False):
                for index,row in spatial.iterrows() :     
                    j_eg = composition[composition["id_phrase"]==row["id_phrase"]]
                    key = [row["ids"][-1] for index,row in j_eg.iterrows()]
                    liste_tuple = []
                    for i in range(0,len(row["ids"])) :
                        indice = bisect(key, row["ids"][i])
                        if(indice!=0):
                            liste_tuple.append(row["ids"][i] + len(j_eg))
                        else:
                            liste_tuple.append(row['ids'][i])
                    spatial.at[index,"ids"] = tuple(liste_tuple)

            if(attributs_eg.empty==False):
                for index,row in spatial.iterrows() :   
                    j_attr = attributs_eg_group[attributs_eg_group["id_phrase"] == row["id_phrase"]]
                    key = [row["ids"][-1] for index,row in j_attr.iterrows()]
                    liste_tuple = []
                    for i in range(0,len(row["ids"])) :
                        indice = bisect(key, row["ids"][i])
                        if(indice!=0):
                            liste_tuple.append(row["ids"][i] + len(j_attr))
                        else:
                            liste_tuple.append(row['ids'][i])
                    spatial.at[index,"ids"] = tuple(liste_tuple)

            #### RELATIONS MODE_TRANSPORT  ####
            transport = self.new_app.get_relation(nodesET,nodesMD,pretagged_doc,3,3,new_rel, G).\
                    append(self.new_app.get_relation(nodesEG,nodesMD,pretagged_doc,3,3,new_rel, G)).\
                    append(self.new_app.get_relation(nodesTOPONYME,nodesMD,pretagged_doc,3,3,new_rel, G)).\
                    append(self.new_app.get_relation(nodesMD,nodesET,pretagged_doc,3,3,new_rel, G)).\
                    append(self.new_app.get_relation(nodesMD,nodesEG,pretagged_doc,3,3,new_rel, G)).\
                    append(self.new_app.get_relation(nodesMD,nodesTOPONYME,pretagged_doc,3,3,new_rel, G))


            transport = transport.reset_index(drop=True)


            if(composition.empty==False):
                for index,row in transport.iterrows() :     
                    j_eg = composition[composition["id_phrase"]==row["id_phrase"]]
                    key = [row["ids"][-1] for index,row in j_eg.iterrows()]
                    liste_tuple = []
                    for i in range(0,len(row["ids"])) :
                        indice = bisect(key, row["ids"][i])
                        if(indice!=0):
                            liste_tuple.append(row["ids"][i] + len(j_eg))
                        else:
                            liste_tuple.append(row['ids'][i])
                    transport.at[index,"ids"] = tuple(liste_tuple)

            if(attributs_eg.empty==False):
                for index,row in transport.iterrows() :   
                    j_attr = attributs_eg_group[attributs_eg_group["id_phrase"] == row["id_phrase"]]
                    key = [row["ids"][-1] for index,row in j_attr.iterrows()]
                    liste_tuple = []
                    for i in range(0,len(row["ids"])) :
                        indice = bisect(key, row["ids"][i])
                        if(indice!=0):
                            liste_tuple.append(row["ids"][i] + len(j_attr))
                        else:
                            liste_tuple.append(row['ids'][i])
                    transport.at[index,"ids"] = tuple(liste_tuple)

            infoAds = InfoAds(attributs_eg, composition, subclass_eg, spatial, attributs_et, transport, df_nodesEG, df_nodesTOPONYME, df_nodesET, df_nodesMD,sent,index_tag_final)

            return infoAds

         except : 
             print('Problème annonce')
        



    def reAttributes(self,data):
        data_attr = pd.DataFrame(columns=["entity","attribute","id_entity","id_attr","id_phrase"])
        if(data.empty==False):
            for index, row in data.iterrows() : 
                row_to_append = pd.DataFrame([{'entity':row['mots_relation'][0], 'attribute':row['mots_relation'][-1],'id_entity':row["ids"][0],"id_attr":row["ids"][-1], 'id_phrase' : row['id_phrase']}])
                data_attr = pd.concat([data_attr,row_to_append])  
        return data_attr 

    def reComposition(self, infoAds,data):
        data_compositon = pd.DataFrame(columns=["eg","toponym","id_eg","id_topo","id_phrase"])
        if(infoAds.attributs_eg.empty == False) : 
            for index, row in infoAds.composition.iterrows() :
                eg = [row_entity for index_entity,row_entity in data.iterrows() if ((row["ids"][0] == row_entity["id_entity"]) & (row['id_phrase']==row_entity["id_phrase"]))]
                if(eg!=[]) : 
                    row_to_append = pd.DataFrame([{'eg':eg[0]['entity'], 'toponym':row['mots_relation'][-1],'id_eg':eg[0]["id_entity"], "id_topo":row["ids"][-1], 'id_phrase' : row['id_phrase']}])

                else :
                    row_to_append = pd.DataFrame([{'eg':row['mots_relation'][0], 'toponym':row['mots_relation'][-1],'id_eg':row['ids'][0],"id_topo":row['ids'][-1], 'id_phrase' : row['id_phrase']}])
                data_compositon = pd.concat([data_compositon,row_to_append]) 
        else :
            for index, row in infoAds.composition.iterrows() :
                row_to_append = pd.DataFrame([{'eg':row['mots_relation'][0], 'toponym':row['mots_relation'][-1],'id_eg':row['ids'][0],"id_topo":row['ids'][-1], 'id_phrase' : row['id_phrase']}])
                data_compositon = pd.concat([data_compositon,row_to_append])
        
        return data_compositon

    def reSpatial(self,infoAds,data_composition,data_attre_eg):
        data_spatial = pd.DataFrame(columns=["et","entity","id_et","id_entity","id_phrase"])
        if(infoAds.composition.empty == False) : 
            for index, row in infoAds.spatial.iterrows() :
                eg_topo = [row_entity for index_entity,row_entity in data_composition.iterrows() if ((row["ids"][0] == row_entity["id_topo"]) & (row['id_phrase']==row_entity["id_phrase"]))]
                eg_attr = [row_entity for index_entity,row_entity in data_attre_eg.iterrows() if ((row["ids"][0] == row_entity["id_entity"]) & (row['id_phrase']==row_entity["id_phrase"]))]
                if(row["mots_relation"][0] not in infoAds.df_nodesET.entity.unique()):
                    if((row["mots_relation"][1].split(" ")[0] not in infoAds.df_nodesEG.entity.unique()) & (row["mots_relation"][1].split(" ")[0] not in infoAds.df_nodesTOPONYME.entity.unique())):
                        et = -1
                    else:
                        et=0
                else :
                    et =0
                if(eg_topo!=[]) : 
                    row_to_append = pd.DataFrame([{'et':row['mots_relation'][et], 'entity':eg_topo[0]['toponym'],"id_et":row["ids"][et],'id_entity':eg_topo[0]["id_topo"], 'id_phrase' : row['id_phrase']}])

                elif(eg_attr!=[]) : 
                    row_to_append = pd.DataFrame([{'et':row['mots_relation'][et], 'entity':eg_attr[0]['entity'],"id_et":row["ids"][et],'id_entity':eg_attr[0]["id_entity"], 'id_phrase' : row['id_phrase']}])

                else : 
                    if(et==0):
                        row_to_append = pd.DataFrame([{'et':row['mots_relation'][et], 'entity':row['mots_relation'][-1],"id_et":row["ids"][et],'id_entity':row["ids"][-1], 'id_phrase' : row['id_phrase']}])
                    else:
                        row_to_append = pd.DataFrame([{'et':row['mots_relation'][et], 'entity':row['mots_relation'][0],"id_et":row["ids"][et],'id_entity':row["ids"][0], 'id_phrase' : row['id_phrase']}])

                data_spatial = pd.concat([data_spatial,row_to_append])
        else:
            for index, row in infoAds.spatial.iterrows() :
                if(row["mots_relation"][0] not in infoAds.df_nodesET.entity.unique()):
                    if((row["mots_relation"][1].split(" ")[0] not in infoAds.df_nodesEG.entity.unique()) & (row["mots_relation"][1].split(" ")[0] not in infoAds.df_nodesTOPONYME.entity.unique())):
                        et = -1
                    else:
                        et=0
                else :
                    et =0
                
                if(et==0):
                    row_to_append = pd.DataFrame([{'et':row['mots_relation'][et], 'entity':row['mots_relation'][-1],"id_et":row["ids"][et],'id_entity':row["ids"][-1], 'id_phrase' : row['id_phrase']}])
                else:
                    row_to_append = pd.DataFrame([{'et':row['mots_relation'][et], 'entity':row['mots_relation'][0],"id_et":row["ids"][et],'id_entity':row["ids"][0], 'id_phrase' : row['id_phrase']}])

                data_spatial = pd.concat([data_spatial,row_to_append])
        return data_spatial

    def reTransport(self,infoAds,data_composition,data_attre_eg):
        data_transport = pd.DataFrame(columns=["md","entity",'id_md',"id_entity","id_phrase"])
        if(infoAds.composition.empty == False) : 
            for index, row in infoAds.transport.iterrows() :
                eg_topo = [row_entity for index_entity,row_entity in data_composition.iterrows() if ((row["ids"][0] == row_entity["id_topo"]) & (row['id_phrase']==row_entity["id_phrase"]))]
                eg_attr = [row_entity for index_entity,row_entity in data_attre_eg.iterrows() if ((row["ids"][0] == row_entity["id_entity"]) & (row['id_phrase']==row_entity["id_phrase"]))]
                if(row["mots_relation"][0] not in infoAds.df_nodesMD.entity.unique()):
                    md = -1
                else :
                    md =0
                if(eg_topo!=[]) : 
                    row_to_append = pd.DataFrame([{'md':row['mots_relation'][md], 'entity':eg_topo[0]['toponym'],"id_md":row["ids"][md],'id_entity':eg_topo[0]["id_topo"], 'id_phrase' : row['id_phrase']}])

                elif(eg_attr!=[]) : 
                    row_to_append = pd.DataFrame([{'md':row['mots_relation'][md], 'entity':eg_attr[0]['entity'],"id_md":row["ids"][md],'id_entity':eg_attr[0]["id_entity"], 'id_phrase' : row['id_phrase']}])

                else : 
                    if(md==0):
                        row_to_append = pd.DataFrame([{'md':row['mots_relation'][md], 'entity':row['mots_relation'][-1],"id_md":row["ids"][md],'id_entity':row["ids"][-1], 'id_phrase' : row['id_phrase']}])
                    else:
                        row_to_append = pd.DataFrame([{'md':row['mots_relation'][md], 'entity':row['mots_relation'][0],"id_md":row["ids"][md],'id_entity':row["ids"][0], 'id_phrase' : row['id_phrase']}])

                data_transport = pd.concat([data_transport,row_to_append])
        else:
            for index, row in infoAds.transport.iterrows() :
                if(row["mots_relation"][0] not in infoAds.df_nodesMD.entity.unique()):
                    md = -1
                else :
                    md =0
                
                if(md==0):
                    row_to_append = pd.DataFrame([{'md':row['mots_relation'][md], 'entity':row['mots_relation'][-1],"id_md":row["ids"][md],'id_entity':row["ids"][-1], 'id_phrase' : row['id_phrase']}])
                else:
                    row_to_append = pd.DataFrame([{'md':row['mots_relation'][md], 'entity':row['mots_relation'][0],"id_md":row["ids"][md],'id_entity':row["ids"][0], 'id_phrase' : row['id_phrase']}])

                data_transport = pd.concat([data_transport,row_to_append])

        return data_transport

    def reGroupEntity(self,data_final, data_composition,infoAds):
        data_final["eg"] = None
        data_final["toponym"] = None
        data_final["id_eg"] = None
        for ind, row in data_final.iterrows():
            found = False
            for ind_eg,row_eg in data_composition.iterrows() : 
                if((row[["id_entity","id_phrase"]].values == row_eg[["id_eg","id_phrase"]].values).all()):
                    data_final.loc[ind,"eg"] = row_eg["eg"]
                    data_final.loc[ind,"toponym"] = row_eg["toponym"]
                    data_final.loc[ind,"id_eg"] = row_eg["id_eg"]
                    found=True
                elif((row[["id_entity","id_phrase"]].values == row_eg[["id_topo","id_phrase"]].values).all()):
                    data_final.loc[ind,"eg"] = row_eg["eg"]
                    data_final.loc[ind,"toponym"] = row_eg["toponym"]
                    data_final.loc[ind,"id_eg"] = row_eg["id_eg"]
                    found=True
                elif((row["entity"] == row_eg["eg"]+" "+row_eg["toponym"])&(row["id_phrase"]==row_eg["id_phrase"])):
                    data_final.loc[ind,"eg"] = row_eg["eg"]
                    data_final.loc[ind,"toponym"] = row_eg["toponym"]
                    data_final.loc[ind,"id_eg"] = row_eg["id_eg"]
                    found=True
            if(found==False):
                for ind_topo, row_topo in infoAds.df_nodesTOPONYME.iterrows():
                    if((row[["entity","id_phrase"]].values == row_topo[["entity","id_phrase"]].values).all()):
                        data_final.loc[ind,"toponym"] = row_topo["entity"]
                        found = True
            if(found==False):
                for ind_topo, row_topo in infoAds.df_nodesEG.iterrows():
                    if((row[["entity","id_phrase"]].values == row_topo[["entity","id_phrase"]].values).all()):
                        data_final.loc[ind,"eg"] = row_topo["entity"]
                        data_final.loc[ind,"id_eg"] = row_topo["id_entity"]
        return data_final

    def reGroupTransport(self, data_final,data_transport):
        data_final["md"] = None
        for ind_et, et in data_final.iterrows() :
            for ind_md, md in data_transport.iterrows(): 
                if((md[["id_entity","id_phrase"]].values == et[["id_et","id_phrase"]].values).all()):
                    data_final.loc[ind_et,"md"] = md["md"]
                elif((md[["id_entity","id_phrase"]].values == et[["id_entity","id_phrase"]].values).all()):
                    data_final.loc[ind_et,"md"] = md["md"]
        return data_final
    
    def reDeleteSpatial(self, data_final):
        df = data_final[data_final.duplicated(subset=['id_et','id_phrase'], keep=False)]
        for ind_et, et in df.iterrows() :
            if(et["et"]!=None):
                for ind_et1, et1 in df[(df.id_et == et["id_et"])&(df.id_phrase==et["id_phrase"])].iterrows():
                    if(((et1["id_entity"]<et["id_et"]) & (et["id_entity"]>et["id_et"])) | ((et1["id_entity"]>et["id_et"]) & (et["id_entity"]<et["id_et"]))):
                        data_final = data_final.drop(index=ind_et1)
        return data_final

    def reAddAtributesET(self, data_final,data_attr_et):
        data_final["measure"] = None
        for ind_et, et in data_final.iterrows() :
            for ind_attr, attr in data_attr_et.iterrows():
                if((et[["id_et","id_phrase"]].values == attr[["id_entity","id_phrase"]]).all()):
                    data_final.loc[ind_et,"measure"] = attr["attribute"]
        return data_final

    def reAddAtributesEG(self, data_final,data_attr_eg):
        data_final["attributes"] = None
        for ind_et, et in data_final.iterrows() :
            lst_attr = []
            for ind_attr, attr in data_attr_eg.iterrows():
                if((et[["id_eg","id_phrase"]].values == attr[["id_entity","id_phrase"]]).all()):
                    lst_attr.append(attr["attribute"])
            data_final.loc[ind_et,"attributes"] = lst_attr
        return data_final

    def reAddEntityIN(self,infoAds,data_final, data_composition): 
        data_composition = data_composition.reset_index(drop=True)
        for ind,row in data_composition.iterrows() : 
            if((row[["eg","id_eg","id_phrase"]].values == data_final.loc[:,["eg","id_eg","id_phrase"]].values).any(0).all()==False):        
                row_to_append = pd.DataFrame([{"et":None,"id_phrase":row["id_phrase"],"md":None,"eg":row['eg'],"toponym":row["toponym"],"id_eg":row["id_eg"]}])
                data_final = pd.concat([data_final,row_to_append])
                
        for ind_topo,row_topo in infoAds.df_nodesTOPONYME.iterrows() : 
            if((row_topo[["entity","id_phrase"]].values == data_final.loc[:,["toponym","id_phrase"]].values).any(0).all()==False):        
                row_to_append = pd.DataFrame([{"et":None,"id_phrase":row_topo["id_phrase"],"md":None,"eg":None,"toponym":row_topo["entity"],"id_eg":None}])
                data_final = pd.concat([data_final,row_to_append])

        for ind_eg,row_eg in infoAds.df_nodesEG.iterrows() : 
            if((row_eg[["entity","id_phrase"]].values == data_final.loc[:,["eg","id_phrase"]].values).any(0).all()==False):        
                row_to_append = pd.DataFrame([{"et":None,"id_phrase":row_eg["id_phrase"],"md":None,"eg":row_eg["entity"],"toponym":None,"id_eg":row_eg["id_entity"]}])
                data_final = pd.concat([data_final,row_to_append])    
        return data_final        