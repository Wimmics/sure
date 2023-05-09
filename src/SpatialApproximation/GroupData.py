import py_stringmatching as sm
import pandas as pd

class GroupData():
    def __init__(self,df_finalKG) :
        self.df_finalKG = df_finalKG

    def remove_char(self,x,char_remove):
        if(x!=None):
            for char in char_remove:
                x = x.replace(char," ")
            return " ".join(x.split())
        else:
            return None
        
    def narrower(self,x,df_narrower): 
        if(x in df_narrower.eg2.unique()) :
            if(df_narrower[df_narrower.eg2==x].me.values[0] > 0) : 
                return df_narrower[df_narrower.eg2==x].eg1.values[0]
        elif(x in df_narrower.eg1.unique()) :
            if(df_narrower[df_narrower.eg1==x].me.values[0] < 0) : 
                return df_narrower[df_narrower.eg1==x].eg2.values[0]
        else:
            return None
            

    def clean_featureType(self,x,df_feature) : 
        if(x.Narrower != None) : 
            if(df_feature[df_feature.Feature==x.Narrower].FeatureType.values[0] != x.FeatureType):
                return df_feature[df_feature.Feature==x.Narrower].FeatureType.values[0]
            else:
                return x.FeatureType
        else:
            return x.FeatureType
        

    def groupSubPlaceName(self,x):
        placeName = ""
        if(str(x["toponym_clean_new_2"])!=None):
            placeName = str(x["toponym_clean_new_2"])
        if(x["eg_clean_new_4"]!=None):
            if(placeName!=""):
                placeName = str(x["eg_clean_new_4"])+" "+ placeName 

        return placeName


    def groupPlaceName(self,x):
        placeName = ""
        if(x["eg_clean_new_4"]!=None):
            placeName = str(x["eg_clean_new_4"])
        if(x["toponym_clean_new_2"]!=None):
            if(placeName==""):
                placeName = str(x["toponym_clean_new_2"])
            else:
                placeName = placeName +" "+str(x["toponym_clean_new_2"])
        if(x["et_tot"]!=None):
            if(placeName!=""):
                placeName = str(x["et_tot"]) +" "+placeName
        return placeName


    def GroupDataMain(self):
        self.df_finalKG["eg_clean_new_4"] = self.df_finalKG["eg_clean_new_4"].apply(lambda x : self.remove_char(x,["/",","]))

        ## Get "SUBCLASS OF" between Features
        df_narrower = pd.DataFrame(columns=["eg1","eg2","me"])
        me = sm.similarity_measure.monge_elkan.MongeElkan(sim_func = sm.similarity_measure.soundex.Soundex().get_raw_score)
        eg_remove = []
        list_to_check  = list(self.df_finalKG["eg_clean_new_4"].unique())
        for eg1 in list_to_check:
            if(eg1 is not None):
                eg_remove.append(eg1)
                list_to_compare = [elem for elem in list_to_check if (elem not in eg_remove)&(elem is not None)]
                for eg2 in list_to_compare:
                    if((eg1!=eg2) & ((set(eg1.split(" ")).issubset(set(eg2.split(" "))))|(set(eg2.split(" ")).issubset(set(eg1.split(" ")))))):
                        res = me.get_raw_score(eg1.split(" "),eg2.split(" ")) - me.get_raw_score(eg2.split(" "),eg1.split(" "))
                        df_narrower.loc[len(df_narrower)] = [eg1,eg2,res]

        ## Get "SUBCLASS OF" Amenity Or LocativeArea
        df_feature = pd.DataFrame()
        df_feature["Feature"] = self.df_finalKG.eg_clean_new_4.unique()[1:]
        df_feature["FeatureType"] = df_feature.Feature.apply(lambda x : "Amenity" if len(self.df_finalKG[(self.df_finalKG.eg_clean_new_4==x)&(self.df_finalKG.et_tot.isna())])==0 else "LocativeArea")

        df_feature["Narrower"] = df_feature.Feature.apply(lambda x : self.narrower(x,df_narrower)) 
        df_feature["FeatureType"] = df_feature.apply(lambda x :self.clean_featureType(x,df_feature),axis=1 )


        ## Group Feature + Toponym

        self.df_finalKG["eg_toponym"] = self.df_finalKG.apply(lambda x : self.groupSubPlaceName(x),axis=1)
        self.df_finalKG["eg_toponym"] = self.df_finalKG.eg_toponym.apply(lambda x : None if x=="" else x) 

        self.df_finalKG["et_eg_toponym_new"] = self.df_finalKG.apply(lambda x : self.groupPlaceName(x),axis=1)
        self.df_finalKG["et_eg_toponym_new"] = self.df_finalKG.et_eg_toponym_new.apply(lambda x : None if x=="" else x) 

        return self.df_finalKG, df_feature   

