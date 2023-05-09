import pandas as pd
import ast 
import geopandas as gpd
from rdflib import Namespace
from rdflib import Graph, URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS, OWL, XSD, TIME,SKOS
import unidecode
import geocoder

import utils.GraphGeneration as GraphGeneration

graphGeneration = GraphGeneration.GraphGeneration()


### IMPORT DATA ###
print("IMPORT DATA")
df_ads_final = pd.read_csv("../../dataset/initial_dataset.csv",sep=';')
df_finalKG = pd.read_pickle("../SpatialApproximation/results/dfFinaleKG.pkl")
df_annot = pd.read_csv("../TextInformationExtraction/results/data_final_annot.csv",sep=";")
attributes_rdf = pd.read_pickle("./SpatialApproximation/results/attributes_rdf.pkl")
df_feature = pd.read_pickle("../SpatialApproximation/results/dfFeature.pkl")


def remove_char(x,char_remove):
    if(x!=None):
        for char in char_remove:
            x = x.replace(char," ")
        return " ".join(x.split())
    else:
        return None


### IMPORT NAMESPACE ### 
sure = Namespace("https://ns.inria.fr/sure#")
suredt = Namespace("https://ns.inria.fr/sure/data/")

geosparql = Namespace("http://www.opengis.net/ont/geosparql#")
sf = Namespace("http://www.opengis.net/ont/sf#")
geo = Namespace("http://www.w3.org/2003/01/geo/wgs84_pos#")
schema = Namespace("http://schema.org/")
geonames = Namespace("http://www.geonames.org/")
oa =  Namespace("http://www.w3.org/ns/oa#")
dc =  Namespace("http://purl.org/dc/elements/1.1/")
dctypes = Namespace("http://purl.org/dc/dcmitype/")
dcterms = Namespace("http://purl.org/dc/terms/format/")
dbo = Namespace("http://dbpedia.org/ontology/")

### CREATE GRAPH ###
g = Graph()
g.bind("geosparql", geosparql)
g.bind("schema", schema)

g.bind("sf", sf)
g.bind("geo", geo)
g.bind("sure", sure)
g.bind("geonames", geonames)
g.bind("oa", oa)
g.bind("dc", dc)
g.bind("dctypes", dctypes)
g.bind("dcterms", dcterms)
g.bind("dbo", dbo)
g.bind("owl", OWL)

g.bind("suredt", suredt)


g.add((sure.RealEstate,  RDFS.subClassOf, geosparql.Feature))


g.add((sure.AbsolutePlace,  RDFS.subClassOf, geosparql.Feature))
g.add((sure.RelativePlace,  RDFS.subClassOf, geosparql.Feature))



g.add((sure.Amenity,  RDFS.subClassOf, geosparql.Feature))
g.add((sure.LocativeArea,  RDFS.subClassOf, geosparql.Feature))

g.add((sure.AlphaCut,  RDFS.subClassOf, sf.Polygon))
g.add((sure.hasAlpha,  RDF.type, RDF.Property))
g.add((sure.hasAlpha,  RDFS.domain, sure.AlphaCut))
g.add((sure.hasAlpha,  RDFS.range, XSD.double))


g.add((sure.hasPrice,  RDF.type, RDF.Property))
g.add((sure.hasPrice,  RDFS.domain, sure.RealEstate))
g.add((sure.hasPrice,  RDFS.range, XSD.double))


g.add((sure.confidence,  RDF.type, RDF.Property))
g.add((sure.confidence,  RDFS.domain, oa.Annotation))
g.add((sure.confidence,  RDFS.range, XSD.double))




for eg in df_finalKG.eg_clean_new_4.value_counts().to_dict().keys():
    eg_node = URIRef(sure+"".join(list(map(str.capitalize, eg.split(" ")))))
    g.add((eg_node, RDF.type, OWL.Class))

    if(df_feature[df_feature.Feature==eg].FeatureType.values[0]=="Amenity"): 
        g.add((eg_node, RDFS.subClassOf,sure.Amenity))
    else:
        g.add((eg_node, RDFS.subClassOf,sure.LocativeArea))
    
    if(df_feature[df_feature.Feature==eg].Narrower.values[0]!=None):
        g.add((eg_node,RDFS.subClassOf,URIRef(sure+"".join(list(map(str.capitalize, df_feature[df_feature.Feature==eg].Narrower.values[0].split(" ")))))))
    
    
for city in df_ads_final.city.unique():  
    city = remove_char(unidecode.unidecode(city.lower()),["-","'","^"])
    print(city)
    node_city = URIRef(suredt+"".join(list(map(str.capitalize, city.split(" ")))))
    g.add((node_city,RDF.type,sure.City))
    g.add((node_city,RDFS.label,Literal(city)))
    geo_code = geocoder.geonames(city+", Alpes-Maritimes",name_equals=city,maxRows=1, country=['FR'], key='geo_tag_phd',featureClass=['A'],featureCode='ADM4',orderby=["relevance"])
    if(geo_code.geonames_id != None): 
        node_gns = URIRef(geonames+str(geo_code.geonames_id))
        g.add((node_city,OWL.sameAs,node_gns))

    else:
        geo_code = geocoder.geonames(city+", Alpes-Maritimes",maxRows=1, country=['FR'], key='geo_tag_phd',featureClass=['A'],featureCode='ADM4',orderby=["relevance"])
        if(geo_code.geonames_id !=None): 
            node_gns = URIRef(geonames+str(geo_code.geonames_id))
            g.add((node_city,OWL.sameAs,node_gns))

        else: 
            geo_code = geocoder.geonames(city+", Alpes-Maritimes",name_equals=city,maxRows=1, country=['FR'], key='geo_tag_phd',featureClass=['A','P'],orderby=["relevance"])
            if(geo_code.geonames_id !=None): 
                node_gns = URIRef(geonames+str(geo_code.geonames_id))
                g.add((node_city,OWL.sameAs,node_gns))
    
### STORE ADVERTISEMETNS AND ANNOTATIONS ###

print("ADVERTISEMENTS")
i=0
ads = 0
for ads in df_ads_final.idAds.unique():
    if(ads%100==0):
        print(ads)
    
    city = remove_char(unidecode.unidecode(df_ads_final[df_ads_final.idAds==ads].city.iloc[0].lower()),["-","'","^"])
    node_city = URIRef(suredt+"".join(list(map(str.capitalize, city.split(" ")))))
    
    g = graphGeneration.graph_RE(ads,node_city,df_ads_final,df_annot,g)
    ads = ads+1
    
    for index,row in df_annot[df_annot.idAds==ads].iterrows() :
        tag = ast.literal_eval(row["tag"])
        g = graphGeneration.graph_Annot(g,ads,tag,i)
        i=i+1


### IMPORT SPATIAL ENTITIES AND RELATIONS ###
print("SPATIAL")
l=0
for inseeCode in df_ads_final.inseeCode.unique() :
    print(inseeCode)
    df_toRDF = df_finalKG[df_finalKG.idAds.isin(df_ads_final[df_ads_final.inseeCode==inseeCode].idAds.unique())]
    df_final = pd.read_pickle("./data/geo/geocoding_"+inseeCode+".pkl")
    gdf=gpd.GeoDataFrame(df_final,crs=4326)
    
    count = pd.DataFrame(df_toRDF.et_eg_toponym_new.value_counts())
    place_to_estimate = list(count[count['et_eg_toponym_new']>=10].index)
    
    for place in place_to_estimate : 
        g = graphGeneration.graph_place(g,df_toRDF,place,inseeCode,l,gdf)
        l=l+1
    
    for place in attributes_rdf[attributes_rdf.inseeCode==inseeCode].eg_toponym.unique() : 
        g = graphGeneration.graph_attributes(g,attributes_rdf,place,inseeCode,attributes_rdf)


### STORE GRAPH ###
print("STORE GRAPH")
g.serialize(destination="./data.ttl",format="turtle")
# g.serialize(destination="./data.rdf",format="xml")
print("END")