import pandas as pd
import ast 
import geopandas as gpd
from rdflib import Namespace
from rdflib import Graph, URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS, OWL, XSD, TIME,SKOS
import unidecode
import geocoder
from shapely.ops import unary_union

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

class GraphGeneration :
    
    def graph_Annot(self,g,idAds,tag,i):

        dict_label = {"TOPONYME":"Toponym","EG":"Feature","ET":"SpatialRelation","MD":"Transport"}

        annot = suredt.Entity+str(i)
        g.add((annot, RDF.type ,oa.Annotation))
        g.add((annot, sure.confidence ,Literal(float(round(tag["confidence"],2)))))
        g.add((annot, oa.hasBody, URIRef(sure+dict_label[tag["label"]])))
        g.add((annot, oa.motivatedBy, oa.classifying ))
        
        source = BNode()
        selector = BNode()
        
        g.add((annot, oa.hasTarget, source))
        g.add((source,oa.hasSource, URIRef(suredt.Text+idAds)))
        g.add((annot, oa.hasTarget, selector))
        g.add((selector, RDF.type, oa.TextPositionSelector))
        g.add((selector, oa.start, Literal(float(tag["id_ner"][0]))))
        g.add((selector, oa.end, Literal(float(tag["id_ner"][-1]))))
        
            
        return g


    def graph_RE(self,idAds,node_city,df_ads_final,data_annot,g) : 
        
        """
        1. URI ANNONCE TYPE SPATIAL OBJECT HOUSING
        2. URI TYPE BATIMENT (APPARTEMENT OU MAISON)
        3. URI has surface, price, number of rooms, floor
        2. GEOMETRY : lat/long ou polygone flou
        5. ANNEXE
        7. LOCALISATION VILLE / DPT
        8. Texte

        """
        
        house = suredt.RealEstate+idAds
        g.add((house, RDF.type ,sure.RealEstate))
        
        if(df_ads_final[df_ads_final.idAds==idAds]["propertyType"].values[0]=="Apartment"): 
            g.add((house, RDF.type , schema.Apartment))
        else:
            g.add((house, RDF.type , schema.House))
        
        
        if(df_ads_final[df_ads_final.idAds==idAds]["floorSize"].isna().values[0]==False):
            g.add((house, schema.floorSize, Literal(float(df_ads_final[df_ads_final.idAds==idAds]["surface"].values[0]))))
        
        if(df_ads_final[df_ads_final.idAds==idAds]["price"].isna().values[0]==False):
            g.add((house, sure.hasPrice, Literal(float(df_ads_final[df_ads_final.idAds==idAds]["prix"].values[0]))))
        
        if(df_ads_final[df_ads_final.idAds==idAds]["roomCount"].isna().values[0]==False):
            g.add((house, schema.numberOfRooms, Literal(float(df_ads_final[df_ads_final.idAds==idAds]["roomCount"].values[0]))))
    
        ## Gemoetry
        g.add((house, geo.lat , Literal(df_ads_final[df_ads_final.idAds==idAds]["lat"].values[0])))
        g.add((house, geo.long , Literal(df_ads_final[df_ads_final.idAds==idAds]["long"].values[0])))
        
        point = suredt.Point+str(df_ads_final[df_ads_final.idAds==idAds]["idAds"].index[0])
        g.add((point, RDF.type, sf.Point))
        point_wkt = Literal("POINT("+ str(df_ads_final[df_ads_final.idAds==idAds]["long"].values[0]) + " "+ str(df_ads_final[df_ads_final.idAds==idAds]["lat"].values[0])+")",datatype=geosparql.wktLiteral)
        g.add((point, geosparql.asWKT, point_wkt))
        g.add((house, geosparql.hasGeometry, point))
        
        
        g.add((house, dbo.inseeCode, Literal(df_ads_final[df_ads_final.idAds==idAds]["inseeCode"].values[0])))
        
        
        if(data_annot[data_annot.idAds==idAds].empty==False):
            ads = URIRef(suredt.Text+idAds)
            text = Literal(data_annot[data_annot.idAds==idAds]["sent"].iloc[0])    
            g.add((house, sure.hasDescription, ads))
            g.add((ads,RDF.type,dctypes.Text))
            g.add((ads,dc.language,Literal("fr")))
        #     g.add((ads,dcterms.format,Literal("text/plain")))
            g.add((ads,dc.description,text))
        

                    
        g.add((house, sure.locatedIn, node_city))
        
        g.add((node_city, RDF.type, sure.City))

        return g

    def graph_attributes(self,g,df,place,inseeCode,attributes_rdf):
        
        eg = df[df.eg_toponym==place].eg_clean_new_4.iloc[0]
        topo = df[df.eg_toponym==place].toponym_clean_new_2.iloc[0]
        attributes = list(attributes_rdf[attributes_rdf.eg_toponym==place].attributes_clean.values)  
        
        if(eg!=None):
            node = URIRef(suredt+"_".join(list(map(str.capitalize,eg.lower().split(" "))))+"_"+"_".join(list(map(str.capitalize,topo.lower().split(" "))))+"_"+inseeCode)
        else:
            node = URIRef(suredt+"_".join(list(map(str.capitalize,topo.lower().split(" "))))+"_"+inseeCode)
        g.add((node, RDF.type, sure.AbsolutePlace))
        for attr in attributes : 
            g.add((node, sure.hasAttribute, Literal(attr)))
        
            
        
        return g

    def graph_place(self,g,df_toRDF,place,inseeCode,l,gdf):
        
        eg = df_toRDF[df_toRDF.et_eg_toponym_new==place].eg_clean_new_4.iloc[0]
        topo = df_toRDF[df_toRDF.et_eg_toponym_new==place].toponym_clean_new_2.iloc[0]
        et = df_toRDF[df_toRDF.et_eg_toponym_new==place].et_tot.iloc[0]

        annonce_loc = df_toRDF[df_toRDF.et_eg_toponym_new==place].idAds.unique()
        
        if(et==None): 
            if(topo!=None):
                if(eg!=None):
                    eg_node = URIRef(sure+"".join(list(map(str.capitalize, eg.lower().split(" ")))))
                    node = URIRef(suredt+"_".join(list(map(str.capitalize,eg.lower().split(" "))))+"_"+"_".join(list(map(str.capitalize,topo.lower().split(" "))))+"_"+inseeCode)
                    g.add((node, RDF.type, eg_node))
                    g.add((node, RDF.type, sure.AbsolutePlace))
                    g.add((node, RDFS.label,Literal(topo)))
                else:
                    node = URIRef(suredt+"_".join(list(map(str.capitalize,topo.lower().split(" "))))+"_"+inseeCode)
                    g.add((node, RDF.type, sure.AbsolutePlace))
                    g.add((node, RDFS.label,Literal(topo)))

            else:
                if(eg!=None):
                    eg_node = URIRef(sure+"".join(list(map(str.capitalize, eg.lower().split(" ")))))
                    node = URIRef(suredt+"_".join(list(map(str.capitalize,eg.lower().split(" "))))+"_"+inseeCode)
                    g.add((node, RDF.type, eg_node))
                    g.add((node, RDF.type, sure.AbsolutePlace))
        else:
            if(topo!=None):
                if(eg!=None):
                    eg_node = URIRef(sure+"".join(list(map(str.capitalize, eg.split(" ")))))
                    node_in = URIRef(suredt+"_".join(list(map(str.capitalize,eg.lower().split(" "))))+"_"+"_".join(list(map(str.capitalize,topo.lower().split(" "))))+"_"+inseeCode)
                    g.add((node_in, RDF.type, eg_node))
                    g.add((node_in, RDF.type, sure.AbsolutePlace))
                    g.add((node_in, RDFS.label,Literal(topo)))
                    
                    node = URIRef(suredt+"_".join(list(map(str.capitalize,et.lower().split(" "))))+"_"+"_".join(list(map(str.capitalize,eg.lower().split(" "))))+"_"+"_".join(list(map(str.capitalize,topo.lower().split(" "))))+"_"+inseeCode)
                    g.add((node, RDF.type, sure.RelativePlace))
                    g.add((node, sure.hasAnchor,node_in))
                    g.add((node, sure.hasSpatialRelation,Literal(et.lower())))
    #                 g.add((node_out, URIRef(greof+"hasMeasure"),Literal(str(measure))))

                else:
                    node_in = URIRef(suredt+"_".join(list(map(str.capitalize,topo.lower().split(" "))))+"_"+inseeCode)
                    g.add((node_in, RDF.type, sure.AbsolutePlace))
                    g.add((node_in, RDFS.label,Literal(topo)))
                    node = URIRef(suredt+"_".join(list(map(str.capitalize,et.lower().split(" "))))+"_"+"_".join(list(map(str.capitalize,topo.lower().split(" "))))+"_"+inseeCode)
                    g.add((node, RDF.type, sure.RelativePlace))
                    g.add((node, sure.hasAnchor,node_in))
                    g.add((node, sure.hasSpatialRelation,Literal(et.lower())))
            else:
                if(eg!=None):
                    eg_node = URIRef(sure+"".join(list(map(str.capitalize, eg.split(" ")))))
                    node_in = URIRef(suredt+"_".join(list(map(str.capitalize,eg.lower().split(" "))))+"_"+inseeCode)
                    g.add((node_in, RDF.type, eg_node))
                    g.add((node_in, RDF.type, sure.AbsolutePlace))
                    
                    node = URIRef(suredt+"_".join(list(map(str.capitalize,et.lower().split(" "))))+"_"+"_".join(list(map(str.capitalize,eg.lower().split(" "))))+"_"+inseeCode)
                    g.add((node, RDF.type, sure.RelativePlace))
                    g.add((node, sure.hasAnchor,node_in))
                    g.add((node, sure.hasSpatialRelation,Literal(et.lower())))
                
            
        if("LD_"+"_".join(place.split(" ")) in gdf.columns):  
            for alpha in [0.2,0.4,0.6,0.8,1]: 
                g.add((URIRef(suredt.AlphaCut+str(l)+"_"+str(alpha)), RDF.type, sure.AlphaCut))
                poly = Literal(unary_union([item.buffer(0) for item in gdf[gdf["LD_"+"_".join(place.split(" "))]>=alpha].geometry]).wkt,datatype=geosparql.wktLiteral)
                g.add((URIRef(suredt.AlphaCut+str(l)+"_"+str(alpha)), geosparql.asWKT, poly))
                g.add((URIRef(suredt.AlphaCut+str(l)+"_"+str(alpha)),sure.hasAlpha, Literal(str(alpha),datatype=XSD.double)))
                g.add((node, geosparql.hasGeometry, URIRef(suredt.AlphaCut+str(l)+"_"+str(alpha))))
                                
            
        for annonce in annonce_loc:
            g.add((suredt.RealEstate+annonce,sure.locatedIn,node))
            
        
        return g