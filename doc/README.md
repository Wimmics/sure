# SURE-KG RDF Data Modeling

The description of each advertisements from the corpus comes in three parts: (1) metadata such as description, price, floorSize, (2) annotations about spatial named entities, and (3) vague and uncertain spatial description. Below we exemplify and described these parts. 

A notebook gives some examples of queries and visualization. 

## Namespaces

Below we use the following namespaces:

```turtle
@prefix rdfs:       <http://www.w3.org/2000/01/rdf-schema#>.
@prefix owl:        <http://www.w3.org/2002/07/owl#>.
@prefix xsd:        <http://www.w3.org/2001/XMLSchema#> .

@prefix dcat:       <http://www.w3.org/ns/dcat#>.
@prefix dc:        <http://purl.org/dc/elements/1.1/>.
@prefix dct:        <http://purl.org/dc/terms/>.
@prefix foaf:       <http://xmlns.com/foaf/0.1/>.
@prefix oa:         <http://www.w3.org/ns/oa#>.
@prefix prov:       <http://www.w3.org/ns/prov#>.
@prefix schema:		<http://schema.org/>.
@prefix sd:     	<http://www.w3.org/ns/sparql-service-description#>
@prefix void:       <http://rdfs.org/ns/void#>.

@prefix sure:      <http://ns.inria.fr/sure#>.
@prefix suredt:    <http://ns.inria.fr/sure/data/>.

```

## Ads metadata

Ads URIs are formatted as `http://ns.inria.fr/sure/data/RealEstate+id` where "id" is the idAds found in the initial dataset.

Ads metadata includes the following items:
- type (`schema:Apartment` or `schema:House`)
- floorSize (`schema.floorSize`)
- price (`sure.hasPrice`)
- Number of rooms (`schema.numberOfRooms`)
- Geometry (`geosparql.hasGeometry`)
- Insee Code (`dbo.inseeCode`)
- City (`sure:locatedIn`)
- Spatial Relations (`sure:locatedIn`)

## Spatial Named entities

The spatial named entities identified in an article are described as **annotations** using the **[Web Annotations Vocabulary](https://www.w3.org/TR/annotation-vocab/)**.
Each annotation consists of the following information:
- the annotation target (`oa:hasTarget`) describes the piece of text identified as a named entity as follows:
    - the source (`oa:hasSource`) is the part of the ad description where the named entity was detected
    - the selecor (`oa:hasSelector`) gives the named entity raw text (`oa:exact`) and its location whithin the source (`oa:start` and `oa:end`)
- the annotation body (`oa:hasBody`) gives the URI of the resource identified as representing the named entity (e.g., `sure:Toponym`)


## Vague and Uncertain Spatial description

The spatial named entities extracted from an the description of an ad give information about places. However, they are often vague and uncertain so we represent their boundaries thanks to the fuzzy set theory (alpha-cut). 

A place is compound of : 
- type of place : Absolute or Relative (`rdf:type`)
- name (`rdfs:label`)
- feature (`rdf:type`)
- attributes (`sure.hasAttribute`)
- Geometry : collection of Alpha-Cut (`geosparql.hasGeometry`)

An Alpha-cut is identified by its WKT representation  (`geosparql.asWKT`) and its alpha (`sure:hasAlpha`).