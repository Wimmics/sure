# SURE-KG dataset generation pipeline

Several steps are involved in the generation the SURE-KG RDF dataset.

This folder provides various tools, scripts and mappings files involved in carrying out these steps.


## Text Information Extraction

Directory [TextInformationExtraction](TextInformationExtraction) provides the tools required to **extract (spatial) information from the ads** using our NER model.

The outputs are saved as *pickle* files.
To download the NER models from Zenodo :   

## Spatial Approximation

The second stage of the pipeline consists in the estimation of the limits of each place extracted with the NER model. 
Directory [SpatialApproximation](SpatialApproximation) provides the scrips used to pre-process the extracted Information and approximate booundaries of a place.


## Graph Generation

The translation in RDF of both treatments is carried out using [GraphGeneration](GraphGeneration).

The output is a RDF file in Turtle syntax.