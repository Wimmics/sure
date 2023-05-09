# SURE : Spatial Uncertainty and Real Estate

This repository contains the piepline and documentation to build and query the SURE-KG RDF datast. 

## SURE-KG dataset

The SURE-KG RDF dataset provides a knowledge graph built from a real dataset to represent Real Estate and Uncertain Spatial Data from Advertisements. It relies on natural language processing and machine learning methods for information extraction, and semantic Web frameworks for representation and integration. It describes more than 100K real estate ads and 6K place-names extracted from French Real Estate advertisements from various online advertiser and located in the French Riviera. It can be exploited by real estate search engines, real estate professionals, or geographers willing to analyze local place-names

## Documentation

- [RDF data modeling](doc/README.md)
- [Generation pipeline](src/pipeline.md)

## SURE Ontology

The SURE (Spatial Uncertainty and Real Estate) ontology namespace is [http://ns.inria.fr/sure/](http://ns.inria.fr/sure/). 
The prefix *sure:* is used to refer to the ontology.

## Downloading and SPARQL Querying

The dataset is downloadable as a set of RDF dumps (in Turtle syntax) from Zenodo: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7885757.svg)](https://doi.org/10.5281/zenodo.7885757)

It can also be queried through our Virtuoso OS SPARQL endpoint http://erebe-vm2.i3s.unice.fr:5000/sparql.
Further details about how named entities are represented in RDF are given in the [RDF Data Modeling](doc/README.md) section.


## License

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
