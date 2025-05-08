# JacobYe-OCR-Work-Spring-2025
Documentation on work done for the Boston College Language Learning Lab Spring 2025

## pdf_language_detector.py

**what this is**: Combination of two open source libraries - PyMuPdf (used for data extraction and analysis of PDF's) and Lingua (one of the best open source language identifiers). Used to identify multiple languages within PDF documents. 

**what this is for**: 1. Identify languages within glossed text that can then be sent into other OCR recognition models best specified for that language 2. Within large texts, identifying the sections with glossed text.

**what this does**: Labels languages within PDF's, returns an output file of the PDF with transparent highlighting boxes labeling each language. Also provides confidence scores for each detection.

**how to use**: install the required packages and run the script using "python pdf_language_detector.py "path to your document". To specify what languages to look for use "-l", adjust confidence thresholds with "-c", and set a minimum text length for detection with "-m".

**further work that can be done**: Pair this with an OCR model to send the labeled text to. Add a better recognition algorithm for glossed text. I've attempted to do this with pages that rapidly switch between languages, another continuation of this project would be to use existing OCR models fine tuned on glossed text to identify gloss sections of the input files. 

## ocr_api_test.py 

**what this is**: Python script utilizing Azure Document Intellignece for OCR.

**what this is for/what it does**: Converting raw glossed text into machine useable .txt documents.

**how to use:**: Update script with Azure credentials (endpoint and key). Also specify the PDF path and output path for your file within the script. Run with "python ocr_api_test.py" 

**further work that can be done**: Temporarily paused due to price constraints. Could be a useful reference against other open source model.s
