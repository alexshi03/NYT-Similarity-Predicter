# NYT-Similarity Predicter

## Motivation
With the NYT claiming that OpenAI built ChatGPT by copying articles from the NYT, we were wondering how similar are the responses that ChatGPT outputs to the original articles. As such, we aim to quantify the similarity of writing style between ChatGPT and articles from the NYT. We also seek to know whether ChatGPT truly gives “Times content particular emphasis.” We are interested in engaging with the third argument of copyright infringement and fair use from the three claims made by the NYT:  “Defendants’ GenAI tools can generate output that recites Times content verbatim, closely summarizes it, and mimics its expressive style.” To study expressive style, our project applies machine learning methods in linguistics to (1) determine heuristics notable between NYT and non-NYT articles and (2) classify what probability GPT-generated articles contain and/or are identical to NYT articles.

## To run
To run the extractor, load pdfs so that the repository structure is ./texts/NYT/genre/yourpdf.pdf

Then, run
```
python parser.py
```
This will give you a .csv file in outputs with the cleaned data from the articles. Run the feature extractor using
```
python extract_features.py --csv ./path/to/your/csv --output OUTPUT_FOLDER_NAME
```
This will create a folder with three files: `advanced_features.csv`, `articles_with_features.csv`, and `basic_features.csv`

Train your model using
```
python training.py ./path/to/your/articles_with_features.csv
```
Finally, evaluate your trained model using
```
python prediction.py
```
Change lines 7 and 10 based on paths to your model and articles_with_features that you want to test.

## Disclaimer
This project is intended to work only on pdfs downloaded using the print to pdf command from the NYT, the FT, and the Washington Post. It has not been tested on other sources, and will probably not work.