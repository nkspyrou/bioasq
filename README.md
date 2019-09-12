# bioasq

question_classifier_ML.py: the core program of the MSc thesis that can classify clinical questions with machine learning techniques based on several criteria

annotated questions: batch of files with question classifications. If file name starts from m then it is a file with judge B classification. Else is judge A classification. Other letters define type of classification (see in thesis text)

jsonparser.py: parses json files with bioasq question sets, used for initial question pooling

training_7b.json: accumulated questions of BioASQ challenge

cohens_kappa.py: produces kappa index for given classifications

pico annotator.py: used for annotation of questions with PICO & TDPH elements



