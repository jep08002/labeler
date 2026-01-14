[Readme.txt](https://github.com/user-attachments/files/24623929/Readme.txt)
README

Done in a virtual environment within VA-1East working directly out of onedrive.

labeler.py updated to labeler_fast.py which calls the nlp function only once per document and compiles lemmas and regex expressions once per run.
Reduces per report runtime from 14 seconds to 0.11 seconds.  Both labeler.py and labeler_fast.py now reflect these changes and are identical. 

Order of operations:
1) Run cdw_report_preprocess.py 
	Takes --infile and --outfile parameters
2) Run batch_label_cdw.py
	Also takes --infile and --outfile
