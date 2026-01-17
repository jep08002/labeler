README

For batch processing of the 560k cxr reports I have them preprocessed in 10k "chunks"
in /Jake_cxr_reports/Preprocessed_reports. 

Within this folder are the python files needed to process these chunks.  

You may need to update the first line of chunk_label_cdw.py to point to the location of
your venv to ensure the correct python build is invoked.  

1) First create and activate your venv.  Then navigate to this folder in the terminal:
\OneDrive - Department of Veterans Affairs\General - BARDA ARDS Project West Haven Team\Jake_cxr_reports

2) Look to see what the next chunk that needs to be done is and then execute:
python chunk_label_cdw.py --infile cxr_cleaned_chunk000n.csv --outfile-base cxr_labeled_chunk000n --save-every 1500
replacing the two "chunk000n.csv" with the appropriate chunk numbers.  

3) The script will run and label all 10k reports in the chunk saving output files every 1500 reports,
or as often as you specify.  When complete, it will compile all these 1500 report files into one
10k report labeled file with the suffix MASTER.

4) Check the master output and make sure it looks ok

5) Move the master output to "/Completed Labeled Chunks" and the now complete cleaned file to 
"/Completed Cleaned Chunks" and delete the "cxr_labeled_chunk000n_000n" files that you no longer need.

6) Repeat
