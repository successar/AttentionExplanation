1. Please get access to the MIMIC data from here --  https://mimic.physionet.org/gettingstarted/access/ .
2. This code code needs 2 files from mimic data -- NOTEEVENTS.csv and DIAGNOSES_ICD.csv .

3. First run all cells in `Clean_Discharge_Summaries.ipynb` notebook (Substitute the appropriate path for the mimic files.) . This will extract all discharge summaries from the mimic dataset and join them with icd9 codes. Also, it will train the word2vec model on the discharge summaries.

4. For Diabetes data, run all cells in `MIMIC_Diabetes.ipynb` notebook .
5. For Anemia data, run all cells in `MIMIC_Anemia.ipynb` notebook.