# ProjectedInteractingDFT 
Projected-interacting densty functional theory implemented in PySCF
The electron-electron interaction is projected onto one or more states, and the projected interaction is introduced into the Kohn-Sham reference sytems. 
Projected hybrids introduce a variable fraction of exact exchange in the projected states, implemented for DFT and TDDFT
Projected-interacting CI introduce full exact exchange and full CI correlation into the projected states. 
Functions euci3 and euci5 allow various treatments of the projected-interacting CI. We use analytic diagonalization of 2x2 CI matrices when the electron-electron interaction is projected onto one state at a time, and use the PySCF full CI solver when projected onto multiple states.

Test example: 
doCr2.py does PiFCI on chromium dimer, projecting onto one Cr valence AO at a time vs. all 12 Cr2 valence AOs requiring a 12-eletron, 15-orbital CAS\
doFluoreneDimer.py does PiFCI on a fluorene dimer diradical, projecting onto monommer-localized singly occupied oritals. 
CoreProjectedHybrids.py tests core-projected hybrids 


