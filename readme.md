# Installation
To run the notebooks, do the following:

Install conda environment:
```
conda env create -f environment.yml
```

Clone a fork of the GENTRL repo enabling use of custom vocabulary. Moses must be reinstalled after installing GENTRL:

```
git clone https://github.com/sebastiandro/GENTRL.git
git clone https://github.com/molecularsets/moses.git
cd GENTRL && python setup.py install && cd .. 
cd moses && python setup.py install && cd ..
```

Download training data:
https://drive.google.com/drive/folders/11ZGar2WRP9d_J5DAYkBebsixGeXZjQa6

Place in the `data` folder