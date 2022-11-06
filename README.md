# MaterialPropertyExtractor
Keyword-based material property extractor for material science articles.

## Dependency

This repo is built with Python 3.9.
Please check `requirements.txt` for the package version.

### External resource

Some functions may need a pre-trained BERT model, which is not included in this repo.
You can get the model from [here](https://drive.google.com/file/d/1I_VqzmrzvcPG0wHCXljHAaIngZW64PL0/view?usp=share_link).
Once it is downloaded to your server or local machine, please move the unzipped folder to the *default* directory `./models`.

## Usage
You can find the argument list and their description in `./pipeline/args.py`.
The entry script is `./extr.py`.
You can find an example of running the extraction pipeline in `./extr.sh`.
Please remember to change the arguments to your specific data location.


