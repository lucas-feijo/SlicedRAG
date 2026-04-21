# SliceGPT Calibration Dataset Comparison

This repository contains the code for the project developed by Gabriele Lacchin and Lucas Feijó for the TUM Practical Course "Building Autonomous Agents and Bots".

## Repository Organization

This repository was made for usage with Google Colab notebooks. The intended usage is to clone this repo into Google Drive using a Colab notebook, and run experiments using the notebooks in `/notebooks`.

The directories are organized as follows:

- [`/src`](src/): core code, evaluation functions
- [`/notebooks`](notebooks/): notebooks used to run the experiments on Google Colab
- [`/data`](data/): model checkpoints, datasets
- [`/SliceGPTModifications`](SliceGPTModifications): our fork of the original SliceGPT repo as a submodule


## External Resources (Google Drive)

All the necessary materials used in the notebooks are available in the following Google Drive folder:  
[https://drive.google.com/drive/folders/1VRkguTVtRkllTlSID9ugy4MHwCacsE_K?usp=sharing
](https://drive.google.com/drive/folders/1r-2Z28wnSf4M3Z0r1wME0tC_UY3FVymT?usp=sharing)

This includes:
- Hidden states
- Pruned models
- QA prediction samples
- Evaluation outputs, intermediate results and other useful materials
These resources are provided so that you do not need to recompute the materials used for the analysis, but can directly run and reproduce the analysis using our precomputed data.

These materials can be accessed from the directories specified inside the notebooks.

Before running the notebooks, make sure to add the shared Google Drive folder to your **“My Drive”**.

### How to add the folder to My Drive

1. Open the shared folder link.
2. Click on **“Add shortcut to Drive”**.
3. Select **“My Drive”** as the destination.

After that, the folder will be accessible in Colab through the mounted Google Drive (e.g., `/content/drive/MyDrive/...`), allowing all resources (hidden states, models, etc.) to be loaded directly.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
