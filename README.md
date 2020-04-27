# DeepArk
![logo](data/deepark_logo.png)

---

## Contents

1. [What is DeepArk?](#what_is_deepark)
2. [The DeepArk server](#webserver)
3. [Running DeepArk locally](#local_usage)
4. [Frequently asked questions](#faqs)
5. [Citing DeepArk](#citation)
6. [Related tools](#related_tools)

## <a name="what_is_deepark"></a> What is DeepArk?

DeepArk is a set of models of the worm, fish, fly, and mouse regulatory codes.
For each of these organism, we constructed a deep convolutional neural network that predict regulatory activities (i.e. histone modifications, transcription factor binding, and chromatin state) directly from genomic sequences.
Besidese accurately predicting a sequence's regulatory activity, DeepArk can predict the effects of variants on regulatory function and profile sequences regulatory potential with _in silico_ saturated mutagenesis.
If you are a researcher with no programming experience or access to GPUs, please take a look at [our free and user-friendly GPU-accelerated webserver](#webserver).
We also provide instructions for [running DeepArk on your own computer](#local_usage).

## <a name="webserver"></a>The DeepArk server

Like most methods using deep learning, DeepArk is designed to run on graphics processing units (GPUs).
However, we did not intend for DeepArk to only be used by researchers with access to high-end GPU clusters.
To lower this barrier, we are publicly hosting DeepArk on a free GPU-accelerated server [here](https://deepark.princeton.edu).
Documentation and guidelines for using the server may be found [here](https://deepark.princeton.edu/help).
If you need to make a large number (e.g. hundreds of thousands) of predictions with DeepArk, we recommend doing so on your local machine.
Instructions on how to do this are provided [below](#local_usage).

## <a name="local_usage"></a> Running DeepArk locally

This repository is all you should need to run DeepArk on your own computer.
The following subsections describe how to install DeepArk locally, and use it to predict the [regulatory activity of genomic sequences](#prediction_howto), predict the [regulatory effects of variants](#vep_howto), and [profile sequences with _in silico_ saturated mutagenesis](#issm_howto)

### Installation

To install DeepArk locally you should first clone this repository as follows:

```
git clone https://github.com/FunctionLab/DeepArk.git
cd DeepArk
```

We recommend managing DeepArk's dependencies with a conda environment as follows:

```
conda env create -f environment.yml
conda activate DeepArk
```

The default conda environment uses [PyTorch](https://github.com/pytorch/pytorch) and CUDA.
If you do not have access to a CUDA-enabled GPU, we recommend using the GPU-accelerated [DeepArk webserver](#webserver).
However, we have also included a CPU-only conda environment in `cpu_environment.yml` if you cannot use the webserver either.
After downloading DeepArk, you will need to download the weights for the network.
You can download all of the weights by running the `download_weights.sh` script as follows:

```
./download_weights.sh
```

Alternatively, you can download the weights for a subset of the species as follows:

```
./download_weights.sh caenorhabditis_elegans danio_rerio
```

### Usage overview

To start using DeepArk, simply run the `DeepArk.py` script with python.
The `model.py` file contains code to build the DeepArk model in python.
The checkpoints to use with DeepArk are included in the `data` directory as `*.pth.tar` files.
The checkpoints for worm, fish, fly, and mouse are saved in `mus_musculus.pth.tar`, `drosophila_melanogaster.pth.tar`, `danio_rerio.pth.tar`, and `caenorhabditis_elegans.pth.tar` respectively.
It is worth noting that these checkpoint files are slight different from the ones produced by training the models with [Selene](https://github.com/FunctionLab/selene), since we have included the arguments required to construct each model object.
Information on each feature predicted by each model can be found in the `*.tsv` files in the `data` directory (e.g. `mus_musculus.tsv` and so on).
These feature information files are described further in [this section of the FAQ](#features).

We describe each use case for DeepArk in further detail in the sections below, but the general syntax for running a command with DeepArk is as follows:

```
python DeepArk.py [COMMAND] [OPTIONS]
```

You can find a listing of commands and some general usage information by using the following command:

```
python DeepArk.py --help
```

You can also find command-specific usage information like so:

```
python DeepArk.py [COMMAND] --help
```

### <a name="prediction_howto"></a>Regulatory activity prediction

Predicting the regulatory activity of a genomic sequence with a DeepArk model is the most straightforward way to use DeepArk.
To do so, you only need a DeepArk model checkpoint and a [FASTA file](https://en.wikipedia.org/wiki/FASTA_format) with the sequences you would like to make predictions for.
Note that the sequences in the FASTA file should be 4095 bases long.
Below is an example showing how to use DeepArk for prediction.

```
python DeepArk.py predict \
    --checkpoint-file 'data/caenorhabditis_elegans.pth.tar' \
    --input-file 'examples/caenorhabditis_elegans_prediction_example.fasta' \
    --output-dir './' \
    --output-format 'tsv' \
    --batch-size '64'
```

Instead of a FASTA file with 4095 base pair sequences, you can alternatively provide DeepArk with a [BED file](https://genome.ucsc.edu/FAQ/FAQformat.html#format1) specifying regions in a reference genome.
If using DeepArk with a BED file, you must include a FASTA file specifying the reference genome sequence to use with it.
Additional information about where to find a FASTA file for a reference genome is included in [this section below](#reference_genomes).
We include an example of this usage below.

```
python DeepArk.py predict \
    --checkpoint-file 'data/caenorhabditis_elegans.pth.tar' \
    --input-file 'examples/caenorhabditis_elegans_prediction_example.bed' \
    --genome-file 'ce11.fa' \
    --output-dir './' \
    --output-format 'tsv' \
    --batch-size '64'
```

Because of its size, the reference FASTA `ce11.fa` must be downloaded separately from this repository.
More information about where to download this file can be found [here](#reference_genomes).
Finally, further information about `predict` and its arguments may be found using the following invocation of `DeepArk.py`:

```
python DeepArk.py predict --help
```

### <a name="vep_howto"></a>Variant effect prediction

To predict the effects of variants with DeepArk, we simply compare the predicted probabilities of the reference sequence to the mutated sequence containing the variant.
To run make predictions for variants, you will need a DeepArk model checkpoint, a [VCF file](https://samtools.github.io/hts-specs/VCFv4.1.pdf) with your variants, and a [FASTA file](https://en.wikipedia.org/wiki/FASTA_format) with the reference genome sequence.
We show an example invocation below.

```
python DeepArk.py vep \
    --checkpoint-file 'data/mus_musculus.pth.tar' \
    --input-file 'examples/mus_musculus_vep_example.vcf' \
    --genome-file 'mm10.fa' \
    --output-dir './' \
    --output-format 'tsv' \
    --batch-size '64'
```

Because of its size, the reference FASTA `mm10.fa` must be downloaded separately from this repository.
More information about where to download this file can be found [here](#reference_genomes).
Finally, additional information about each argument for `vep` can be found using the following command:

```
python DeepArk.py vep --help
```

### <a name="issm_howto"></a>_In silico_ saturated mutagenesis

_In silico_ saturated mutagenesis (ISSM) allows us to profile the regulatory potential of sequences by predicting the effects of all possible mutations in that sequence.
Note that ISSM generates roughly 17400 predictions per sequence, so it is much slower than the other prediction methods.
To profile sequences with ISSM, you will need a DeepArk model checkpoint and a [FASTA file](https://en.wikipedia.org/wiki/FASTA_format) with at least one entry in it.
Note that the sequences in the FASTA file should be 4095 bases long.
We show an example invocation of ISSM command below.

```
python DeepArk.py issm \
    --checkpoint-file 'data/drosophila_melanogaster.pth.tar' \
    --input-file 'examples/drosophila_melanogaster_issm_example.fasta' \
    --output-dir './' \
    --output-format 'tsv' \
    --batch-size '64'
```

Additional information regarding `issm` and its argument may be found with the following command:

```
python DeepArk.py issm --help
```

## <a name="faqs"></a> Frequently asked questions (FAQs)

1. [What regulatory features predicted by each DeepArk model?](#features)
2. [Where can I download reference genomes to use with DeepArk?](#reference_genomes)
3. [How do I force DeepArk to use or ignore my GPU?](#toggle_cuda)
4. [How can I leverage multiple GPUs with DeepArk?](#data_parallel)
5. [How do I set the number of threads used when I run DeepArk without a GPU?](#threading)
6. [How can I speed up _in silico_ saturated mutagenesis?](#issm_speedup)
7. [How did you train DeepArk?](#training)
8. [How accurate is DeepArk?](#performance)
9. [How do I cite DeepArk?](#citation_faq)
10. [Why are DeepArk's checkpoints different from Selene's?](#checkpoints)

#### <a name="features"></a>1. What regulatory features are predicted by each DeepArk model?

The features predicted by each model are included in the `*.tsv` files in the `data` directory.
The information for worm, fish, fly, and mouse is stored in `caenorhabditis_elegans.tsv`, `danio_rerio.tsv`, `drosophila_melanogaster.tsv`, and `mus_musculus.tsv` respectively.
For a given row in these files, the `index` column specifies the corresponding entry in the DeepArk model output prediction vector.
These index values start at zero.
All of the information and metadata regarding the experiments was sourced from [ChIP-atlas](http://dx.doi.org/10.18908/lsdba.nbdc01558-000).
Additional information about data from ChIP-atlas can be found [here](https://github.com/inutano/chip-atlas/wiki#tables-summarizing-metadata-and-files).

#### <a name="reference_genomes"></a>2. Where can I download reference genomes to use with DeepArk?

There are many possible sources for reference genomes.
For most cases, we recommend downloading genomes from [RefSeq](https://www.ncbi.nlm.nih.gov/refseq/), [ENSEMBL](http://ensemblgenomes.org/), or the [UCSC genome browser](http://hgdownload.soe.ucsc.edu/downloads.html).

#### <a name="toggle_cuda"></a>3. How do I force DeepArk to use or ignore my GPU?

There are a few situations where you are using DeepArk with CUDA-enabled PyTorch on a machine with a GPU, but you do not want to use the GPU to run DeepArk.
Conversely, you may want DeepArk to crash if it cannot use a GPU.
This behavior can be achieved by explicitly specifying whether DeepArk should use a CUDA or not.
To force DeepArk to use or ignore the GPU, set the `--cuda` or `--no-cuda` flag during the invocation of any command.
To demonstrate this, we modify the `vep` example from above to not use the GPU as follows:

```
python DeepArk.py vep \
    --checkpoint-file 'data/mus_musculus.pth.tar' \
    --input-file 'examples/mus_musculus_vep_example.vcf' \
    --genome-file 'mm10.fa' \
    --output-dir './' \
    --output-format 'tsv' \
    --batch-size '64' \
    --no-cuda
```

If you do not explicitly specify whether to use a GPU or not, DeepArk will use `torch.cuda.is_available` to decide.
If it returns `True`, then DeepArk will use the GPU.
Otherwise, DeepArk will not attempt to leverage a GPU.

#### <a name="data_parallel"></a>4. How can I leverage multiple GPUs with DeepArk?

Running DeepArk on multiple GPUs in parallel is straightforward.
To toggle whether DeepArk should leverage multiple GPUs, simply specify the `--data-parallel` or `--no-data-parallel` flags.
This will toggle batch-level data parallelism on and off respectively.
We modify the `issm` example from above to use data parallelism as follows:

```
python DeepArk.py issm \
    --checkpoint-file 'data/drosophila_melanogaster.pth.tar' \
    --input-file 'examples/drosophila_melanogaster_issm_example.fasta' \
    --output-dir './' \
    --output-format 'tsv' \
    --batch-size '64' \
    --data-parallel
```

If you do not have more than one GPU available, then toggling data parallelism is unlikely to improve DeepArk's runtime performance.

#### <a name="threading"></a>5. How do I set the number of threads used when I run DeepArk without a GPU?

If you are using DeepArk without a GPU, you may want to alter the number of threads being used by PyTorch.
To do so, simply set the `--n-threads` argument to `DeepArk.py`.
This sets the number of PyTorch threads in a call to `torch.set_num_threads`.
As a demonstration of the proper usage, we modify the `predict` example from above to use 16 threads would work as follows:

```
python DeepArk.py predict \
    --checkpoint-file 'data/caenorhabditis_elegans.pth.tar' \
    --input-file 'examples/caenorhabditis_elegans_prediction_example.fasta' \
    --output-dir './' \
    --output-format 'tsv' \
    --batch-size '64' \
    --n-threads '16'
```

#### <a name="issm_speedup"></a>6. How can I speed up _in silico_ saturated mutagenesis?

_In silico_ saturated mutagenesis (ISSM) is generally the slowest process for DeepArk, in part because it is making far more predictions (i.e. roughly 17400 predictions per input sequence) than the other methods.
Consequently, ISSM will generally take longer to write its output to file than other methods.
A simple way to speed up ISSM runtime is to write predictions to [HDF5](http://portal.hdfgroup.org/display/knowledge/What+is+HDF5) files instead of TSV files.
We also recommend using a GPU when running ISSM.
If ISSM appears to be running slowly when using the GPU, make sure to force DeepArk to crash if it cannot access said GPU by [explicitly specifying CUDA use](#toggle_cuda).
If ISSM is too slow on a single GPU, you may want to consider using [multiple GPUs](#data_parallel).
If you do not have access to a GPU, you can use the GPU-accelerated [DeepArk webserver](#webserver) to run your ISSM experiments.

#### <a name="training"></a>7. How did you train DeepArk?

DeepArk was trained using [Selene](https://github.com/FunctionLab/selene), our PyTorch-based library for developing deep learning models of biological sequences.
All training details, such as model hyperparameters, will be described in a forthcoming manuscript.

#### <a name="performance"></a>8. How accurate is DeepArk?

DeepArk is quite accurate, and we are currently quantifying performance on a rigorous benchmark.
Details regarding performance will be thoroughly discussed in a forthcoming manuscript.

#### <a name="citation_faq"></a>9. How do I cite DeepArk?

If you use the DeepArk webserver or run DeepArk locally, we ask that you cite DeepArk.
Specific instructions on citing DeepArk can be found in [this section](#citation).

#### <a name="checkpoints"></a>10. Why are DeepArk's checkpoints different from Selene's?

To simplify DeepArk's use, we have include the constructor arguments for the model in the checkpoint files.
We also removed information that was not relevant to model inference (e.g. the minimum loss during training).
This allows us to distribute the model as two files: the `model.py` file and the weights file.
Clearly, this is different from the checkpoints generated by Selene.
To convert a checkpoint from Selene for use with DeepArk, use the `convert_checkpoint.py` script.
Documentation for this script can be accessed via `python scripts/convert_checkpoint.py --help`.

## <a name="citation"></a> Citing DeepArk

If you use DeepArk in your publication, please cite it.
We include a BibTex citation below.

```
@article{DeepArk,
  author = {Evan M Cofer, and Jo{\~{a}}o Raimundo, and Alicja Tadych, and Yuji Yamazaki, and Aaron K Wong, and Chandra L Theesfeld, and Michael S Levine, and Olga G Troyanskaya},
  title = {{DeepArk}: modeling \textit{cis}-regulatory codes of model species with deep learning},
  doi = {10.1101/2020.04.23.058040},
  url = {https://doi.org/10.1101/2020.04.23.058040},
  year = {2020},
  month = apr,
  journal = {biorXiv}
}
```

## <a name="related_tools"></a> Related tools

Please check out [Selene](https://github.com/FunctionLab/selene), our library for developing sequence-based deep learning models in [PyTorch](https://github.com/pytorch/pytorch).
Our paper on Selene is available in [Nature Methods](https://doi.org/10.1038/s41592-019-0360-8) or as a preprint [here](https://www.biorxiv.org/content/10.1101/438291v3).
