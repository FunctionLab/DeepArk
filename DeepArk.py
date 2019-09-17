#!/usr/bin/env python3
"""
Command line interface for using DeepArk models.

This file was created on September 13, 2019
Author: Evan Cofer
"""
import enum
import os

import click
import selene_sdk
import selene_sdk.predict
import torch
import torch.nn

from model import DeepArkModel


def load_model_from_checkpoint_file(path):
    """Reads the model parameters from the specified checkpoint file, and
    uses them to generate a model object. The weights of the model object are
    then loaded from the weights file."""
    checkpoint = torch.load(path, map_location=torch.device("cpu"))
    model = DeepArkModel(**checkpoint["kwargs"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


class FileType(enum.Enum):
    FASTA = 1
    VCF = 2
    BED = 3


def get_file_type(path):
    """Determines the type of file at the specified path."""
    ext = os.path.splitext(path.lower())[1]
    if ext in {".fa", ".fasta", "fna"}:
        return FileType.FASTA
    elif ext == ".vcf":
        return FileType.VCF
    elif ext == ".bed":
        return FileType.BED
    else:
        msg = "Could not determine file type for file at {}".format(path)
        raise ValueError(msg)


@click.group()
@click.pass_context
def cli(context):
    """
    Command line interface for DeepArk, a set of deep neural networks for predicting 
 regulatory activity (e.g. transcription factor binding) from genomic sequences for 
 worm, fly, and mouse. For more details, see the relevant publication.

    ***If you use this method in a paper, please cite it.***
    """
    # Seed RNG.
    seed = 1337
    torch.manual_seed(seed)
    pass


@cli.command()
@click.option("--checkpoint-file", nargs=1, required=True, type=click.Path(exists=True),
    help=("The file containing the weights of the DeepArk model that you want to make " 
          "predictions with."))
@click.option("--input-file", nargs=1, required=True, type=click.Path(exists=True),
    help=("A FASTA file containing sequences to make predictions for, or a BED file containing "
          "coordinates for said sequences. If using a BED file, you must also specify a file "
          "for the --genome-file argument."))
@click.option("--genome-file", nargs=1, required=False, default=None, type=click.Path(exists=True),
    help=("A FASTA file containing the reference genome sequence to make predictions on. This "
          "argument is only required if are using a BED file for --input-file option."))
@click.option("--output-dir", nargs=1, required=True, type=click.Path(exists=False),
    help=("Directory to write the output to."))
@click.option("--output-format", nargs=1, required=True, type=click.Choice(["hdf5", "tsv"]),
    help="Format to save output predictions in.")
@click.option("--batch-size", nargs=1, default=64, required=False, type=click.INT,
    help="Size of the mini-batch to use during prediction.")
@click.option("--cuda/--no-cuda", nargs=1, default=None, required=False,
    help=("Toggle CUDA use by PyTorch. If not specified, will use `torch.cuda.is_available()`"
         " to decide CUDA use."))
@click.option("--data-parallel/--no-data-parallel", nargs=1, default=False, required=False,
    help=("Toggle data-level parallelism use by PyTorch. Default is `False`."))
@click.option("--n-threads", nargs=1, default=1, required=False, type=click.INT, 
    help="Number of threads for PyTorch.")
@click.pass_context
def predict(context, checkpoint_file, input_file, genome_file, output_dir, output_format, batch_size, cuda, data_parallel, n_threads):
    """
    Make predictions for genomic sequences.

    The input file should be a FASTA file containing the sequences to perform in silico mutagenesis on.
 Alternatively, you can use a BED file and FASTA reference file. In both cases, the sequences must be at
  least 5797 bases long. If sequences are longer, the middle 5797 bases will be used for prediction.
    """
    # Setup.
    torch.set_num_threads(n_threads)
    if cuda is None:
        cuda = torch.cuda.is_available()

    # Load model.
    model = load_model_from_checkpoint_file(checkpoint_file)
    sequence_length = model.sequence_length
    n_features = model.n_features
    model.eval()
    if cuda:
        model.cuda()

    # Build selene AnalyzeSequences object.
    genome = selene_sdk.sequences.Genome if genome_file is None else selene_sdk.sequences.Genome(genome_file)
    pred_obj = selene_sdk.predict.AnalyzeSequences(model,
                                                   checkpoint_file,
                                                   model.sequence_length,
                                                   [str(i) for i in range(model.n_features)],
                                                   batch_size,
                                                   use_cuda=cuda,
                                                   data_parallel=data_parallel,
                                                   reference_sequence=genome)

    # Check input file formats and make predictions.
    ft = get_file_type(input_file)
    if ft == FileType.FASTA:
        pred_obj.get_predictions_for_fasta_file(input_file, output_dir, output_format)
    elif ft == FileType.BED:
        pred_obj.get_predictions_for_bed_file(input_file, output_dir, output_format)
    else:
        msg = "{} does not appear to be a FASTA file or a BED file.".format(input_file)
        raise ValueError(msg)
   

@cli.command()
@click.option("--checkpoint-file", nargs=1, required=True, type=click.Path(exists=True),
    help=("The file containing the weights of the DeepArk model that you want to make " 
          "predictions with."))
@click.option("--input-file", nargs=1, required=True, type=click.Path(exists=True),
    help=("A VCF file containing the variants to predict the effects of."))
@click.option("--genome-file", nargs=1, required=True, type=click.Path(exists=True),
    help=("A FASTA file containing the reference genome sequence that variants were"
          " called with."))
@click.option("--output-dir", nargs=1, required=True, type=click.Path(exists=False),
    help=("Directory to write the output to."))
@click.option("--output-format", nargs=1, required=True, type=click.Choice(["hdf5", "tsv"]),
    help="Format to save output predictions in.")
@click.option("--batch-size", nargs=1, default=64, required=False, type=click.INT,
    help="Size of the mini-batch to use during prediction.")
@click.option("--cuda/--no-cuda", nargs=1, default=None, required=False,
    help=("Toggle CUDA use by PyTorch. If not specified, will use `torch.cuda.is_available()`"
         " to decide CUDA use."))
@click.option("--data-parallel/--no-data-parallel", nargs=1, default=False, required=False,
    help=("Toggle data-level parallelism use by PyTorch. Default is `False`."))
@click.option("--n-threads", nargs=1, default=1, required=False, type=click.INT, 
    help="Number of threads for PyTorch.")
@click.pass_context
def vep(context, checkpoint_file, input_file, genome_file, output_dir, output_format, batch_size, cuda, data_parallel, n_threads):
    """
    Predict the regulatory effects of variants.

    Input should follow the VCF file format, but should not include more than one variant per 
line (e.g. triallelic sites) or variants that cannot be represented with the four canonical base
pairs (e.g. some very long or named variants). Although variants could be as long as 5797 bases, 
we recommend only considering shorter variants (e.g. <1000 bases). To consider more than one 
variant at a time (i.e. haplotypes), use `predict` to make predictions on a FASTA file with 
the alternative haplotype sequences.
    """
    # Setup.
    torch.set_num_threads(n_threads)
    if cuda is None:
        cuda = torch.cuda.is_available()

    # Load model.
    model = load_model_from_checkpoint_file(checkpoint_file)
    sequence_length = model.sequence_length
    n_features = model.n_features
    model.eval()
    if cuda:
        model.cuda()

    # Check input file type.
    ft = get_file_type(input_file)
    if ft != FileType.VCF:
        msg = "{} does not appear to be a VCF file.".format(input_file)
        raise ValueError(msg)

    # Build selene AnalyzeSequences object.
    genome = selene_sdk.sequences.Genome(genome_file)
    pred_obj = selene_sdk.predict.AnalyzeSequences(model,
                                                   checkpoint_file,
                                                   model.sequence_length,
                                                   [str(i) for i in range(model.n_features)],
                                                   batch_size,
                                                   use_cuda=cuda,
                                                   data_parallel=data_parallel,
                                                   reference_sequence=genome)

    # Run predictions on variants.
    pred_obj.variant_effect_prediction(input_file, 
                                       save_data=["abs_diffs", "diffs", "logits", "predictions"],
                                       output_dir=output_dir, 
                                       output_format=output_format)


@cli.command()
@click.option("--checkpoint-file", nargs=1, required=True, type=click.Path(exists=True),
    help=("The file containing the weights of the DeepArk model that you want to make " 
          "predictions with."))
@click.option("--input-file", nargs=1, required=True, type=click.Path(exists=True),
    help=("A FASTA file containing sequences to perform in silico mutagenesis on."))
@click.option("--output-dir", nargs=1, required=True, type=click.Path(exists=False),
    help=("Directory to write the output to."))
@click.option("--output-format", nargs=1, required=True, type=click.Choice(["hdf5", "tsv"]),
    help="Format to save output predictions in.")
@click.option("--batch-size", nargs=1, default=64, required=False, type=click.INT,
    help="Size of the mini-batch to use during prediction.")
@click.option("--cuda/--no-cuda", nargs=1, default=None, required=False,
    help=("Toggle CUDA use by PyTorch. If not specified, will use `torch.cuda.is_available()`"
         " to decide CUDA use."))
@click.option("--data-parallel/--no-data-parallel", nargs=1, default=False, required=False,
    help=("Toggle data-level parallelism use by PyTorch. Default is `False`."))
@click.option("--n-threads", nargs=1, default=1, required=False, type=click.INT, 
    help="Number of threads for PyTorch.")
@click.pass_context
def ism(context, checkpoint_file, input_file, output_dir, output_format, batch_size, cuda, data_parallel, n_threads):
    """
    Perform saturated in silico mutagenesis.

       The input file should be a FASTA file containing the sequences to perform in silico mutagenesis on.
The sequences should be at least 5797 bases long. If sequences are longer, the middle 5797 base will be 
used for prediction.
    """
    # Setup.
    torch.set_num_threads(n_threads)
    if cuda is None:
        cuda = torch.cuda.is_available()

    # Load model.
    model = load_model_from_checkpoint_file(checkpoint_file)
    sequence_length = model.sequence_length
    n_features = model.n_features
    model.eval()
    if cuda:
        model.cuda()

    # Build selene AnalyzeSequences object.
    pred_obj = selene_sdk.predict.AnalyzeSequences(model,
                                               checkpoint_file,
                                               model.sequence_length,
                                               [str(i) for i in range(model.n_features)],
                                               batch_size,
                                               use_cuda=cuda,
                                               data_parallel=data_parallel,
                                               reference_sequence=selene_sdk.sequences.Genome)

    # Check input file type.
    ft = get_file_type(input_file)
    if ft == FileType.FASTA:
        pred_obj.in_silico_mutagenesis_from_file(input_file,
                                                 save_data=["abs_diffs", "diffs", "logits", "predictions"],
                                                 output_dir=output_dir,
                                                 mutate_n_bases=1,
                                                 output_format=output_format)
    else:
        msg = "{} does not appear to be a FASTA file.".format(input_file)
        raise ValueError(msg)


if __name__ == "__main__":
    cli(obj=dict())
