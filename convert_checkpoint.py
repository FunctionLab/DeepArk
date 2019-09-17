"""
Script to convert checkpoints from Selene to checkpoints for use in DeepArk scripts.

This file was created on September 13, 2019
Author: Evan Cofer
"""
import collections

import torch
import click

@click.command()
@click.option("--input-file", nargs=1, required=True, type=click.Path(exists=True), 
    help="Path that the current model checkpoint is located at.")
@click.option("--output-file", nargs=1, required=True, type=click.Path(exists=False), 
    help="Path to save the converted checkpoint at.")
@click.option("--kwargs", nargs=1, required=True, type=click.STRING, 
    help="Keyword arguments for the model constructor. These are not normally accessible to selene or"
    " other frameworks.")
@click.option("--n-peel", default=0, required=False, type=click.INT,
    help="Specify the number of wrapping modules to peel off of the model key strings. This "
    "is useful if the model was built with DataParallel or another module wrapping it, but you "
    "do not want to enforce user use of that module.")
@click.option("--arch-str", nargs=1, required=True, type=click.STRING, 
    help="String to store for architecture name. Useful if the architecture name is \"DataParallel\".")
def run(input_file, output_file, kwargs, n_peel, arch_str):
    """
    This script allows conversion from Selene-generated model checkpoints to a trimmed 
    down version that does not include data that is training-specific (e.g. optimizer and
    its parameters). The script adds model constructor arguments to the model checkpoint,
    so that only the model checkpoint file needs to be distributed (i.e. no configuration
    files). The script can also remove parts of the model weight strings, which comes in 
    handy if you want to distribute the checkpoint file of a model trained with
    DataParallel, and want to simplify the model loading process.
    """
    checkpoint = torch.load(input_file, map_location=torch.device("cpu")) # Load checkpoint, and enforce cpu usage.
    for k in list(checkpoint.keys()):
        if k not in ["arch", "state_dict"]:
            checkpoint.pop(k)
    if n_peel > 0:
        new_state_dict = collections.OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            k = k.split(".", n_peel)[-1]
            new_state_dict[k] = v
        checkpoint["state_dict"] = new_state_dict
    checkpoint["kwargs"] = eval(kwargs)
    checkpoint["arch"] = arch_str
    torch.save(checkpoint, output_file)


if __name__ == "__main__":
    run()
    