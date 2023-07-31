#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

"""
by David Brown (db) - 20221216
Adapted from "/Users/dbrow208/Documents/galick_gun/snp_sites/run_snpsites.py"

Reference https://github.com/sanger-pathogens/snp-sites

'This application takes in a multi fasta alignment, finds all the SNP sites, then outputs the SNP sites in the following formats:

    -m  =   a multi fasta alignment,
    -v  =   VCF,
    -p  =   relaxed phylip format.
    -r  =   internal reference sequence
'

Input is a filepath to a multiFASTA alignment (DNA or protein)

Run within the 'binf_snp-sites' conda environment

TODO:
"""

###
# VARIABLES - Make changes here
# # A filepath string to an input multiFASTA (can be gzipped)
this_input              =   "/Users/dbrow208/Documents/galick_gun/snp_sites/subset_900_ns_protein_mutS.faa"
# Designate output filepath string (do NOT include suffixes) for the snp-sites program calls
this_snp_sites_output   =   "/Users/dbrow208/Documents/galick_gun/snp_sites/20221216_snp_sites_consensus_test"

###
# FUNCTIONS

# A Python function that calls the 'snp-sites' program an input multiFASTA alignment file (can be gzipped) and outputs files to a directory (indicated by filepath)
def callSNPsites( input_multiFASTA, output_filepath):
    """
    Inputs:
    -   aligned multiFASTA file (can be gzipped)
    -   desired path for the 3 output files (string)

    Files created:
    -   MultiFASTA, only containing SNPs for machine learning and parsimony trees. (a 'SNP matrix')
    -   Relaxed Phylip, for RAxML trees.
    -   VCF, for positional references of the SNPs.
    -   FASTA, an internal reference sequence

    Returns:
    -   three strings, containing the output filepaths
    """
    # Builds the output filepath strings
    name_multiFASTA = output_filepath + ".fasta"
    name_phylip     = output_filepath + ".phy"
    name_vcf        = output_filepath + ".vcf"
    # Builds the snp-sites call strings from the inputs
    multiFASTA_call = "snp-sites -m -o " + name_multiFASTA + " " + str(input_multiFASTA)
    phylip_call     = "snp-sites -p -o " + name_phylip + " " + str(input_multiFASTA)
    vcf_call        = "snp-sites -v -o " + name_vcf + " " + str(input_multiFASTA)
    # Call 'snp-sites'
    os.system(multiFASTA_call)
    os.system(phylip_call)
    os.system(vcf_call)
    # Returns the string filepaths of the three created files.
    return(name_multiFASTA, name_phylip, name_vcf)

# A Python function that represents the full consensus sequence from a multiFASTA alignment, using a VCF file as reference.
def makeConsensusFASTA( some_multiFASTA, some_vcf, some_output_filepath ):
    """
    Inputs:
    -   aligned multiFASTA file (cannot be gzipped)
    -   the VCF file associated with that multiFASTA file (usually from snp-sites)
    -   desired path for the output FASTA file (string)

    Files created:
    -   FASTA file containing the consensus sequence
    """
    # Read in the multiFASTA and capture the first record
    these_seqs = []
    for seq_record in SeqIO.parse(some_multiFASTA, "fasta"):
        these_seqs.append(seq_record)
    this_record = these_seqs[0]     # Select a sequence from the multiFASTA to model the consensus on.
    # Convert Seq object to a list
    this_seq = str(this_record.seq)
    this_seq = [ this_seq[i] for i in range(0, len(this_seq))]

    # Extract the SNP positions and reference bases from the VCF file.
    vcf_df = pd.read_csv(some_vcf, skiprows=3, sep="\t")    # Skips the three heading rows.
    these_pos = vcf_df['POS'].tolist()                      # Capture the SNP positions column.
    these_pos = [ int(i) - 1 for i in these_pos ]           # Force the SNP positions column to integer and subtract 1. db 20221216 NOTE: VCF files are 1-based, Python is 0-based. Reference https://samtools.github.io/hts-specs/VCFv4.1.pdf 
    these_base = vcf_df['REF'].tolist()                     # Capture the REF base column.

    # Create dictionary, build the consensus sequence, and cast to a "Seq" object
    dict_pos_base = dict(zip( these_pos, these_base ))
    print(dict_pos_base)
    #print(len(first_seq))
    #print(first_seq)
    for key in dict_pos_base.keys():
        this_seq[key] = dict_pos_base[key]
    out_seq = "".join(this_seq)
    out_seq = Seq(out_seq)

    # Populate a "SeqRecord" object for output. A sequence, an ID, and a description are required.
    out_seq_record = SeqRecord(out_seq)
    this_id = some_multiFASTA.split("/")[-1]
    this_id = this_id.split(".")[0]
    out_seq_record.id = "CONSENSUS_" + this_id
    out_seq_record.description = "Consensus sequence generated from the multiFASTA file: " + some_multiFASTA + ", and the VCF file: " + some_vcf

    # Write the output
    output_FASTA = some_output_filepath + ".consensus"
    SeqIO.write(out_seq_record, output_FASTA, "fasta")


###
# EXECUTION
this_multiFASTA, this_phylip, this_vcf  = callSNPsites( this_input, this_snp_sites_output )
makeConsensusFASTA( this_input, this_vcf, this_snp_sites_output )