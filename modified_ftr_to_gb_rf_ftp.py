#!/usr/bin/env python3

import pandas as pd
import feather

"""
SECOND STEP.
Modification of "ftr_to_genome_acc_URL.py".

This script takes a list of .ftr filepaths, opens each file, parses it for the RefSeq column, compares it against a reference key, and creates .txt and .sh files with GCF or GCA URLs.

This is designed as part of a work around of the SRA Toolkit to enable mass download via the UNCC HPC DTN.

by David Brown 20220331

Then run the bash script output of this file.
Then run make_bash_call_prokka.py.
"""

"""
TODO
- Need to figure out how to get the full file name, as I have only partials (the accession information is shortened)
-- See this https://ftp.ncbi.nlm.nih.gov/genomes/genbank/bacteria/Escherichia_coli/all_assembly_versions/README.txt
-- Parse these: ("asm_acc" in tsv/ftr should be "asm_name" in the .txt)
--- https://ftp.ncbi.nlm.nih.gov/genomes/genbank/bacteria/Escherichia_coli/assembly_summary.txt (GCA)
--- https://ftp.ncbi.nlm.nih.gov/genomes/refseq/bacteria/Escherichia_coli/assembly_summary.txt  (GCF)
--- Capture column 20
-- Use column 20 to correctly adjust the ftp URLs
- Add flag for silent/reduced output
- Add flag for redirecting output? (or just rely on nohup?)
"""



# The web address prefix to ftp site for RefSeq (GCF) or GenBank (GCA) files
URL_prefix = "://ftp.ncbi.nlm.nih.gov/genomes/all/"

# Choose the type of file to retrieve here by uncommenting one of the below
this_suffix = "_genomic.fna.gz"     # Genome FASTA
#this_suffix = "_genomic.gff.gz"   # Annotations as GFF3
#this_suffix = "" #
#this_suffix = "" #

# Lookup key files for accessing the FTP site
#file_ass_sum_gb = "/Users/dbrow208/Documents/galick_gun/ncbi_ftp_assembly_summary_20220331/assembly_summary_genbank.txt"
#file_ass_sum_rf = "/Users/dbrow208/Documents/galick_gun/ncbi_ftp_assembly_summary_20220331/assembly_summary_refseq.txt"
file_ass_sum_gb = "/Users/dbrow208/Documents/galick_gun/test_prokka_roary/hpc_test_20220621/assembly_summary_genbank.txt"
file_ass_sum_rf = "/Users/dbrow208/Documents/galick_gun/test_prokka_roary/hpc_test_20220621/assembly_summary_refseq.txt"

# Open as dataframes; tab-separated; assign header; skip 1 row; use more memory in case of mixed types.
df_ass_sum_gb = pd.read_csv(file_ass_sum_gb, sep='\t', header=1, skiprows=0, low_memory=False)
df_ass_sum_rf = pd.read_csv(file_ass_sum_rf, sep='\t', header=1, skiprows=0, low_memory=False)
#print(df_ass_sum_gb.columns)
# List of input filepaths for .ftr files to be parsed for SRA information
"""
fp_inputs = [   "/Users/dbrow208/Documents/galick_gun/test_prokka_roary/20220601_test_run/metadata_subset_host_Gallus_geo_loc_name_Denmark_72.ftr",
                "/Users/dbrow208/Documents/galick_gun/test_prokka_roary/20220601_test_run/metadata_subset_host_Gallus_geo_loc_name_USA_138.ftr"
]
"""
#fp_inputs = ["/Users/dbrow208/Documents/galick_gun/test_prokka_roary/hpc_test_20220621/metadata_subset_asm_level_Scaffold_8678.ftr"]
fp_inputs = ["/Users/dbrow208/Documents/galick_gun/test_prokka_roary/hpc_test_20220829/metadata_subset_asm_level_Scaffold_geo_loc_USA_CA_standardized_101.ftr"]
# Generate the URLs for 
for i in fp_inputs:
    # Isolate subset name for outfile
    this_file = i.split("/")[-1]
    this_file = this_file.split(".")[0]    # Capture the subset name

    # Outfile name(s)
    outfile_ftps_gb = "/Users/dbrow208/Documents/galick_gun/test_prokka_roary/hpc_test_20220829/gb_ftps_for_" + this_file + ".txt"
    outfile_bash_gb = "/Users/dbrow208/Documents/galick_gun/test_prokka_roary/hpc_test_20220829/gb_bash_for_" + this_file + ".sh"
    outfile_ftps_rf = "/Users/dbrow208/Documents/galick_gun/test_prokka_roary/hpc_test_20220829/rf_ftps_for_" + this_file + ".txt"
    outfile_bash_rf = "/Users/dbrow208/Documents/galick_gun/test_prokka_roary/hpc_test_20220829/rf_bash_for_" + this_file + ".sh"

    # Read in dataframe
    this_df = pd.read_feather(i)

    # Transform the pandas Series for the "Run" column to a list
    acc_list = this_df["asm_acc"].dropna().to_list()

    # This is a dumb solution, but it works. Some entries within the "Run" column are multivalue. The below fixes that issue, but it could lead to extra work when associating metadata in the future. -db 20220310
    acc_list = ",".join(acc_list)
    acc_list = acc_list.split(",")

    """
    # This formats the RefSeqs into the proper format for online accession
    formatted_genome_list = []
    for acc in acc_list:
        temp_acc = acc.split(".")[0]
        temp_acc = temp_acc.replace("_", "")
        split_acc = [ temp_acc[i:i+3] for i in range(0, len(temp_acc), 3) ]
        split_acc.append(acc)
        formatted_genome_list.append(split_acc)
    """

    # Instantiate empty lists to hold the GenBank (GCA) FTP addresses and "paired" accession from RefSeq.
    ftps_gb = []
    paired_rf_acc = []
    # Loop over the accessions, capturing the GenBank (GCA) FTP address and the RefSeq accessions (GCF)
    for acc in acc_list:
        try:
            acc_ftp_hit = df_ass_sum_gb[ df_ass_sum_gb['# assembly_accession']==acc ]['ftp_path'].values[0]
            gca_hit = acc_ftp_hit.split('/')[-1]
            acc_ftp_hit = acc_ftp_hit + "/" + gca_hit + this_suffix
            ftps_gb.append(acc_ftp_hit)
            acc_rf_hit = df_ass_sum_gb[ df_ass_sum_gb['# assembly_accession']==acc ]['gbrs_paired_asm'].values[0]
            paired_rf_acc.append(acc_rf_hit)
        except:
            continue
    # Format for the GenBank (GCA) outputs
    output_ftps_gb = [ j for j in ftps_gb if j != None ]
    output_bash_gb = [ 'wget ' + k for k in ftps_gb if k != None ]

    # Instantiate empty list to hold the RefSeq (GCF) FTP addresses
    ftps_rf = []
    for pair in paired_rf_acc:
        try:
            pair_ftp_hit = df_ass_sum_rf[ df_ass_sum_rf['# assembly_accession']==pair ]['ftp_path'].values[0]
            gcf_hit = pair_ftp_hit.split('/')[-1]
            pair_ftp_hit = pair_ftp_hit + "/" + gcf_hit + this_suffix
            ftps_rf.append(pair_ftp_hit)
        except:
            continue
    # Format for the RefSeq (GCF) outputs
    output_ftps_rf = [ q for q in ftps_rf if q != None ]
    output_bash_rf = [ 'wget ' + x for x in ftps_rf if x != None ]

    # Write outfile with only URLs
    with open( outfile_ftps_gb, 'w' ) as f1:
        f1.writelines("%s\n" % l for l in output_ftps_gb)

    # Write outfile as a bash script utilizing curl.
    with open( outfile_bash_gb, 'w' ) as f2:
        f2.write("#!/bin/bash\n\n")
        f2.writelines("%s\n" % l for l in output_bash_gb)
    
    # Write outfile with only URLs
    with open( outfile_ftps_rf, 'w' ) as f3:
        f3.writelines("%s\n" % l for l in output_ftps_rf)

    # Write outfile as a bash script utilizing curl.
    with open( outfile_bash_rf, 'w' ) as f4:
        f4.write("#!/bin/bash\n\n")
        f4.writelines("%s\n" % l for l in output_bash_rf)
    