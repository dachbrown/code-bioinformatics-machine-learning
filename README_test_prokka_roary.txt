By David Brown (db) 20220606

This directory contains tests for running the Prokka and Roary programs.

General steps in analysis:
1) Acquire contig or scaffold .fna files in a single directory.
	- Potentially useful files for this step:
		-- "explore_ftr_dataframe_and_subset.py" - reads, explores, and subsets metadata from NCBI.
		-- "modified_ftr_to_gb_rf_ftp.py"	- identifies FTP links to NCBI and prepares bash script for download.
		-- also consider "fasterq-dump" from NCBI "sra-toolkit" if SRA information is available.
2) Run "make_bash_call_prokka.py" in a normal bioinformatics environment
3) Switch to the "prokka_roary" environment.
4) Run the created bash script to call Prokka.
5) Run Roary on the outputs of Prokka.
6) Switch back to a normal bioinformatics environment.
7) Infer a phylogenetic tree using RAxML from the Roary-created "core_gene_alignment.aln" file.
8) Run "roary_plots/roary_plots.py" on the inferred tree and Roary files.
9) Prepare Roary-created "gene_presence_absence.csv" for ML input.
10) Consider processing the "results_Prokka/ffn_files/" directory to make a sequence database for SNP analysis.


TNT:
- Reread dissertation and check hypotheses
- Try to work with TNT
- Update this process for the UNCC HPC, keeping in mind the following Anaconda information and included "prokka_roary" environment files (and prokka --cpus 0)
-- Small attempt made 20220606. Will be difficult. Backburner for now.
"""
-------------------------------------------------------------------------------------------- /apps/usr/modules/compilers --------------------------------------------------------------------------------------------
alphafold/2.1.1    anaconda3/2020.11(default)  gcc/10.3.0(default)  go/1.16.4(default)     intel/2019           lua/5.3.5           openjdk/11           scala/2.12.13
anaconda2/2019.10  bazel/2.0.0                 gcc/11.2.0           hpc-sdk/21.3(default)  intel/2020(default)  lua/5.4.2(default)  openjdk/13(default)  scala/2.13.5(default)
anaconda3/2019.07  bazel/3.1.0(default)        go/1.14.2            hpc-sdk/21.3-mpi       julia/1.6.0          nasm/2.14           openjdk/15           yasm/1.3.0
"""
- Clean up this folder, deprecate and or remove, especially for data
- Work on the machine learning
	- especially scores, visuallizing scores and results
	- also trying a leave one out approach


Current testing
- DONE 0603 vs 0605 are demonstrating slightly different Prokka calls (genus vs kingdom) on same data Gallus Denmark 72
-- DONE passed 0605 to Roary
-- DONE Then move to 0605 alignment to HPC for tree inference.
--- Have compared bestTree and bestTree_optimized for 0603 run. Slightly different results. Frequency and pie files unchanged. Matrix, slight changes. Appears to mimic demo results online.
---- Need to compare 0605 and 0603. Potentially a few ~29 more genes in kingdom run than genus (0605 vs 0603)
---- Looks to be roughly the same, tree shape appears similar, could be due to random seed
- DO I NEED TO CHANGE RANDOM SEED?
- 0606 is demonstrating a different dataset Gallus US 138 for improving ML practice data (prediction of region, not just MDR)
-- DONE When Roary reduces resources, need to pass 0606 to Prokka
-- DONE 0606 Prokka
- Began 0606 Roary
- once both individual Roary is done, try Roary on both combined (Denmark & US) to predict MDR types (sets of AMR)

for finish 1
New - all done?
- pop out muts/l/h genes from the ffn files?
- create alignment
- call SNPs
Old
- transform to pres/abs (like convert)
- do the same selectkbest (like cross_val)

- check more genes?
- check on HPC (quickly, then ask Jon) about the environment

- reread some stuff on my stuff

- arrange meetings
-- guo
-- white
-- sung
-- denis? or dornburg?
- make questions for all


- try prokka/roary on hpc, then contact Jon
- finish 2nd chicken RAxML
- combined chicken Roary, then RAxML
- Chp. 2, question 1
-- define functional & id snps in muts/l/h
-- map snp "characters" or functionality to tree in TNT, ybyra, or BEAST? check notes
-- map mdr characters or functionality to tree in TNT, ybyra, or BEAST? check notes
-- run on 2nd chicken
-- run on combined chicken
-- run bigger or sum up to committee
- Chp. 3, question 2
-- do cross validation
-- do leave one out?
-- run on 2nd chicken
-- run on combined chicken
-- run bigger or sum up to committee
- Chp. 4, question 3
-- ID clusters (from predictions?)
-- generate trees of clusters & trees within clusters
-- check hypotheses


dr white
tweaking roary
- start with apology
-- done
- general thoughts on optimization
-- done, he does not know if soft core can be explicitly changed
- report # genes for 3Mb E. coli (~2800 in core)
-- likely ok
- useful add on scripts, softwares he recommends
-- sent me a list
- roary ranking of core genome genes (order)
-- believes it is descending order
- i have not used kraken, is it important for verification? how important?
-- likely not. suggested checkm. will consider
- him
-- he suggested thinking about the legacy perl script issue
-- asked me to look into vcf files from roary

guo questions
- i don't mean "function" I mean activity level but for mine i still might be able to say functional change or activity level (definitely a spectrum)
- potential around ATP binding ~853
- potential around hydrolysis at 2082 & 694 (Glu to Ala), but I only see Glu to Met
- how to define functional?
-- as altering mutation rate (how to calculate that in a wild sample?)
-- as pseudogenes? do my weird types?
-- read alphafold stuff (images) and guess? check domains?
-- the literature suggests deletions, but not much "variable" or "differential" activity. I do not have deletions, either in current test set or larger set as a whole.

- what type of mutation
-- indel, beginning, end, synonymous, nonsynonymous

- map to point on structure (look for clusters, active sites, etc.), but look specifically for hydrophilic to hydrophobic in an active site to notice a DECREASE in activity
***
- mutation structural location & type (hydrophilic/hydrophobic)
***
- programs have been developed to check the effect of a point mutation on a protein (only for a single point mutation)


notes from way
Way seems to think that I should stay far away from any mention of rate. I will not be able to compare the raw # of mutations from one strain to another, as I lack any information about time. The best I could do is coding against noncoding, but as I have annotated, I lack noncoding data. This leads to the following thoughts

I likely can just look at the "consensus" from an alignment [MAKE SURE TO ALIGN, AS THE FFN FILES ARE NOT ALIGNED] and then count the differences

1 get a list of all genes involved in replication fidelity (base excision, other repair, not just mismatch)
2 count the snps (nonsynonymous) in those replication fidelity genes
3 assign levels of AMR to each substrain
4 either compare a single nonsynonymous replication fidelity snp to the AMR rank (a rank test)
OR
compare the proportion of AMR strains against each "spectra" of replication fidelity snps
OR
compare AMR strains against each single replication fidelity SNP
OR
compare AMR strains and the spectrum of mutations at synonymous sites

chi2 is for goodness of fit
not as concerned about recent findings regarding a negative fitness for synonymous SNPs

note to self, will need to realign all of the specific sequences, as there appears to be some frameshift stuff going on, at least in my 72 danish chickens set. had to move things right about 6bp.


also, will need to align again, when popping out genes


hpc testing
locally, decide on subset via explore_ftr_dataframe_and_subset.py
then, on hpc, use dtn_template.slurm to download files via the ftps urls 
then, on hpc, use anaconda_template.slurm to make the bash calling script, run prokka, and run roary

to do, be able to run raxml on the output, or within an all in one
consider upgrading prokka from 1.12 to 1.14
	- cannot due to conflicts with libgcc (for gnu, possibly linux specific) and zlib as a result of libgcc
identify why 6267 only returned ~5700 sequences

downloaded from uniprot FASTA (compressed) with canonical and isoform muthls from e coli & shigella (79 seqs) using:
(gene:muts OR gene:muth OR gene:mutl) name:"mismatch repair" (organism:"escherichia coli" OR organism:shigella) AND reviewed:yes

downloaded from uniprot FASTA (compressed) with canonical and isoform muthls from e coli & shigella (468 seqs) using:
work on the search term below
(gene:bla OR name:ampicillin OR name:resistant OR name:aminoglycoside OR name:beta-lactam OR name:cephem OR name:macrolides OR name:penem OR name:penicillin OR name:phenicol OR name:quinolone OR name:tetracycline OR name:resistance OR name:multidrug OR name:drug OR name:antibiotic OR name:antimicrobial) (organism:"escherichia coli" OR organism:shigella) AND reviewed:yes
https://www.uniprot.org/uniprot/?query=%28gene%3Abla+OR+name%3Aampicillin+OR+name%3Aresistant+OR+name%3Aaminoglycoside+OR+name%3Abeta-lactam+OR+name%3Acephem+OR+name%3Amacrolides+OR+name%3Apenem+OR+name%3Apenicillin+OR+name%3Aphenicol+OR+name%3Aquinolone+OR+name%3Atetracycline+OR+name%3Aresistance+OR+name%3Amultidrug+OR+name%3Adrug+OR+name%3Aantibiotic+OR+name%3Aantimicrobial%29+%28organism%3A%22escherichia+coli%22+OR+organism%3Ashigella%29+AND+reviewed%3Ayes&sort=score

ran a big test, using a protein file for muthls on ~5700 samples from the 6286 USA dataset on the HPC. Allowed 14 days for Prokka/Roary and 14 days for RAxML
MUST USE A FASTA FILE DATABASE DUE TO MAKEBLASTDB VERSION 2.2 NOT AUTODETECTING THINGS.
-- I need to convert from a simple fasta to a detailed fasta as noted by Prokka.