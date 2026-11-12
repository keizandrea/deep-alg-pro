#!/usr/bin/env python
# coding: utf-8

from Bio import SeqIO
import os,sys

# Inspecting a few records to understand raw data
clean_sequences = []
short_sequences = []
unwanted_char_sequences = []

def clean_fasta(input_file, output_file, min_length, unwanted_char, batch_size):


    records = list(SeqIO.parse(input_file, 'fasta'))
    
    for record in records:
        # checking length and character exclusion
        if len(record.seq) >= min_length and unwanted_char not in str(record.seq).upper():
            clean_sequences.append(record)
        elif len(record.seq) < min_length: 
            short_sequences.append(str(record.seq))
        elif unwanted_char in str(record.seq).upper():
            unwanted_char_sequences.append(record.seq)

    remainder = len(clean_sequences) % batch_size
    if remainder != 0:
        clean_sequences = clean_sequences[:-remainder]
        print(f'You need to drop {remainder} sequences(s) to make it divisible by {batch_size}.')

    SeqIO.write(clean_sequences, output_file, 'fasta')

if __name__ == '__main__':
    if len(sys.argv) < 4:
        # <> means required, [] means optional
        print('Usage: python clean_fasta.py <input_path> <min_length> <unwanted_char> [batch_size]')
        sys.exit(1)

    in_file = sys.argv[1]
    min
    dir_name = os.path.dirname(input_file)
    base_name = os.path.basename(input_file)
    output_file = os.path.join(dir_name, 'cleaned_'+ base_name)
    # print(dir_name, base_name)
    print(f'Your output file: {output_file}')

    valid_count = len(clean_sequences) - (len(clean_sequences) % batch_size)
    print(valid_count)
    final_sequences = clean_sequences[:valid_count]

    SeqIO.write(final_sequences, output_file, 'fasta')
    print(f'Saved {len(final_sequences)} valid sequences to {output_file} :) Time to work!')
