import re
import os
import pandas as pd

def get_first_header(note):
    if '## ' in note:
        first_header_prefix = '## '
        if '### ' in note:
            first_header_prefix = '### '

        first_header = note.split(first_header_prefix)[1].split('\n')[0]
    else:
        first_header = ''

    return first_header


def clean(note):
    # remove zero-links
    note = re.sub(r'\[.*\]', '', note)

    # remove tags and headers
    note = re.sub(r'\#.*\n', '', note)

    # remove \n
    note = re.sub('\n', ' ', note)

    # remove lines
    note = re.sub('---', ' ', note)

    # remove **
    note = re.sub('\*', '', note)
    
    return note

def num_letters(note):
    return len(re.sub(r'[^а-яА-Яa-zA-Z]', '', note))


def add_tags(row):
    note, note_cluster, header_cluster = row['raw_note'], row['note_cluster'], row['header_cluster']
    if '---' not in note[-8:]:
        note += '\n\n---'
    cleaned_note = re.sub(r'\n\n\[\[NN.*\]\]\n\n---', '', note)

    tag_pattern = f'\n\n[[NN #{note_cluster}]] [[NN ##{header_cluster}]]\n\n---'
    filled_note = cleaned_note + tag_pattern    
    return filled_note