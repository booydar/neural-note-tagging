import numpy as np
import pandas as pd
import torch
import os
from transformers import AutoModel, AutoTokenizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

from utils import num_letters, get_first_header, clean, add_tags

ENCODE_KWARGS = {'truncation': True, 'padding': 'max_length', 'pad_to_multiple_of': 1, 'max_length':512}

class NeuralTagger():
    # def __init__(self, folder_path, model_name, min_n_clusters, max_n_clusters, affinity, device, note_min_length):
    def __init__(self, folder_path, model_name, **kwargs):
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.kwargs = kwargs

        print('Parsing database...')
        self.db_df = self.parse_folder(folder_path, kwargs['note_min_length'])
        
        num_notes = self.db_df.shape[0]
        if kwargs['min_n_clusters'] < 1:
            self.min_n_clusters = int(kwargs['min_n_clusters'] * num_notes)
            self.max_n_clusters = int(kwargs['max_n_clusters'] * num_notes)


    def run(self):
        
        bs = self.kwargs['batch_size']
        self.model.to(self.kwargs['device'])

        notes, headers = list(self.db_df.note.values), list(self.db_df.header.values)
        print('Vectorizing notes...')
        note_embeddings, header_embeddings = self.vectorize(notes, headers, bs)
        print('Clustering notes...')
        header_clusters, note_clusters = self.cluster(header_embeddings), self.cluster(note_embeddings)

        self.db_df['note_cluster'] = note_clusters
        self.db_df['header_cluster'] = header_clusters
        self.db_df['filled_note'] = self.db_df.apply(add_tags, axis=1)

        print('Filling notes...')
        write_notes(self.db_df.path, self.db_df.filled_note)

        print('Done!')

    def vectorize(self, notes, headers, bs):

        tokenized_header = self.tokenizer.batch_encode_plus(headers, **ENCODE_KWARGS)['input_ids']
        tokenized_notes = self.tokenizer.batch_encode_plus(notes, **ENCODE_KWARGS)['input_ids']

        tokenized_headers = torch.Tensor(np.vstack(tokenized_header)).long()
        tokenized_notes = torch.Tensor(np.vstack(tokenized_notes)).long()

        vectorized_headers = process_by_batch(self.model, tokenized_headers, bs)
        vectorized_notes = process_by_batch(self.model, tokenized_notes, bs)

        ### take average of sequence outputs as ambedding
        header_embeddings = vectorized_headers.mean(dim=-2)
        note_embeddings = vectorized_notes.mean(dim=-2)

        return note_embeddings, header_embeddings

    
    def cluster(self, embeddings):
        metric = self.kwargs['affinity']
        min_n_clust, max_n_clust = self.min_n_clusters, self.max_n_clusters
        silhouettes = []
        for n_clust in range(min_n_clust, max_n_clust):
            clust_model = KMeans(n_clusters=n_clust)#, affinity=metric)
            clusters = clust_model.fit_predict(embeddings)

            silhouettes.append(silhouette_score(embeddings, clusters))
        
        best_n_clust = min_n_clust + np.argmax(silhouettes)
        clust_model = KMeans(n_clusters=best_n_clust)#, affinity=metric)
        clusters = clust_model.fit_predict(embeddings)
        return clusters


    def parse_folder(self, path, length_thr=100):

        path, folders, files = next(os.walk(path))

        db_df = pd.DataFrame()
        if len(folders) > 0:
            for f in folders:
                folder_path = os.path.join(path, f)
                f_res_df = self.parse_folder(folder_path)
                db_df = pd.concat([db_df, f_res_df])

        for fn in files:
            if '.md' not in fn:
                continue

            filepath = os.path.join(path, fn)
            with open(filepath, 'r') as f:
                note = f.read()

            if num_letters(note) < length_thr:
                continue

            header = get_first_header(note)
            if not header: 
                header = fn[:-3]

            cleaned_note = clean(note)

            note_dict = {'name': fn, 'path':filepath, 'header': header, 'note': cleaned_note, 'raw_note':[note]}

            db_df = pd.concat([db_df, pd.DataFrame(note_dict)])
        
        return db_df


def process_by_batch(model, tensor, batch_size):
    n_chunks = tensor.shape[0] // batch_size
    chunked = torch.chunk(tensor, n_chunks)

    outputs = []
    for batch in chunked:
        out = model(batch.to(device=model.device)).last_hidden_state
        outputs.append(out.cpu().detach())

    return torch.cat(outputs)


def write_notes(paths, filled_notes):
    for path, note in zip(paths, filled_notes):
        with open(path, 'w') as f:
            f.write(note)