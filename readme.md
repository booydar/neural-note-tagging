## Automatic note tagging using neural networks

### Problem
Have you noticed that once you've written down your brilliant idea in a note app, it tends to just flow away, moved down by new unnecessary notes? 

If you have, you probably know about [Zettelkasten](https://en.wikipedia.org/wiki/Zettelkasten).
It is a system of connections between notes based on **search**, **links** and **tags**.
Connections between notes allow to create a so-called "second brain" with interconnected ideas.

However links and tags need to be created manually, which makes this system not so fun to use.

### Idea
This repository contains code for automatic tagging (creating tags/zero-links) notes using pretrained Transformer neural networks. Notes are vectorized using pretrained NLP models and then clustered using unsupervised KMeans algorithm.

![**Before**](images/b4.jpg?raw=True "Plain uncategorized notes")

![**After**](images/after.jpg?raw=True "Notes linked to clusters")
<!-- **Before: **
[[images/b4.jpg]]

**After: **
[[images/after.jpg]] -->

### Algorithm
1) Notes are loaded with python
2) Headers and note text are vectorized with a pretrained NN with good semantic understanding (like multilingual sentence BERT from [deeppavlov](https://deeppavlov.ai) library)
3) Vectors are separated to N clusters, where optimal N is selected manually or to maximize the silhouette clustering score.
4) Each cluster is vebalized as separate tag/zero-link, unifying all cluster notes together.
5) Header and text cluster labels are used as new tags and are appended to original files.


### Usage
- Make sure you have a GPU or enough CPU to run a NN inference.
- Install torch, transformers, sklearn
- Add path to your DB to config.json.
- Run tagging:
    python main.py
- Visualize using some zettelkasten tool (e.g. [Obsidian](https://obsidian.md/))
