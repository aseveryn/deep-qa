# OVERVIEW

This code implements a convolutional neural network architecture for learning to match question and answer sentences described in the paper:

Aliaksei Severyn and Alessandro Moschitti. *Learning to Rank Short Text Pairs with Convolutional Deep Neural Networks*. SIGIR, 2015

The network features a state-of-the-art convolutional sentence model, advanced question-answer matching model, and introduces a novel relational model to encode related words in a question-answer pair.

The addressed task is a popular answer sentence selection benchmark, where the goal is for each question to select relevant answer sentences. The dataset was first introduced by (Wang et al., 2007) and further elaborated by (Yao et al., 2013). It is freely [availabe](http://cs.jhu.edu/~xuchen/packages/jacana-qa-naacl2013-data-results.tar.bz2).

Evaluation is performed using the standard 'trec_eval' script.


# DEPENDENCIES

- python 2.7+
- numpy
- [theano](http://deeplearning.net/software/theano/)
- scikit-learn (sklearn)
- pandas
- tqdm
- fish
- numba

Python packages can be easily installed using the standard tool: pip install <package>


# EMBEDDINGS

The pre-initialized word2vec embeddings have to be downloaded from [here](https://drive.google.com/folderview?id=0B-yipfgecoSBfkZlY2FFWEpDR3M4Qkw5U055MWJrenE5MTBFVXlpRnd0QjZaMDQxejh1cWs&usp=sharing).


# BUILD

To build the required train/dev/test sets in the suitable format for the network run:

>$ sh run_build_datasets.sh

It will parse the raw XML files containg QA pairs and convert them into a suitable format for the deep learning model.
The output files are stored under the folders TRAIN and TRAIN-ALL corresponding to the TRAIN and TRAIN-ALL training settings as described in the paper.

At the next step the script will extract the word embeddings for the all words in the vocabulary.
We use the pre-trained word embeddings obtained by running the word2vec tool on a merged Wiki dump and Aquaint corpus (provided under the 'embeddings' folder.
The missing words are randomly initalized with the uniform distribution [-0.25; +0.25]. For the further details please refer to the paper.


# TRAIN AND TEST

To train the model in the TRAIN setting run:

>$ python run_nnet.py TRAIN

in the TRAIN-ALL setting using 53,417 qa pairs:

>$ python run_nnet.py TRAIN-ALL

The parameters of the trained network are dumped under the 'exp.out' folder.

The results reported by the 'trec_eval' script should be around these numbers:

TRAIN:
MAP: 0.7325
MRR: 0.8018

TRAIN-ALL:
MAP: 0.7654
MRR: 0.8186

NOTE: Small variations on different platforms are expected due to differences in random seeds which affect random initialization of network weights.

# REFERENCES

Peter Clark Xuchen Yao, Benjamin Van Durme and Chris Callison-Burch.
Answer extraction as sequence tagging with tree edit distance.
In NAACL, 2013.

Mengqiu Wang, Noah A. Smith, and Teruko Mitaura.
What is the jeopardy model? a quasi- synchronous grammar for qa.
In EMNLP, 2007.

# License

This software is licensed under the [Apache 2 license](http://www.apache.org/licenses/LICENSE-2.0).
