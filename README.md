# GNN Pre-Training for Context-Based Fake News Detection

We evaluate how pre-training of GNNs on the FakeNewsNet (Politifact/Gossipcop) dataset influenced fake news detection performance.

This repository contains all the required code to re-run all of our experiments. In addition, we provide our pre-trained models.


### Pre-Trained Models

`models_GOS`: Models that have been pre-trained on the Gossipcop dataset. Name of each file represents node- and graph-level pre-training tasks as well as run id.

`models_POL`: Models that have been pre-trained on the Politifact dataset. Same names pattern as for Gossipcop models.


### Re-Running Experiments

`run_pretraining.ipynb`: Code to re-run the pretraining (this step can be skipped if you use the models from the folders just described).

`run_finetuning.ipynb`: Code to re-run the fine-tuning.


# Citation

The paper based on this idea was accepted at LREC-COLING 2024. If you use parts of our code or adopt our approach we kindly ask you to cite our work as follows:
```
@misc{donabauer2024challenges,
    title={Challenges in Pre-Training Graph Neural Networks for Context-Based Fake News Detection: An Evaluation of Current Strategies and Resource Limitations}, 
    author={Gregor Donabauer and Udo Kruschwitz},
    year={2024},
    eprint={2402.18179},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

# References

We adopt GNN pre-training strategies and graph-based fake news detection approaches from the following papers:
```
Strategies for Pre-training Graph Neural Networks. W. Hu*, B. Liu*, J. Gomes, M. Zitnik., P. Liang, V. Pande, J. Leskovec.
International Conference on Learning Representations (ICLR), 2020. 
```

```
Exploring Fake News Detection with Heterogeneous Social Media Context Graphs. G.Donabauer, U.Kruschwitz.
European Conference on Information Retrieval (ECIR), 2023.
```