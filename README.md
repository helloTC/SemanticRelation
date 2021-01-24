# SemanticRelation
Code for paper: Semantic relatedness emerges in deep convolutional neural networks designed for object recognition

## Code
1. Prepare WordNet semantic similarity: WordNet_preparation.py <br />
2. Extraction of DCNN activation: extract_imagenet_activation.py <br />
3. Stability of representational similarity for different implementations: DCNN_consistency.py <br />
4. Model training: train_model.py <br />
5. Developmental trajectory: corr_traincatesim.py, merge_mat.py and corr_traincatesim_singleobj.py <br />
6. Examine no parent-child relationship among the 1000 categories: examine_no_conceptual_relationship.py <br />
7. Effect of task demand on the emergence of semantic relatedness: taskdemand_comparecorr.py <br />

## Files
Data was provided in folder: data.
- cate_pathsim_wup.npy: WordNet semantic similarity <br />
- hypernyms_idx.npy: rearrange categories according to the WordNet hierarchy (For detail please refer to WordNet_preparation.py) <br />
- DCNNsim: representational similarity of different DCNNs. <br />
- AlexNetsim_layers: Representational similarity of AlexNet of each layers. Noted that validation_corr_alexnet_fc3.npy is same to validation_corr_alexnet_fc.npy from DCNNsim <br />
- DevelopTraj: developmental trajectory for the original AlexNet and the single-object AlexNet. Values from them are correspondences to the WordNet semantic similarity in different training stages.
- TaskDemand: Averaged responses of AlexNet-Cate2, AlexNet-Cate19 and AlexNet-Cate1000 in different layers. Responses are formatted as [Category x Units].

Feel free to contact me if you have any questions :)
