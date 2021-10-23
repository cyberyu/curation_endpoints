

from snorkel.labeling.model import LabelModel, MajorityLabelVoter
import numpy as np
from typing import List

def fit_and_predict_snorkel(label_matrix: List[List[int]]) -> List[int]:
    """[summary]

    Arguments:
        label_matrix {List[List[int]]} -- 
            shape is (n,m) where n is num examples, and m is number of classes.
            Each entry should be the index of the predicted class or -1 for 
            abstain. Therefore the bounds on the matrix elements are >= -1 
            and < m.

    Returns:
        List[int] -- predictions for each sentence
    """
    label_matrix = np.array(label_matrix)
    num_classes = max(label_matrix.flatten()) + 1
    if num_classes <= 0:
        # What to do when no preds. Just return empty preds
        return np.array([-1]*label_matrix.shape[0])
    
    label_model = LabelModel(cardinality=num_classes, verbose=True)
    label_model.fit(L_train=label_matrix)
    
    # now get predictions with resulting model
    preds = label_model.predict(label_matrix)
    return preds.tolist()


def predict_majority_vote(label_matrix: List[List[int]]) -> List[int]:
    """
    Arguments:
        label_matrix {List[List[int]]} -- 
            shape is (n,m) where n is num examples, and m is number of classes.
            Each entry should be the index of the predicted class or -1 for 
            abstain. Therefore the bounds on the matrix elements are >= -1 
            and < m.

    Returns:
        List[int] -- predictions for each sentence
    
    """
    label_matrix = np.array(label_matrix)
    num_classes = max(label_matrix.flatten()) + 1
    majority_model = MajorityLabelVoter(cardinality=num_classes)
    maj_preds = majority_model.predict(L=label_matrix, tie_break_policy='random')
    return maj_preds.tolist()


if __name__=='__main__':
    
    L_ex = np.array([[0,0,-1], [-1,0,1], [1,-1,0],[-1,1,1]])
    print('Snorkel preds: ', fit_and_predict_snorkel(L_ex))
    
    print('Majority vote preds: ', predict_majority_vote(L_ex))
    #L_test = np.array([[0,1,-1], [-1,1,1], [1,0,1]])



