from torchcrf import CRF
import torch
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
Note this code is primarily copied from https://github.com/kmkurn/pytorch-crf/tree/master/torchcrf with modifications to allow a Fuzzy-CRF layer, and also 
functionality to penalize illegal transitions
'''


class CustomCRF(CRF):
    '''
    If just normal CRF layer: Mainly from https://github.com/kmkurn/pytorch-crf/tree/master/torchcrf except also penalizes illegal transitions
    during training.
    '''
    def __init__(self, num_tags, label_list, penalty=0):
        super().__init__(num_tags)
        
        end_transitions_pen, start_transitions_pen, transitions_pen = self.add_hard_crf_bilu(label_list,penalty)
        self.end_transitions_pen_argmax = end_transitions_pen
        self.start_transitions_pen_argmax = start_transitions_pen
        self.transitions_pen_argmax = transitions_pen
        
        # no penalty at inference time
        end_transitions_pen_inference, start_transitions_pen_inference, transitions_pen_inference = self.add_hard_crf_bilu(label_list,0) 
        self.end_transitions_pen_inference = end_transitions_pen_inference
        self.start_transitions_pen_inference = start_transitions_pen_inference
        self.transitions_pen_inference = transitions_pen_inference


    def add_hard_crf_bilu(self, labs, penalty):
        '''
        Creates penalties to add to illegal transitions such as I-Per -> B-Per
        '''
        
        num_tags = self.num_tags
        #only want to start either 'O', 'B-', 'U'
        #recall we use Positioned labels shifted down by one
        inds_to_use = [i-1 for i,v in enumerate(labs) if v[0] in ['B']] + [num_tags-1]
        start_transitions_pen = penalty * torch.ones(num_tags) #don't want to start with
        start_transitions_pen[inds_to_use] = 0

        #End with either 'O', 'B', 'I' (recall we have 'O' as last element)
        inds_to_use = [i-1 for i,v in enumerate(labs) if v[0] in ['B','I']] + [num_tags-1]
        end_transitions_pen = penalty * torch.ones(num_tags) 
        end_transitions_pen[inds_to_use] = 0

        # Transitions are from the state in the rows, to the state in the col
        transitions_pen = penalty*torch.ones(num_tags,num_tags)
        for i,v in enumerate(labs):
            #if i == 0, should correspond to last row of end_transitions
            inds_to_use = []
            for j,v2 in enumerate(labs):
                if ((v == 'O' and v2[0] in ['O','B']) or
                   (v[0] == 'B' and (v2[0] == 'O' or (v2[0] == 'I' and v2[2:] == v[2:]) or (v2[0] == 'B' and v2[2:] != v[2:]))) or 
                   (v[0] == 'I' and (v2[0] == 'O' or (v2[0] == 'I' and v2[2:] == v[2:]) or (v2[0] == 'B' and v2[2:] != v[2:])))):
                    
                    inds_to_use.append(j-1 if j!=0 else num_tags-1)
            
            row_ind = i-1 if i!=0 else num_tags-1
            transitions_pen[[row_ind]*len(inds_to_use), inds_to_use] = 0

        transitions_pen = transitions_pen.to(DEVICE) 
        start_transitions_pen = start_transitions_pen.to(DEVICE) 
        end_transitions_pen = end_transitions_pen.to(DEVICE) 
        return end_transitions_pen, start_transitions_pen, transitions_pen
    
     
    def decode(self, emissions: torch.Tensor,
               mask: torch.ByteTensor = None, use_hard_crf=True, em_argmax=False):
        
        self._validate(emissions, mask=mask)
        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        return self._viterbi_decode(emissions, mask, use_hard_crf, em_argmax)
    
    
    def _viterbi_decode(self, emissions: torch.FloatTensor,
                        mask: torch.ByteTensor, use_hard_crf=True, em_argmax=False):
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length, batch_size = mask.shape

        end_transitions_pen = self.end_transitions_pen_inference if not em_argmax else self.end_transitions_pen_argmax 
        start_transitions_pen = self.start_transitions_pen_inference if not em_argmax else self.start_transitions_pen_argmax 
        transitions_pen = self.transitions_pen_inference if not em_argmax else self.transitions_pen_argmax 
        
        score = self.start_transitions + start_transitions_pen*use_hard_crf + emissions[0]
        history = []

        # score is a tensor of size (batch_size, num_tags) where for every batch,
        # value at column j stores the score of the best tag sequence so far that ends
        # with tag j
        # history saves where the best tags candidate transitioned from; this is used
        # when we trace back the best tag sequence

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        for i in range(1, seq_length):
            # Broadcast viterbi score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emission = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the score of the best
            # tag sequence so far that ends with transitioning from tag i to tag j and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + transitions_pen*use_hard_crf + broadcast_emission

            # Find the maximum score over all possible current tag
            # shape: (batch_size, num_tags)
            next_score, indices = next_score.max(dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions + end_transitions_pen*use_hard_crf

        # Now, compute the best path for each sample

        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []

        for idx in range(batch_size):
            # Find the tag which maximizes the score at the last timestep; this is our best tag
            # for the last timestep
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]

            # We trace back where the best last tag comes from, append that to our best tag
            # sequence, and trace it back again, and so on
            for hist in reversed(history[:seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            # Reverse the order because we start from the last timestep
            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list
    


# Here we define a Fuzzy-CRF layer
class FuzzyCRF(CRF):
    def __init__(self, num_tags):
        super().__init__(num_tags)
    
    def _viterbi_decode(self, emissions: torch.FloatTensor,
                        mask: torch.ByteTensor):
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length, batch_size = mask.shape

        # Start transition and first emission
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]
        
        history = []

        # score is a tensor of size (batch_size, num_tags) where for every batch,
        # value at column j stores the score of the best tag sequence so far that ends
        # with tag j
        # history saves where the best tags candidate transitioned from; this is used
        # when we trace back the best tag sequence

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        for i in range(1, seq_length):
            # Broadcast viterbi score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emission = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the score of the best
            # tag sequence so far that ends with transitioning from tag i to tag j and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emission

            # Find the maximum score over all possible current tag
            # shape: (batch_size, num_tags)
            next_score, indices = next_score.max(dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions 

        # Now, compute the best path for each sample

        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []

        for idx in range(batch_size):
            # Find the tag which maximizes the score at the last timestep; this is our best tag
            # for the last timestep
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]

            # We trace back where the best last tag comes from, append that to our best tag
            # sequence, and trace it back again, and so on
            for hist in reversed(history[:seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            # Reverse the order because we start from the last timestep
            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list
    
    
    # Only need to override the forward method (decoding is fine)
    def forward(
            self,
            emissions: torch.Tensor,
            class_mask: torch.Tensor = None,
            mask: torch.ByteTensor = None,
            reduction: str = 'sum',
            tags = None,
    ) -> torch.Tensor:
        """Compute the conditional log likelihood of a sequence of tags given emission scores.
        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            class_mask (`~torch.ByteTensor`)

            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
            reduction: Specifies  the reduction to apply to the output:
                ``none|sum|mean|token_mean``. ``none``: no reduction will be applied.
                ``sum``: the output will be summed over batches. ``mean``: the output will be
                averaged over batches. ``token_mean``: the output will be averaged over tokens.
        Returns:
            `~torch.Tensor`: The log likelihood. This will have size ``(batch_size,)`` if
            reduction is ``none``, ``()`` otherwise.
        """
        #self._validate(emissions, tags=tags, mask=mask)
        if reduction not in ('none', 'sum', 'mean', 'token_mean'):
            raise ValueError(f'invalid reduction: {reduction}')
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            class_mask = class_mask.transpose(0, 1)
            mask = mask.transpose(0, 1)

        # shape: (batch_size,)

        numerator = self._compute_score(emissions, class_mask, mask)
        
        # shape: (batch_size,)
        denominator = self._compute_normalizer(emissions, mask)
        # shape: (batch_size,)
        llh = numerator - denominator

        if reduction == 'none':
            return llh
        if reduction == 'sum':
            return llh.sum()
        if reduction == 'mean':
            return llh.mean()
        assert reduction == 'token_mean'
        return llh.sum() / mask.type_as(emissions).sum()


    def _compute_score(self, emissions: torch.Tensor, class_mask: torch.Tensor,
            mask: torch.ByteTensor) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size) has 1 when is not padding
        # class_mask: Has one on classes which were voted on by weak labelers for each word
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length = emissions.size(0)

        # Start transition score and first emission; score has size of
        # (batch_size, num_tags) where for each batch, the j-th column stores
        # the score that the first timestep has tag j
        # shape: (batch_size, num_tags)

        # NOTE: Any unwanted sequence needs to have score -inf to not count in the loss
        score = self.start_transitions + emissions[0] + class_mask[0] # 0 out contribution from any non allowed classes

        for i in range(1, seq_length):
            # Broadcast score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emissions = emissions[i].unsqueeze(1)
            broadcast_class_mask = class_mask[i].unsqueeze(1)
            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the sum of scores of all
            # possible tag sequences so far that end with transitioning from tag i to tag j
            # and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emissions + broadcast_class_mask

            # Sum over all possible current tags, but we're in score space, so a sum
            # becomes a log-sum-exp: for each sample, entry i stores the sum of scores of
            # all possible tag sequences so far, that end in tag i
            # shape: (batch_size, num_tags)
            next_score = torch.logsumexp(next_score, dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Sum (log-sum-exp) over all possible tags
        # shape: (batch_size,)
        return torch.logsumexp(score, dim=1)

    
