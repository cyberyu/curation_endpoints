
import argparse

def get_args():
    parser = argparse.ArgumentParser('parameters')
    
    parser.add_argument('--file_ext', type=str, default='', help='extension of file to write results dataframe')
    parser.add_argument('--model_extension', type=str, default='', help='extension name for saved model')
    parser.add_argument('--local_data_path', type=str, default='/domino/datasets/jerrod_parker/WeakSupervision/scratch/ModelData/',
                        help='base directory to store models / cache data such as reusable token embeddings.')
    parser.add_argument('--s3_data_path', type=str, help='path to directory containing the training data in s3 bucket')
    parser.add_argument('--load_saved_model', action='store_true')
    #flags to change the pretrained model we're using 
    parser.add_argument('--model_to_use', default='distilbert-base-cased', help='model name is from https://huggingface.co/transformers/pretrained_models.html and must have fast tokenizer')
    parser.add_argument('--reuse_embeddings', action='store_true', help='Run bert model of choice over dataset to get embeddings that can be reused each epoch')
    parser.add_argument('--pred_spans_model_ext', default=None, help='Is name of model extension to get predicted entity spans from.')
    parser.add_argument('--weak_supervision_model', default='dws', choices=['dws', 'dws_use_spans', 'fuzzy_crf', 'simple_baseline'])
    #Training arguments
    parser.add_argument('--batch_num_spans', type=int, default=64, 
                        help='number of entity spans per batch to use when training on candidate entity spans')
    parser.add_argument('--batch_num_toks', type=int, default=256,
                        help='number of tokens per minibatch')

    parser.add_argument('--numpy_seed', type=int, default=2019, help='Set <= 0 if want random seed')
    parser.add_argument('--iteration_to_begin_em', type=int, default=1)
    parser.add_argument('--train_models', type=int, default=1, help='Trains DWS or Fuzzy CRF')
    
    #Arguments for LSTM
    parser.add_argument('--use_lstm', type=int, default=1,
                         help='whether to use lstm')
    parser.add_argument('--lstm_chunk_sz', type=int, default=128,
                         help='number of tokens for LSTM to encode at once')
    parser.add_argument('--lstm_num_layers', type=int, default=1,
                         help='number of tokens for LSTM to encode at once')
    parser.add_argument('--lstm_hidden_sz', type=int, default=768,
                         help='number of tokens for LSTM to encode at once')

    #DWS Arguments
    parser.add_argument('--illegal_trans_penalty', type=float, default=10, help='penalize illegal transitions with the negative of this number during the EM phase')
    parser.add_argument('--num_restarts', type=int, default=2, 
                         help='Number of random restarts')

    parser.add_argument('--block_sz', type=int, default=500,
                         help='number of tokens for Roberta to encode at once')
    
    parser.add_argument('--initial_lr', type=float, default=0.001, help='initial learning rate during warm start')
    parser.add_argument('--initial_lr_em', type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--train_iterations", type=int, default=7)
    parser.add_argument("--batch_size", type=int, default=256, help='Not used for tokenwise')
    parser.add_argument("--entity_embed_sz", type=int, default=768, help='Dimension of the entity and class embeddings')
    parser.add_argument('--grad_clip_value', type=float, default=40)

    parser.add_argument('--use_token_level_spans', type=int, default=1, help="Is always true")
    parser.add_argument('--min_votes_fuzzy', type=int, default=1,
                         help='min votes a token must have for FuzzyLSTMCRF to use in loss function')
    parser.add_argument('--min_votes_for_token', type=int, default=1,
                         help='minimum number of weak labelers to vote on token for us to include in loss for DWS')
    parser.add_argument('--min_votes_during_warm_start', type=int, default=1,
                         help="In warmup before doing EM, only train on tokens with more than \
                              this many votes. This is reasonable since during warm start there are no \
                              'nones' so may only want to train on examples which we know are very likely \
                              actually an entity (here assuming lots of votes means there is probably an \
                              entity at the token)")
    #args = parser.parse_args()
    args, unknown_args = parser.parse_known_args()
    return args
