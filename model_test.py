from lib.solver_WGAN import CaptioningSolver
from lib.generator_WGAN import CaptionGenerator
from lib.utils import load_coco_data
import os

def main():

    # switch is for specifying the targeted sentiment:
    #  1 is for positive
    # -1 is for negative

    switch = [1]

    # data_save_path is the location of your image captioning dataset and needs to be changed
    if switch == [1]:
       data_save_path = './data/positive/'
    else:
       data_save_path = './data/negative/'

    # other paths need to be customized
    image_save_path = './images/'
    save_path = './models/'
    save_path_pretrain = './models/'
    log_save_path = './log/'


    # load data splits and the vocabulary
    data = load_coco_data(data_path=data_save_path, split='train')
    word_to_idx = data['word_to_idx']
    val_data = load_coco_data( data_path=data_save_path, split='val' )
    test_data = load_coco_data( data_path=data_save_path, split='test' )

    # build a model
    model = CaptionGenerator( word_to_idx, dim_feature=[49, 2048], dim_embed=300,
                                       dim_hidden=512, n_time_step=20, beam_index=5, prev2out=True,
                                                 ctx2out=True, alpha_c=1.0, selector=True, dropout=True )

    # build the training and testing
    solver = CaptioningSolver( model, data, val_data, n_epochs=20, batch_size=64, dis_batch_size=80, update_rule='adam',
                                          learning_rate=0.0001, print_every=1, save_every=1, image_path=image_save_path,
                                    pretrained_model=None, model_path=save_path, test_model=os.path.join(save_path_pretrain+'model_15'),
                                     print_bleu=True, log_path=log_save_path, rollout_num=5, dis_dropout_keep_prob=1.0,
                               data_path=data_save_path )

    # test the model after training
    solver.test( test_data, 'test', senti=switch )

if __name__ == "__main__":
    main()
