import torch

from params_proto.neo_proto import ParamsProto, PrefixProto, Proto

class Config(ParamsProto):
    # misc
    seed = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bucket = '/home/paperspace/decision-diffuser/weights/'
    dataset = 'env1' #'my_datasetExperiment1Expert.hdf'

    ## modelbbb
    model = 'models.TemporalUnet'
    diffusion = 'models.GaussianInvDynDiffusion'
    horizon = 12
    n_diffusion_steps = 200
    action_weight = 10
    loss_weights = None
    loss_discount = 1
    predict_epsilon = True
    dim_mults = (1, 4, 8)
    returns_condition = True
    calc_energy=False
    dim=128
    condition_dropout=0.25
    condition_guidance_w = 1.3
    test_ret=0.9 # was ist das?
    renderer = 'utils.MuJoCoRenderer'

    ## dataset
    loader = 'datasets.GoalDataset'
    normalizer = 'CDFNormalizer'
    preprocess_fns = []
    clip_denoised = True
    use_padding = True
    include_returns = True
    discount = 0.99
    max_path_length = 1000
    hidden_dim = 256
    ar_inv = False
    train_only_inv = False
    termination_penalty = None
    returns_scale = 400.0 # Determined using rewards from the dataset

    ## training
    n_steps_per_epoch = 1000
    loss_type = 'l2'
    n_train_steps = 1e5
    batch_size = 32
    learning_rate = 2e-4
    gradient_accumulate_every = 2
    ema_decay = 0.995
    log_freq = 100
    save_freq = 5000
    sample_freq = 10000
    n_saves = 5
    save_parallel = True
    n_reference = 8
    save_checkpoints = False
