import argparse


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def parse_args():
    parser = argparse.ArgumentParser(description="""S""")
    parser.add_argument("--nc", help="Imgs dim", type=int, default=1)
    parser.add_argument("--size", help="Image size", type=int, default=32)
    parser.add_argument("--nz", help="Hidden vectr dims",
                        type=int,  default=100)
    parser.add_argument("--ngf", help="Iter size of the model",
                        type=int,  default=16)  # CHANGE LATER TO 32 / 64
    parser.add_argument("--batch_size", type=int,  default=32)
    parser.add_argument("--save_weights_path", default="./weights")
    parser.add_argument("--dataset", help="mnist|cifar|path to custom dataset")
    parser.add_argument("--model", help="ganomaly|skipganomaly",
                        choices=['ganomaly', 'skipganomaly'], default="ganomaly")
    parser.add_argument(
        "--print_steps_freq", help="How often to print  the results", default=300)

    # Train
    parser.add_argument("--w_adv", default=1)
    parser.add_argument("--w_con", default=50)
    parser.add_argument("--w_enc", default=1)
    parser.add_argument(
        '--lr', help='Learning rate', default=1e-3)
    parser.add_argument(
        '--epochs', help='Number of epochs', default=1)
    parser.add_argument(
        '--abnormal_class', help='Abnormal class number', type=int, required=True)
    parser.add_argument(
        '--visualize_imgs', help='Visualize images while training',
        default=False, action="store_true")
    parser.add_argument(
        "--fit_like", choices=["fit", "fit_with_test"], default="fit_with_test")

    # Test
    parser.add_argument("--g_weights_path", type=str)
    parser.add_argument("--d_weights_path", type=str)

    return dotdict(vars(parser.parse_args()))
