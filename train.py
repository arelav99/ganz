from lib.models.gan import Ganomaly
from lib.models.skipgan import SkipGanomaly
from lib.options import parse_args
from lib.dataloader import load_dataset, split_dataset

if __name__ == "__main__":
    opts = parse_args()

    X_tr, X_tst, Y_tst = load_dataset(
        opts.dataset, opts.abnormal_class)

    # if opts.model == "ganomaly":
    #     gan = Ganomaly(opts)
    # elif opts.model == "skipganomaly":
    #     gan = SkipGanomaly(opts)

    # if opts.fit_like == "fit":
    #     gan.fit(X_tr)
    # elif opts.fit_like == "fit_with_test":
    #     gan.fit_with_test(X_tr, (X_tst, Y_tst))
