import hydra
import matplotlib
import matplotlib.pyplot as plt
from hydra.utils import instantiate

matplotlib.use('TkAgg', force=False)


@hydra.main(version_base=None, config_path="cfg/datasets/dataset", config_name="nonlinear_oscillator_dataset")
def main(cfg):
    dataset = instantiate(cfg)
    if __debug__:
        dataset.plot()
        plt.show()


if __name__ == "__main__":
    main()
