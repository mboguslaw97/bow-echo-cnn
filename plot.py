import matplotlib.pyplot as plt
import numpy as np
import pickle

ratios = [10] #range(10, 45, 5)
mults = [10] #[.25, .5, 1, 2, 4]

abs_dir = '/storage/home/meb6031/bow_echo_cnn/plots/run1/'  # aci
train_accs_map = pickle.load(open(os.path.join(abs_dir, 'train_accs_map.p'), "rb"))
test_accs_map = pickle.load(open(os.path.join(abs_dir, 'test_accs_map.p'), "rb"))

for accs_map, stage in [(train_accs_map, 'train'), (test_accs_map, 'test')]:
    for i, acc_of in enumerate(['total', 'not bow echo', 'bow echo']):
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 4, 1)
        for ratio in ratios:
            plt.xlabel('multiplier')
            plt.ylabel('accuracy')
            accs = [accs_map[(ratio, mult)][i] for mult in mults]
            plt.plot(mults, accs, '-o')
        # plt.legend(range(10, 45, 5), title='ratio:')

        for i, ratio in enumerate(range(10, 45, 5)):
            plt.subplot(2, 4, i + 2)
            plt.title('ratio: {}'.format(ratio))
            accs = [accs_map[(ratio, mult)][i] for mult in mults]
            plt.plot(mults, accs, '-o')
        filename = '/storage/home/meb6031/bow_echo_cnn/plots/run1/{}_{}_accuracy.png'.format(stage, acc_of)
        plt.savefig(filename, bbox_inches='tight')