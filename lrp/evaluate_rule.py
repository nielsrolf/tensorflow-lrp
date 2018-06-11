import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

"""
Evaluate Heatmap rules as proposed in arxiv.org/pdf/1509.06321.pdf
"""


class HeatmapEval():
    def __init__(self, networks, names, rules, rule_names, data, window_size=1):
        """
        Creates an object to evaluate different heatmapping methods on a number of neural networks
        :param networks: list of Network objects
        :param rules: list of (method [, kwargs]) accepted by Network.lrp
        :param data: a Data object
        :param window_size: the area of window_size x window_size around a pixel will be pertubated
        """
        self.networks, self.rule_names, self.data, self.window_size, self.names = \
            networks, rule_names, data, window_size, names
        self.nn_names = {nn: name for nn, name in zip(networks, names)}
        rules = [rule if isinstance(rule, (list, tuple)) else [rule] for rule in rules]
        rules = [rule if len(rule)==2 else rule+[{}] for rule in rules]
        self.rules = rules
        self.heatmaps = {nn: [nn.lrp(data.y_, rule[0], **rule[1]) for rule in rules] for nn in networks}
        self.effect = None

    def neighborhood(self, pixel):
        if self.window_size == 1: return pixel
        try:
            features_count = np.prod(self.data.test_batch().shape[1:])
            pixel_ids = np.array(range(int(features_count)))
            img = np.reshape(pixel_ids, self.data.img_shape)
            pixel_x, pixel_y = np.where(img==pixel)[0]
            n_offset = int(self.window_size/2)
            p_offset = self.window_size - n_offset
            max_x, max_y = self.data.img_shape[:2]
            neighbors = img[max(0, pixel_x-n_offset):min(pixel_x+p_offset, max_x),
                            max(0, pixel_y-n_offset):min(pixel_y+p_offset, max_y)]
            return neighbors.flatten()
        except AttributeError:
            raise AttributeError(
                "In order to perturb the neighborhood of size ({0}, {0}), data must have the attribute 'img_shape'"
                .format(self.window_size))

    def eval(self, nn, H):
        # evals on test set
        # returns average score of correct class for [0, ..., #features] number of perturbations
        h = nn.sess.run(H, feed_dict=self.data.test_batch())
        order = np.argsort(-h)
        images, labels = self.data.test_batch()[self.data.X], self.data.test_batch()[self.data.y_]
        scores = []

        for pixel in order:
            perturbation, neighborhood = self.get_neighborhood(pixel)
            images[neighborhood] = perturbation
            scores.append(
                np.mean(
                    nn.sess.run(nn.y, feed_dict={self.data.X: images, self.data.y_: labels}).dot(labels)
                )
            )

        return np.array(scores)

    def compare(self):
        # neural networks should be trained at this point
        effect = {rule: {} for rule in self.rules}  # rule => nns u {"avg"} => score/rnd
        for nn, H in self.heatmaps.items():
            scores = self.eval(nn, H)
            rnd_scores = self.eval(nn, tf.random.normal(self.data.X_test.shape))
            effect[rule][self.nn_name[nn]] = scores/rnd_scores
        for rule in self.rules:
            effect[rule]["avg"] = np.mean(effect[rule].values(), axis=0)

        self.effect = effect

    def plot_effect(self, result_dir):
        assert self.effect is not None, "call .compare() before .plot_effect()"

        # overview per rule
        for rule_id, rule, rule_name in enumerate(zip(self.rules, self.rule_names)):
            for dom in self.effect[rule]:
                effect = self.effect[rule][dom]
                plt.plot(range(len(effect)), effect, label=dom)
            plt.savefig(result_dir+"/"+rule_name+".png")
            plt.clf()

        # overview per network
        for dom in self.effect[rule]:
            for rule_id, rule, rule_name in enumerate(zip(self.rules, self.rule_names)):
                effect = self.effect[rule][dom]
                plt.plot(range(len(effect)), effect, label=rule_name)
            plt.savefig(result_dir+"/"+dom+".png")











