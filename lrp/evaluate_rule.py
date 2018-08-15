import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from lrp import utils, flatten_batch
import gc
import os

"""
Evaluate Heatmap rules as proposed in arxiv.org/pdf/1509.06321.pdf
"""

NUM_IMAGES = 100

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
            networks, rule_names + ["rnd"], data, window_size, names
        self.nn_names = {nn: name for nn, name in zip(networks, names)}
        rules = [rule if isinstance(rule, (list, tuple)) else [rule] for rule in rules]
        rules = [rule if len(rule)==2 else rule+[{}] for rule in rules]
        self.rules = rules
        self.heatmaps = {nn: [flatten_batch(nn.lrp(data.y_, rule[0], **rule[1])) for rule in rules] for nn in networks}
        self.effect = None

        self.rnd_heatmap = tf.random_normal([NUM_IMAGES]+list(self.data.img_shape), 0, 1)


    """
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
    """

    def eval(self, nn, H, visualize=False):
        print("Eval ", self.nn_names[nn])
        # evals on test set
        # returns average score of correct class for [0, ..., #features] number of perturbations
        images, labels = np.array(self.data.test_batch()[self.data.X][:NUM_IMAGES]), self.data.test_batch()[self.data.y_][:NUM_IMAGES]
        test_data = {self.data.X: images,
                     self.data.y_: labels}
        h = nn.sess.run(H, feed_dict=test_data)
        order = np.argsort(-h).T
        scores = []

        print("Scores")
        print("order", order.shape)
        for i, pixel in enumerate(order):
            #neighborhood = self.neighborhood(pixel)
            for image_id, p in enumerate(pixel):
                images[image_id,p] = np.random.normal(0, 1)

            if i % 5 == 0:
                scores.append(
                    np.mean(
                        np.sum(nn.sess.run(nn.y, feed_dict={self.data.X: images, self.data.y_: labels})*labels, axis=1)
                    )
                )
            if i%50 == 0 and visualize:
                utils.visualize(images[:16], utils.graymap_direct, "pertubated/{}-pert-{}.png".format(visualize, i))
        return np.array(scores)

    def compare(self):
        # neural networks should be trained at this point
        effect = {rule: {} for rule in self.rule_names}  # rule => nns u {"avg"} => score/rnd
        os.makedirs("pertubated", exist_ok=True)
        for nn, all_H in self.heatmaps.items():
            for H, rule_name in zip(all_H, self.rule_names):
                print("eval", nn, rule_name)
                scores = self.eval(nn, H, self.nn_names[nn]+"-"+rule_name)
                gc.collect()
                effect[rule_name][self.nn_names[nn]] = scores
            print("eval", nn, "rnd")
            effect["rnd"][self.nn_names[nn]] = self.eval(nn, self.rnd_heatmap)
        for rule in self.rule_names:
            effect[rule]["avg"] = np.mean(np.array(list(effect[rule].values())), axis=0)

        self.effect = effect

    def plot_effect(self, result_dir):
        assert self.effect is not None, "call .compare() before .plot_effect()"

        for rule_name in self.effect.keys():
            plt.clf()
            plt.title(rule_name)
            for nn_name in self.effect[rule_name].keys():
                scores = self.effect[rule_name][nn_name]
                plt.plot(np.array(list(range(len(scores))))*5, scores, label=nn_name)
            plt.legend()
            plt.savefig(result_dir+"/"+rule_name+".png")
            nn_names = self.effect[rule_name].keys()

        for nn_name in nn_names:
            if nn_name == "avg": continue
            plt.clf()
            plt.title(nn_name)
            for rule_name in self.effect.keys():
                scores = self.effect[rule_name][nn_name]
                plt.plot(np.array(list(range(len(scores))))*5, scores, label=rule_name)
            plt.legend()
            plt.savefig(result_dir+"/"+nn_name+".png")

        """
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
        """











