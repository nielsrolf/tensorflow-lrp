from lrp import *
from lrp.data import Data
from lrp import read_mnist
import os
from lrp.evaluate_rule import HeatmapEval
from lrp import utils
import os

def eval_example():
    mnist = read_mnist.read_data_sets("{}/datasets/mnist".format(os.environ["TF_PROJECTS"]), one_hot=True)
    data = Data(X=mnist.train.images, y=mnist.train.labels,
                X_val=mnist.validation.images, y_val=mnist.validation.labels,
                X_test=mnist.test.images, y_test=mnist.test.labels)
    fcn = Network([Format(), FirstLinear(784), ReLU(), NextLinear(10)], data.X, data.y_)
    cnn = Network([Format(), FirstConvolution([5, 5, 1, 32]), ReLU(), Pooling(),
                   NextConvolution([5, 5, 32, 64]), ReLU(), Pooling(),
                   NextLinear(1024), ReLU(),
                   NextLinear(10)],
                   data.X, data.y_)
    nns = [fcn, cnn]
    rules = ["deeptaylor",
             "zbab",
             ("simple", {"reference": data.X_val})]

    evaluator = HeatmapEval(nns, ["fcn", "cnn"],
                            rules, ["deeptaylor", "zbab", "ref-deeptaylor"],
                            data)

    with tf.Session() as sess:
        for nn in nns:
            nn.set_session(sess)
        sess.run(tf.global_variables_initializer())

        os.makedirs("evaluate/heatmaps", exist_ok=True)

        for nn_name, nn in zip(["fcn", "cnn"], nns):
            nn.fit(data, stopping_criterion=lambda i, _: i<300)
            for rule_name, H in zip(evaluator.rule_names, evaluator.heatmaps[nn]):
                heatmaps = sess.run(H, feed_dict={data.X: data.X_test[:16], data.y_: data.y_test[:16]})
                utils.visualize(heatmaps, utils.heatmap_original, "evaluate/heatmaps/{}-{}.png".format(
                    nn_name, rule_name))

        evaluator.compare()
        evaluator.plot_effect(".")



eval_example()


