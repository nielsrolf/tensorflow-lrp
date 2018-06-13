from lrp import *
from lrp.data import Data
from lrp import read_mnist
import os
from lrp.evaluate_rule import HeatmapEval
from lrp import utils
import os

import pdb

def conservation_test(architecture, a_name, rule_args, rule_name, **rule_kwarg):
    mnist = read_mnist.read_data_sets("{}/datasets/mnist".format(os.environ["TF_PROJECTS"]), one_hot=True)
    data = Data(X=mnist.train.images, y=mnist.train.labels,
                X_val=mnist.validation.images, y_val=mnist.validation.labels,
                X_test=mnist.test.images, y_test=mnist.test.labels)
    nn = Network(architecture, data.X, data.y_)

    if rule_kwarg.get("reference")=="val":
        rule_kwarg["reference"] = data.X_val

    H, C = nn.layerwise_lrp(data.y_, *rule_args, **rule_kwarg)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        nn.set_session(sess)
        nn.fit(data, stopping_criterion=lambda i, _: i<300)

        h, c = sess.run([H, C], feed_dict=data.validation_batch())
        print("Layerwise conservation for", a_name, rule_name, ":", c)
        utils.visualize(h[0][:16], utils.heatmap_original, "evaluate/heatmaps/{}-{}-conservation.png".format(
            a_name, rule_name))





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
    # rules = ["deeptaylor",
    #         "zbab",]

    rules = [("simple", {"reference": data.X_val})]
    #rules = ["simple"]

    evaluator = HeatmapEval(nns, ["fcn", "cnn"],
                            rules, ["simple"], # ["deeptaylor", "zbab", "ref-deeptaylor"],
                            data)

    with tf.Session() as sess:
        for nn in nns:
            nn.set_session(sess)
        sess.run(tf.global_variables_initializer())

        os.makedirs("evaluate/heatmaps", exist_ok=True)

        for nn_name, nn in zip(["fcn", "cnn"], nns):
            nn.fit(data, stopping_criterion=lambda i, _: i<150)

            for rule_name, H in zip(evaluator.rule_names, evaluator.heatmaps[nn]):
                heatmaps, prediction = sess.run([H, nn.y], feed_dict={data.X: data.X_test[:16], data.y_: data.y_test[:16]})
                utils.visualize(heatmaps, utils.heatmap_original, "evaluate/heatmaps/{}-{}.png".format(
                    nn_name, rule_name))
                relevance = np.sum(heatmaps, axis=tuple(range(1, len(heatmaps.shape))))
                prediction = np.sum(prediction*data.y_test[:16], axis=1)
                conservation_error =  np.where(prediction>0, prediction-relevance, relevance)
                print("Conservation Error for", nn_name, rule_name, "\n: ", conservation_error)

        evaluator.compare()
        evaluator.plot_effect(".")

def var_relevance(architecture):
    mnist = read_mnist.read_data_sets("{}/datasets/mnist".format(os.environ["TF_PROJECTS"]), one_hot=True)
    data = Data(X=mnist.train.images, y=mnist.train.labels,
                X_val=mnist.validation.images, y_val=mnist.validation.labels,
                X_test=mnist.test.images, y_test=mnist.test.labels)
    nn = Network(architecture, data.X, data.y_)


    x_mean, x_var = tf.placeholder(tf.float32, data.X.shape[1:]), tf.placeholder(tf.float32, data.X.shape[1:])
    p_mean, p_var = nn.get_mean_var(x_mean, x_var)

    nn.create_session()
    nn.fit(data, stopping_criterion=lambda i, _: i<51)

    emp_mean, emp_var = np.mean(data.X_val, axis=0), np.var(data.X_val, axis=0)

    # filters for feature sets; 1: feature is not known, 0: feature is known
    zero = np.zeros([28, 28])

    m_c = np.zeros([28, 28])
    m_c[9:19, 9:19] = 1

    s_c = np.zeros([28, 28])
    s_c[13:15, 13:15] = 1

    l = np.zeros([28, 28]) + 1

    m_left = np.zeros([28, 28])
    m_left[:10, 9:19] = 1

    for sample_id in range(10):
        utils.visualize(data.X_val[sample_id][None,...], utils.heatmap_original, "sample-{}.png".format(sample_id))

        for features, feature_name in zip([m_c, s_c, l, m_left], ["m_c", "s_c", "l", "m_left"]):
            plt.imshow(features)
            plt.savefig(feature_name+".png")
            features = np.reshape(features, emp_var.shape)
            mean, var = nn.sess.run([p_mean, p_var], feed_dict={
                x_mean: np.reshape((features*emp_mean) + (1-features)*data.X_val[sample_id], list(data.X.shape[1:])),
                x_var: np.reshape((features*emp_var), list(data.X.shape[1:]))
            })
            print("(sample ", sample_id, ")", feature_name, var/mean)

    nn.close_sess()





cnn = lambda : [Format(), NextConvolution([5, 5, 1, 32]), ReLU(), Pooling(),
                   NextConvolution([5, 5, 32, 64]), ReLU(), Pooling(),
                   NextLinear(1024), ReLU(),
                   NextLinear(10)]

fcn = lambda : [Format(), FirstLinear(784), ReLU(), NextLinear(10)]


var_relevance(cnn())

conservation_test(cnn(), "cnn", ["zbab"], "zab-ref", reference="val")
conservation_test(cnn(), "cnn", ["zbab"], "zbab", reference=None)
conservation_test(cnn(), "cnn", ["ab"], "ab-ref", reference="val")
conservation_test(fcn(), "fcn", ["zbab"], "zab-ref", reference="val")
conservation_test(fcn(), "fcn", ["zbab"], "zbab", reference=None)
conservation_test(fcn(), "fcn", ["ab"], "ab-ref", reference="val")