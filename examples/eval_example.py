from lrp import *
from lrp.data import Data
from lrp import read_mnist
import os
from lrp.evaluate_rule import HeatmapEval
from lrp import utils
import os
from lrp.generator import *

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

    rules = ["simple",
             ("simple", {"reference": data.X_val}),
             "deeptaylor",
             ("deeptaylor", {"reference": data.X_val}),
             "zbab",
             ("zbab", {"reference": data.X_val})]

    evaluator = HeatmapEval(nns, ["fcn", "cnn"],
                            rules, ["simple", "simple-ref", "deeptaylor", "deeptaylor-ref", "zbab", "zbab-ref"],
                            data)

    with tf.Session() as sess:
        for nn in nns:
            nn.set_session(sess)
        sess.run(tf.global_variables_initializer())

        os.makedirs("evaluate/heatmaps", exist_ok=True)

        for nn_name, nn in zip(["fcn", "cnn"], nns):
            print("Evaluate", nn_name)
            nn.fit(data, stopping_criterion=lambda i, _: i<1500)

            for rule_name, H in zip(evaluator.rule_names, evaluator.heatmaps[nn]):
                print("Evaluate", rule_name)
                heatmaps, prediction = sess.run([H, nn.y], feed_dict={data.X: data.X_test[:16], data.y_: data.y_test[:16]})
                utils.visualize(heatmaps, utils.heatmap_original, "evaluate/heatmaps/{}-{}.png".format(
                    nn_name, rule_name))
                print("Visualized heatmaps")

        print("--------- COMPARE -----------------")
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


#eval_example()

def generator_example():
    mnist = read_mnist.read_data_sets("{}/datasets/mnist".format(os.environ["TF_PROJECTS"]), one_hot=True)
    X_train = mnist.train.images[:100]
    X_test = mnist.train.images[100:110]
    generator = GaussianGenerator(X_train)

    s_c = np.ones([28, 28])
    s_c[13:15, 13:13] = 0

    m_c = np.ones([28, 28])
    m_c[11:17, 11:17] = 0
    m_c = m_c.flatten()

    m_l = np.ones([28, 28])
    m_l[18:14, 11:17] = 0
    m_l = m_l.flatten()

    for image_id, image_ in enumerate(X_test):
        # show image with all filters + relevance
        for filter_id, filter in enumerate([s_c, m_c, m_l]):
            image = np.array(image_)
            filtered_image = np.zeros([784, 3])
            filtered_image[:,0] = image*0.8
            filtered_image[:,1] = image*0.8
            filtered_image[:,2] = image*0.8
            filtered_image[:,0][filter==0] += 0.2
            filtered_image[:,1][filter!=0] += 0.2
            filtered_image =  np.reshape(filtered_image, [28, 28, 3])
            plt.subplot(3, 3, filter_id)
            plt.imshow(filtered_image)
            plt.title(["s_c", "m_c", "m_l"][filter_id])

        for g_id, generator in [GaussianGenerator(X_train), StupidGenerator(X_train)(X_test, s_c)]:
            for filter_id, filter in enumerate([s_c, m_c, m_l]):
                X_gen = generator(X_test, s_c)

        plt.savefig("{}.png".format(image_id))
        plt.clf()

    """

    s_c = np.reshape(s_c, [np.prod(s_c.shape)])
    X_gen = generator(X_test, s_c)
    utils.visualize(X_gen, utils.graymap, "gaussgen")

    X_gen = StupidGenerator(X_train)(X_test, s_c)
    utils.visualize(X_gen, utils.graymap, "stupidgen")
    """


    

generator_example()

"""

cnn = lambda : [Format(), NextConvolution([5, 5, 1, 32]), ReLU(), Pooling(),
                   NextConvolution([5, 5, 32, 64]), ReLU(), Pooling(),
                   NextLinear(1024), ReLU(),
                   NextLinear(10)]

fcn = lambda : [Format(), FirstLinear(784), ReLU(), NextLinear(10)]




#var_relevance(cnn())

conservation_test(cnn(), "cnn", ["zbab"], "zab-ref", reference="val")
conservation_test(cnn(), "cnn", ["zbab"], "zbab", reference=None)
conservation_test(cnn(), "cnn", ["ab"], "ab-ref", reference="val")
conservation_test(fcn(), "fcn", ["zbab"], "zab-ref", reference="val")
conservation_test(fcn(), "fcn", ["zbab"], "zbab", reference=None)
conservation_test(fcn(), "fcn", ["ab"], "ab-ref", reference="val")
"""

