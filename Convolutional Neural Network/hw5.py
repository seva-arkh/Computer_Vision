import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm


def get_mini_batch(im_train, label_train, batch_size):
    mini_batch_x = []
    mini_batch_y = []
    zero_x = [0 for x in range(196)]
    zero_y = [0 for x in range(10)]

    rand_ = np.random.permutation(im_train.shape[1])
    for i in range(0,math.ceil(im_train.shape[1]/batch_size)):
      
      l_x = []
      l_y = []

      if im_train.shape[1]//batch_size == i: 
          p = im_train.shape[1]%batch_size
          if not p:
              break
      else:
          p = batch_size

      for j in range(batch_size):
        if j<p:
            ind = rand_[i*batch_size +j]
            l_x.append(im_train[:, ind])

            l = [0 for x in range(10)]
            l[label_train[0][ind]] = 1
            l_y.append(l)
        else:
            l_x.append(zero_x)
            l_y.append(zero_y)

      l_x = np.array(l_x)
      l_y = np.array(l_y)

      mini_batch_x.append(l_x.T)
      mini_batch_y.append(l_y.T)

    mini_batch_x = np.asarray(mini_batch_x)
    mini_batch_y = np.array(mini_batch_y)
    
    return mini_batch_x, mini_batch_y


def fc(x, w, b):
    y = np.add(np.matmul(w, x), b)
    return y


def fc_backward(dl_dy, x, w, b):
    dl_dx = np.dot(dl_dy.T, w)
    dl_dw = np.dot(dl_dy, x.T)
    dl_db = dl_dy
    
    return dl_dx.T, dl_dw, dl_db


def relu(x):
    y = np.maximum(0, x)
    return y


def relu_backward(dl_dy, x, y):
    dl_dx = dl_dy * (np.where(x>0, 1, 0) )
    return dl_dx


def loss_cross_entropy_softmax(x, y):
    exps = np.exp(x)
    y_tilde = exps / np.sum(exps)
    dl_dx = y_tilde - y
    dl_dx.reshape(1,len(dl_dx))
    l = -1 * np.sum(y * np.log(y_tilde))
    return l, dl_dx



def conv(x, w_conv, b_conv):
    x_h, x_w, c_1 = x.shape
    w_h, w_w, _, c_2 = w_conv.shape
    x = np.pad(x,((1, 1),(1, 1), (0, 0)))
    y = np.zeros((x_h, x_w, c_2))

    for i in range(c_2):
        filter = w_conv[i, :, :]
        bias = b_conv[i]
        for j in range(x_h):
            for k in range(x_w):
                y[j,k,i] = np.sum(filter * x[j:j+w_h, k:k + w_w, :]) + bias
    return y


def conv_backward(dl_dy, x, w_conv, b_conv):
    x_h, x_w, c_1 = x.shape
    w_h, w_w, _, c_2 = w_conv.shape
    x = np.pad(x,((1, 1),(1, 1), (0, 0)))

    dl_dw = np.zeros(w_conv.shape)
    dl_db = np.zeros(b_conv.shape)

    for i in range(c_2):
        bias_grad = np.sum(dl_dy[:, :, i])
        dl_db[i] = bias_grad

        filter_grad = np.zeros((w_h, w_w, c_1))

        for j in range(x_h):
            for k in range(x_w):
                filter_grad += dl_dy[j, k, i] * x[j:j+w_h, k:k + w_w, :]

    return dl_dw, dl_db

def pool2x2(x):
    stride = 2
    x_h, x_w, c = x.shape
    h = x_h//2
    w = x_w//2
    y = np.zeros((x_h//2, x_w//2, c))
    for k in range(c):
        for i in range(h):
            for j in range(w):
                y[i, j , k] = np.max(x[i*stride:(i+1)*stride, j*stride:(j+1)*stride, k])

    return y

def pool2x2_backward(dl_dy, x):
    h, w, c = x.shape
    h_out, w_out = dl_dy.shape[:2]
    dl_dx = np.zeros_like(x)
    for i in range(h_out):
        for j in range(w_out):
            window = x[2*i:2*i+2, 2*j:2*j+2, :]
            m = np.max(window, axis=(0,1))
            for k in range(c):
                for p in range(2):
                    for q in range(2):
                        if window[p,q,k] == m[k]:
                            dl_dx[2*i+p, 2*j+q, k] += dl_dy[i,j,k]
    return dl_dx

def flattening(x):
    return x.reshape((-1, 1))


def flattening_backward(dl_dy, x):
    return dl_dy.reshape(x.shape)


def train_mlp(mini_batch_x, mini_batch_y, learning_rate=0.1, decay_rate=0.95, num_iters=10000):
    nh = 30

    # initialization network weights
    w1 = np.random.randn(nh, 196)  # (30, 196)
    b1 = np.zeros((nh, 1))  # (30, 1)
    w2 = np.random.randn(10,nh)  # (10, 30)
    b2 = np.zeros((10, 1))  # (10, 1)

    k = 0
    losses = np.zeros((num_iters, 1))
    for iter in tqdm(range(num_iters), desc='Training MLP'):
        if (iter + 1) % 1000 == 0:
            learning_rate = decay_rate * learning_rate
        dl_dw1_batch = np.zeros((nh, 196))
        dl_db1_batch = np.zeros((nh, 1))
        dl_dw2_batch = np.zeros((10, nh))
        dl_db2_batch = np.zeros((10, 1))
        batch_size = mini_batch_x[k].shape[1]
        ll = np.zeros(batch_size)
        for i in range(batch_size):
            x = mini_batch_x[k][:, [i]]
            y = mini_batch_y[k][:, [i]]

            # forward propagation
            h1 = fc(x, w1, b1)
            h2 = relu(h1)
            h3 = fc(h2, w2, b2)

            # loss computation (forward + backward)
            l, dl_dy = loss_cross_entropy_softmax(h3, y.astype(int))
            ll[i] = np.mean(l)

            # backward propagation
            dl_dh2, dl_dw2, dl_db2 = fc_backward(dl_dy, h2, w2, b2)
            dl_dh1 = relu_backward(dl_dh2, h1, h2)
            dl_dx, dl_dw1, dl_db1 = fc_backward(dl_dh1, x, w1, b1)

            # accumulate gradients
            dl_dw1_batch += dl_dw1
            dl_db1_batch += dl_db1
            dl_dw2_batch += dl_dw2
            dl_db2_batch += dl_db2

        losses[iter] = np.mean(ll)
        k = k + 1
        if k > len(mini_batch_x) - 1:
            k = 0

        # accumulate gradients
        w1 -= learning_rate * dl_dw1_batch / batch_size
        b1 -= learning_rate * dl_db1_batch / batch_size
        w2 -= learning_rate * dl_dw2_batch / batch_size
        b2 -= learning_rate * dl_db2_batch / batch_size

    return w1, b1, w2, b2, losses



def train_cnn(mini_batch_x, mini_batch_y, learning_rate=0.05, decay_rate=0.95, num_iters=10000):
    # initialization network weights
    w_conv = 0.1 * np.random.randn(3, 3, 1, 3)
    b_conv = np.zeros((3, 1))
    w_fc = 0.1 * np.random.randn(10, 147)
    b_fc = np.zeros((10, 1))

    k = 0
    losses = np.zeros((num_iters, 1))
    for iter in tqdm(range(num_iters), desc='Training CNN'):
        if (iter + 1) % 1000 == 0:
            learning_rate = decay_rate * learning_rate
            # print('iter {}/{}'.format(iter + 1, num_iters))

        dl_dw_conv_batch = np.zeros(w_conv.shape)
        dl_db_conv_batch = np.zeros(b_conv.shape)
        dl_dw_fc_batch = np.zeros(w_fc.shape)
        dl_db_fc_batch = np.zeros(b_fc.shape)
        batch_size = mini_batch_x[k].shape[1]
        ll = np.zeros((batch_size, 1))

        for i in range(batch_size):
            x = mini_batch_x[k][:, [i]].reshape((14, 14, 1))
            y = mini_batch_y[k][:, [i]]

            # forward propagation
            h1 = conv(x, w_conv, b_conv)  # (14, 14, 3)
            h2 = relu(h1)  # (14, 14, 3)
            h3 = pool2x2(h2)  # (7, 7, 3)
            h4 = flattening(h3)  # (147, 1)
            h5 = fc(h4, w_fc, b_fc)  # (10, 1)

            # loss computation (forward + backward)
            l, dl_dy = loss_cross_entropy_softmax(h5, y)
            ll[i] = l

            # backward propagation
            dl_dh4, dl_dw_fc, dl_db_fc = fc_backward(dl_dy, h4, w_fc, b_fc)  # (147, 1), (10, 147), (10, 1)
            dl_dh3 = flattening_backward(dl_dh4, h3)  # (7, 7, 3)
            dl_dh2 = pool2x2_backward(dl_dh3, h2)  # (14, 14, 3)
            dl_dh1 = relu_backward(dl_dh2, h1, h2)  # (14, 14, 3)
            dl_dw_conv, dl_db_conv = conv_backward(dl_dh1, x, w_conv, b_conv)  # (3, 3, 1, 3), (3, 1)

            # accumulate gradients
            dl_dw_conv_batch += dl_dw_conv
            dl_db_conv_batch += dl_db_conv
            dl_dw_fc_batch += dl_dw_fc
            dl_db_fc_batch += dl_db_fc

        losses[iter] = np.mean(ll)
        k = k + 1
        if k > len(mini_batch_x) - 1:
            k = 0

        # update network weights
        w_conv -= learning_rate * dl_dw_conv_batch / batch_size
        b_conv -= learning_rate * dl_db_conv_batch / batch_size
        w_fc -= learning_rate * dl_dw_fc_batch / batch_size
        b_fc -= learning_rate * dl_db_fc_batch / batch_size

    return w_conv, b_conv, w_fc, b_fc, losses


def visualize_training_progress(losses, num_batches):
    # losses - (n_iter, 1)
    num_iters = losses.shape[0]
    num_epochs = math.ceil(num_iters / num_batches)
    losses_epoch = np.zeros((num_epochs, 1))
    losses_epoch[:num_epochs-1, 0] = np.mean(
        np.reshape(losses[:(num_epochs - 1)*num_batches], (num_epochs - 1, num_batches)), axis=1)
    losses_epoch[num_epochs-1] = np.mean(losses[(num_epochs - 1)*num_batches:])

    fig, axs = plt.subplots(1, 2, constrained_layout=True)
    axs[0].plot(range(num_iters), losses), axs[0].set_title('Training loss w.r.t. iteration')
    axs[0].set_xlabel('Iteration'), axs[0].set_ylabel('Loss'), axs[0].set_ylim([0, 5])
    axs[1].plot(range(num_epochs), losses_epoch), axs[1].set_title('Training loss w.r.t. epoch')
    axs[1].set_xlabel('Epoch'), axs[1].set_ylabel('Loss'), axs[1].set_ylim([0, 5])
    fig.suptitle('MLP Training Loss', fontsize=16)
    plt.show()


def infer_mlp(x, w1, b1, w2, b2):
    # x - (m, 1)
    h1 = fc(x, w1, b1)
    h2 = relu(h1)
    h3 = fc(h2, w2, b2)
    y = np.argmax(h3)
    return y


def infer_cnn(x, w_conv, b_conv, w_fc, b_fc):
    # x - (H(14), W(14), C_in(1))
    h1 = conv(x, w_conv, b_conv)  # (14, 14, 3)
    h2 = relu(h1)  # (14, 14, 3)
    h3 = pool2x2(h2)  # (7, 7, 3)
    h4 = flattening(h3)  # (147, 1)
    h5 = fc(h4, w_fc, b_fc)  # (10, 1)
    y = np.argmax(h5)
    return y


def compute_confusion_matrix_and_accuracy(pred, label, n_classes):
    # pred, label - (n, 1)
    accuracy = np.sum(pred == label) / len(label)
    confusion = np.zeros((n_classes, n_classes))
    for j in range(n_classes):
        for i in range(n_classes):
            # ground true is j but predicted to be i
            confusion[i, j] = np.sum(np.logical_and(label == j, pred == i)) / label.shape[0]
    return confusion, accuracy


def visualize_confusion_matrix(confusion, accuracy, label_classes):
    plt.title("Accuracy = {:.3f}".format(accuracy))
    plt.imshow(confusion)
    ax, fig = plt.gca(), plt.gcf()
    plt.xticks(np.arange(len(label_classes)), label_classes)
    plt.yticks(np.arange(len(label_classes)), label_classes)
    ax.set_xticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.set_yticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    plt.show()


def get_MNIST_data(resource_dir):
    with open(resource_dir + 'mnist_train.npz', 'rb') as f:
        d = np.load(f)
        image_train, label_train = d['img'], d['label']  # (12k, 14, 14), (12k, 1)
    with open(resource_dir + 'mnist_test.npz', 'rb') as f:
        d = np.load(f)
        image_test, label_test = d['img'], d['label']  # (2k, 14, 14), (2k, 1)

    label_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    return image_train, label_train, image_test, label_test, label_classes


if __name__ == '__main__':
    work_dir = ''

    image_train, label_train, image_test, label_test, label_classes = get_MNIST_data(work_dir)
    image_train, image_test = image_train.reshape((-1, 196)).T / 255.0, image_test.reshape((-1, 196)).T / 255.0

    # # Part 1: Multi-layer Perceptron
    # # train
    # mini_batch_x, mini_batch_y = get_mini_batch(image_train, label_train.T, batch_size=32)
    # w1, b1, w2, b2, losses = train_mlp(mini_batch_x, mini_batch_y,
    #                                    learning_rate=0.1, decay_rate=0.9, num_iters=10000)
    # visualize_training_progress(losses, len(mini_batch_x))
    # np.savez(work_dir + 'mlp.npz', w1=w1, b1=b1, w2=w2, b2=b2)

    # # test
    # pred_test = np.zeros_like(label_test)
    # for i in range(image_test.shape[1]):
    #     pred_test[i, 0] = infer_mlp(image_test[:, i].reshape((-1, 1)), w1, b1, w2, b2)
    # confusion, accuracy = compute_confusion_matrix_and_accuracy(pred_test, label_test, len(label_classes))
    # visualize_confusion_matrix(confusion, accuracy, label_classes)

    # Part 2: Convolutional Neural Network
    # train
    np.random.seed(0)
    mini_batch_x, mini_batch_y = get_mini_batch(image_train, label_train.T, batch_size=32)
    w_conv, b_conv, w_fc, b_fc, losses = train_cnn(mini_batch_x, mini_batch_y,
                                                   learning_rate=0.1, decay_rate=0.9, num_iters=1000)
    visualize_training_progress(losses, len(mini_batch_x))
    np.savez(work_dir + 'cnn.npz', w_conv=w_conv, b_conv=b_conv, w_fc=w_fc, b_fc=b_fc)

    # test
    pred_test = np.zeros_like(label_test)
    for i in range(image_test.shape[1]):
        pred_test[i, 0] = infer_cnn(image_test[:, i].reshape((14, 14, 1)), w_conv, b_conv, w_fc, b_fc)
    confusion, accuracy = compute_confusion_matrix_and_accuracy(pred_test, label_test, len(label_classes))
    visualize_confusion_matrix(confusion, accuracy, label_classes)
