from tinynet.core import Backend as np


def evaluate_classification_accuracy(model, epoch, data, label):
    y_predict = model.predict(data, batch_size=1)
    acc = np.mean(
        np.equal(np.argmax(y_predict, axis=-1), np.argmax(label, axis=-1)))
    print("Epoch: {}, Acc: {}".format(epoch, acc))

def save_model(model, epoch,data,label):
    model.export('./model-{}.tnn'.format(epoch))
