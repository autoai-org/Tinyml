from tinynet.core import Backend as np


def evaluate_classification_accuracy(model, epoch, data, label):
    correct = 0
    batch_size = 1
    for i in range(0,len(data), batch_size):
        y_predict = model.predict(data[i:i+batch_size])
        correct += np.argmax(y_predict, axis=1) == label[i:i+batch_size]
    print("Epoch: {}, Acc: {}%".format(epoch, 100 * correct/len(data)))

def save_model(model, epoch,data,label):
    model.export('./model-{}.tnn'.format(epoch))
