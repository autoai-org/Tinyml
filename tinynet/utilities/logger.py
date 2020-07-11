#coding:utf-8

VERBOSE = 1

def print_net_summary(layers):
    for each in layers:
        print(each.summary())

def log_trainining_progress(epoch,total_epochs, loss_sum, loss_mean):
    if (VERBOSE==1):
        print("[tinynet] epoch: {}/{}, loss(sum): {}, loss(mean): {}".format(epoch + 1, total_epochs, loss_sum, loss_mean))
    else:
        pass

def log_backward_gradient(layer_name, mean_gradient):
    if (VERBOSE>=2):
        print('backwarding {} gradient: {}'.format(layer_name, mean_gradient))
