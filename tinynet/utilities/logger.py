#coding:utf-8

VERBOSE = 1

def print_net_summary(layers):
    for each in layers:
        print(each.summary())

def log_trainining_progress(epoch,total_epochs, loss_sum, loss_mean):
    if (VERBOSE>=1):
        print("[Tinynet] epoch: {}/{}, loss(sum): {}, loss(mean): {}".format(epoch + 1, total_epochs, loss_sum, loss_mean))
    else:
        pass

def log_backward_gradient(layer_name, mean_gradient):
    if (VERBOSE>=2):
        print('backwarding {} gradient: {}'.format(layer_name, mean_gradient))

def log_training_time(time_in_seconds):
    if (VERBOSE >= 1):
        print("[Tinynet] Finished training in {} seconds".format(time_in_seconds))