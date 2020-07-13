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

def log_training_time(time_in_seconds):
    if (VERBOSE >= 1):
        print("[Tinynet] Finished training in {} seconds".format(time_in_seconds))

def output_intermediate_result(layername, output, type):
    if (VERBOSE >= 3):
        print('--- {} {} ---'.format(type, layername))
        print('> shape: '+str(output.shape))
        print(output)
        print('=== end of data ===')
