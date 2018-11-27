#!/bin/python3

# Checks prediction function against actual data
# Simplest version of checking -- not super accurate
def simple_check(pred_fun, real_data):
    gen_data = []
    for i in range(len(real_data)):
        gen_data.append(pred_fun(i))
    gen_mat = np.array(gen_data)
    real_mat = np.array(real_data)
    return np.average(gen_mat - real_mat)

