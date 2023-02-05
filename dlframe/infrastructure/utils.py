import matplotlib.pyplot as plt

def plot_hyper(value, loss, name, work_dir):
    plt.figure()
    plt.scatter(value, loss)
    plt.title(f'{name}')
    plt.savefig(f'{work_dir}/{name}.png')
    plt.close()
        
def plot_hyper_dict(dic, loss, work_dir):
    keys = dic.keys()
    for key in keys:
        plot_hyper(dic[key], loss, key, work_dir)
        
def dict_to_str(dic):
    lst = []
    for item in dic.items():
        lst.append(item[0] + str(item[1]))
        
    return '_'.join(lst)