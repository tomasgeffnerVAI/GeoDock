import os
from tqdm import tqdm


def get_list(root):
    good_list = []
    complexes_orig = [f.path for f in os.scandir(root) if f.is_dir()]
    for p in complexes_orig:
        files_complex = [f for f in os.listdir(p) if os.path.isfile(os.path.join(p, f))]
        n_pt_comp = sum([".pt" == f[-3:] for f in files_complex])
        
        p_holo = os.path.join(p, "holo")
        files_holo = [f for f in os.listdir(p_holo) if os.path.isfile(os.path.join(p_holo, f))]
        n_pt_holo = sum([".pt" == f[-3:] for f in files_holo])

        if n_pt_comp == 1 and n_pt_holo == 2:
            good_list.append(p.split("/")[-1])
        else:
            print("No")
    return good_list


def save_list(clean_list, mode, root):
    p = os.path.join(root, f"processed_{mode}.txt")
    with open(p, 'w') as f:
        for item in clean_list:
            f.write("%s\n" % item)


root = "/home/tomasgeffner/pinder_copy/"

root_train = "/home/tomasgeffner/pinder_copy/splits_v2/train/"
clean_list_train = get_list(root_train)
save_list(clean_list_train, "train", root)

root_test = "/home/tomasgeffner/pinder_copy/splits_v2/test/"
clean_list_test = get_list(root_test)
save_list(clean_list_test, "test", root)
