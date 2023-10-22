import torch
import random


def get_cuts(L1, L2):
    c1_start, c1_end = None, None
    c2_start, c2_end = None, None
    
    crop_size = 500
    if L1 + L2 > crop_size:
        crop_size_per_chain = crop_size // 2
        
        if L1 > L2:
            
            if L2 < crop_size_per_chain:
                crop_len = crop_size - L2
                n = random.randint(0, L1 - crop_len)
                c1_start, c1_end = n, n + crop_len
            else:
                n1 = random.randint(0, L1 - crop_size_per_chain)
                n2 = random.randint(0, L2 - crop_size_per_chain)
                c1_start, c1_end = n1, n1 + crop_size_per_chain
                c2_start, c2_end = n2, n2 + crop_size_per_chain

        elif L2 > L1:
            
            if L1 < crop_size_per_chain:
                crop_len = crop_size - L1
                n = random.randint(0, L2 - crop_len)
                c2_start, c2_end = n, n + crop_len
            else:
                n1 = random.randint(0, L1 - crop_size_per_chain)
                n2 = random.randint(0, L2 - crop_size_per_chain)
                c1_start, c1_end = n1, n1 + crop_size_per_chain
                c2_start, c2_end = n2, n2 + crop_size_per_chain

        else:
            n = random.randint(0, L1 - crop_size_per_chain)  # This is the original, should be done separately for both?
            c1_start, c1_end = n, n + crop_size_per_chain
            c2_start, c2_end = n, n + crop_size_per_chain

    return (c1_start, c1_end), (c2_start, c2_end)


def crop_features(pemb1, pemb2, pair, posit):
    """
    pemb1 [1, L1, 1280]
    pemb2 [1, L2, 1280]
    pair [1, L1+L2, L1+L2, 1280]
    posit [1, L1+L2, L1+L2, 1280]
    """
    L1 = pemb1.shape[1]
    L2 = pemb2.shape[1]
    
    (c1_start, c1_end), (c2_start, c2_end) = get_cuts(L1, L2)
    
    # Crop embedding 1
    if c1_start is not None:
        pemb1 = pemb1[:, c1_start : c1_end, :]
    
    # Crop embedding 2
    if c2_start is not None:
        pemb2 = pemb2[:, c2_start : c2_end, :]
    
    # Create mask for cropping
    if c1_start is None:
        mask1 = torch.ones(L1, dtype=bool)
    else:
        mask1 = torch.zeros(L1, dtype=bool)
        mask1[c1_start : c1_end] = 1
    
    if c2_start is None:
        mask2 = torch.ones(L2, dtype=bool)
    else:
        mask2 = torch.zeros(L2, dtype=bool)
        mask2[c2_start : c2_end] = 1
    
    mask = torch.cat([mask1, mask2])
    
    # Crop
    pair = pair[:, mask, :, :][:, :, mask, :]
    posit = posit[:, mask, :, :][:, :, mask, :]

    return pemb1, pemb2, pair, posit


def crop_targets(label_coords, label_rotat, label_trans, L1, L2):
    # label_coords [1, L1 + L2, 3, 3]
    # label_rotat [1, L1 + L2, 3, 3]
    # label_trans [1, L1 + L2, 3]
    (c1_start, c1_end), (c2_start, c2_end) = get_cuts(L1, L2)

    # Create mask for cropping
    if c1_start is None:
        mask1 = torch.ones(L1, dtype=bool)
    else:
        mask1 = torch.zeros(L1, dtype=bool)
        mask1[c1_start : c1_end] = 1
    
    if c2_start is None:
        mask2 = torch.ones(L2, dtype=bool)
    else:
        mask2 = torch.zeros(L2, dtype=bool)
        mask2[c2_start : c2_end] = 1
    
    mask = torch.cat([mask1, mask2])

    # print(label_coords.shape, label_rotat.shape, label_trans.shape, mask.shape)

    label_coords = label_coords[:, mask, :, :]
    label_rotat = label_rotat[:, mask, :, :]
    label_trans = label_trans[:, mask, :]

    if c1_start is None:
        sep = L1
    else:
        sep = c1_end - c1_start
    
    return label_coords, label_rotat, label_trans, sep



