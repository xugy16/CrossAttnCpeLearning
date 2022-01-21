import torch


def cal_mean_glovec(phase_cpt, glove_obj):
    """ mean the glovecs """
    word_list = phase_cpt.split(" ")
    glovec_list = [glove_obj[word] for word in word_list]
    phase_glovec = torch.stack(glovec_list, dim=0).mean(dim=0)
    return phase_glovec


def load_cpt_glove(cpt_list, glove_obj):
    """
    costom import embeddings;
    1. for words not exist;
    2. we need to modify the glovec;
    """
    # 1: lower case;
    cpt_list = [cpt.lower() for cpt in cpt_list]

    # 2: for zappos (should account for everything)
    custom_map = {
        'Faux.Fur': 'faux fur',
        'Faux.Leather': 'faux leather',
        'Full.grain.leather': 'grain',
        'Hair.Calf': 'hair',
        'Patent.Leather': 'patent',
        'Nubuck': 'cattle',

        'Boots.Ankle': 'boots',
        'Boots.Knee.High': 'knee-high',
        'Boots.Mid-Calf': 'midcalf',
        'Shoes.Boat.Shoes': 'shoes',
        'Shoes.Clogs.and.Mules': 'clogs',
        'Shoes.Flats': 'flats',
        'Shoes.Heels': 'heels',
        'Shoes.Loafers': 'loafers',
        'Shoes.Oxfords': 'oxfords',
        'Shoes.Sneakers.and.Athletic.Shoes': 'sneakers'}

    # 2: lower-case keys
    custom_map = dict((k.lower(), v) for k, v in custom_map.items())

    # 3:  check whether we have the same attrs or objs
    uni_cpt_list = [cpt if cpt not in custom_map else custom_map[cpt] for cpt in cpt_list]

    # 4: returned resutls;
    embeds = [glove_obj[word] if word not in custom_map else cal_mean_glovec(custom_map[word], glove_obj) for word in cpt_list]
    embeds = torch.stack(embeds)

    # return the embedding matrix;
    return embeds, uni_cpt_list