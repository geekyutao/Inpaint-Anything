import torch


def merge_template_search(inp_list, return_search=False, return_template=False):
    """NOTICE: search region related features must be in the last place"""
    seq_dict = {"feat": torch.cat([x["feat"] for x in inp_list], dim=0),
                "mask": torch.cat([x["mask"] for x in inp_list], dim=1),
                "pos": torch.cat([x["pos"] for x in inp_list], dim=0)}
    if return_search:
        x = inp_list[-1]
        seq_dict.update({"feat_x": x["feat"], "mask_x": x["mask"], "pos_x": x["pos"]})
    if return_template:
        z = inp_list[0]
        seq_dict.update({"feat_z": z["feat"], "mask_z": z["mask"], "pos_z": z["pos"]})
    return seq_dict


def get_qkv(inp_list):
    """The 1st element of the inp_list is about the template,
    the 2nd (the last) element is about the search region"""
    dict_x = inp_list[-1]
    dict_c = {"feat": torch.cat([x["feat"] for x in inp_list], dim=0),
              "mask": torch.cat([x["mask"] for x in inp_list], dim=1),
              "pos": torch.cat([x["pos"] for x in inp_list], dim=0)}  # concatenated dict
    q = dict_x["feat"] + dict_x["pos"]
    k = dict_c["feat"] + dict_c["pos"]
    v = dict_c["feat"]
    key_padding_mask = dict_c["mask"]
    return q, k, v, key_padding_mask
