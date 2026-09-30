"""
This file implements our adaptive attacks used for evaluation
"""
from dataclasses import dataclass, asdict
import torch
from .vlm import VLM, VLMName
from .diffpure import purify
from .utils import get_memory_consumption
import gc
import string
import numpy as np
import time

def project(x, y, eps=1e-12):
    """
    Project x onto y.
    x, y: tensors with the same shape.
    """
    coeff = (x * y).sum() / (y.square().sum() + eps)
    return coeff * y

#DETECTION THRESHOLDS DERIVED FROM VQAv2 Training Set (AdverStep self-targeted LR ablation)
det_thresh_dict_glr = {
    1:{
    VLMName.QWEN_2p5_VL_3B:{

        'gait_bounds': {'loss_lo1': 0.011474609375, 'loss_hi1': 1.734375,
        'norm_lo1': 0.0014281999319791794, 'norm_hi1': 8.242098808288574,
        'cosim_lo1': -0.9882821440696716, 'cosim_hi1': 0.998663067817688,
        'mloss_lo1': 0.0191650390625, 'mloss_hi1': 2.546875,
        'ldif_lo1': -0.1337890625, 'ldif_hi1': 1.4375},

        'gait_bounds_percentiles': {'loss_lo1_1': 0.05606689453125, 'loss_hi1_1': 1.4845312499999999, 'loss_lo1_2': 0.0688427734375, 'loss_hi1_2': 1.445390625, 'loss_lo1_5': 0.0922607421875, 'loss_hi1_5': 1.3751953124999998,
        'loss_lo1_10': 0.125927734375, 'loss_hi1_10': 1.3359375, 'loss_lo1_15': 0.15625, 'loss_hi1_15': 1.2974609375000004, 'loss_lo1_20': 0.1826171875, 'loss_hi1_20': 1.265625,
        'ldif_lo1_1': -0.0918017578125, 'ldif_hi1_1': 0.8985937499999999, 'ldif_lo1_2': -0.075205078125, 'ldif_hi1_2': 0.7110156249999999, 'ldif_lo1_5': -0.054736328125, 'ldif_hi1_5': 0.515625,
        'ldif_lo1_10': -0.0390869140625, 'ldif_hi1_10': 0.4181640624999998, 'ldif_lo1_15': -0.030346679687500003, 'ldif_hi1_15': 0.3595214843750001, 'ldif_lo1_20': -0.0234375, 'ldif_hi1_20': 0.3203125}}
},
    4:{
    VLMName.QWEN_2p5_VL_3B:{
        'gait_bounds': {'loss_lo1': 0.011474609375, 'loss_hi1': 1.734375,
        'norm_lo1': 0.0014281999319791794, 'norm_hi1': 8.242098808288574,
        'cosim_lo1': -0.8006856441497803, 'cosim_hi1': 0.9324864149093628,
        'mloss_lo1': 0.01318359375, 'mloss_hi1': 2.421875,
        'ldif_lo1': -0.1767578125, 'ldif_hi1': 1.578125},

        'gait_bounds_percentiles': {'loss_lo1_1': 0.05606689453125, 'loss_hi1_1': 1.4845312499999999, 'loss_lo1_2': 0.0688427734375, 'loss_hi1_2': 1.445390625, 'loss_lo1_5': 0.0922607421875, 'loss_hi1_5': 1.3751953124999998,
        'loss_lo1_10': 0.125927734375, 'loss_hi1_10': 1.3359375, 'loss_lo1_15': 0.15625, 'loss_hi1_15': 1.2974609375000004, 'loss_lo1_20': 0.1826171875, 'loss_hi1_20': 1.265625,
        'ldif_lo1_1': -0.0937646484375, 'ldif_hi1_1': 0.9455468749999998, 'ldif_lo1_2': -0.080078125, 'ldif_hi1_2': 0.7110156249999999, 'ldif_lo1_5': -0.057666015625, 'ldif_hi1_5': 0.5626953124999998,
        'ldif_lo1_10': -0.0346923828125, 'ldif_hi1_10': 0.4765625, 'ldif_lo1_15': -0.025454711914062504, 'ldif_hi1_15': 0.390625, 'ldif_lo1_20': -0.015625, 'ldif_hi1_20': 0.33984375}}
},
    16:{
    VLMName.QWEN_2p5_VL_3B:{
        'gait_bounds': {'loss_lo1': 0.011474609375, 'loss_hi1': 1.734375,
        'norm_lo1': 0.0014281999319791794, 'norm_hi1': 8.242098808288574,
        'cosim_lo1': -0.7682589292526245, 'cosim_hi1': 0.7338777780532837,
        'mloss_lo1': 0.0157470703125, 'mloss_hi1': 2.515625,
        'ldif_lo1': -0.1533203125, 'ldif_hi1': 1.7109375},

        'gait_bounds_percentiles': {'loss_lo1_1': 0.05606689453125, 'loss_hi1_1': 1.4845312499999999, 'loss_lo1_2': 0.0688427734375, 'loss_hi1_2': 1.445390625, 'loss_lo1_5': 0.0922607421875, 'loss_hi1_5': 1.3751953124999998,
        'loss_lo1_10': 0.125927734375, 'loss_hi1_10': 1.3359375, 'loss_lo1_15': 0.15625, 'loss_hi1_15': 1.2974609375000004, 'loss_lo1_20': 0.1826171875, 'loss_hi1_20': 1.265625,
        'ldif_lo1_1': -0.0966943359375, 'ldif_hi1_1': 0.9226953124999993, 'ldif_lo1_2': -0.077158203125, 'ldif_hi1_2': 0.765625, 'ldif_lo1_5': -0.056640625, 'ldif_hi1_5': 0.5546875,
        'ldif_lo1_10': -0.0439453125, 'ldif_hi1_10': 0.4765625, 'ldif_lo1_15': -0.026440429687500003, 'ldif_hi1_15': 0.3987304687500002, 'ldif_lo1_20': -0.015722656249999994, 'ldif_hi1_20': 0.34375}}
},
    64:{
    VLMName.QWEN_2p5_VL_3B:{
        'gait_bounds': {'loss_lo1': 0.011474609375, 'loss_hi1': 1.734375,
        'norm_lo1': 0.0014281999319791794, 'norm_hi1': 8.242098808288574,
        'cosim_lo1': -0.5524802207946777, 'cosim_hi1': 0.738776445388794,
        'mloss_lo1': 0.01708984375, 'mloss_hi1': 2.65625,
        'ldif_lo1': -0.158203125, 'ldif_hi1': 2.453125},

        'gait_bounds_percentiles': {'loss_lo1_1': 0.05606689453125, 'loss_hi1_1': 1.4845312499999999, 'loss_lo1_2': 0.0688427734375, 'loss_hi1_2': 1.445390625, 'loss_lo1_5': 0.0922607421875, 'loss_hi1_5': 1.3751953124999998,
        'loss_lo1_10': 0.125927734375, 'loss_hi1_10': 1.3359375, 'loss_lo1_15': 0.15625, 'loss_hi1_15': 1.2974609375000004, 'loss_lo1_20': 0.1826171875, 'loss_hi1_20': 1.265625,
        'ldif_lo1_1': -0.108427734375, 'ldif_hi1_1': 0.9457812499999996, 'ldif_lo1_2': -0.076171875, 'ldif_hi1_2': 0.6955468749999998, 'ldif_lo1_5': -0.0517578125, 'ldif_hi1_5': 0.5628906249999996,
        'ldif_lo1_10': -0.0390625, 'ldif_hi1_10': 0.4572265624999998, 'ldif_lo1_15': -0.02734375, 'ldif_hi1_15': 0.3831054687500002, 'ldif_lo1_20': -0.0197998046875, 'ldif_hi1_20': 0.34375}}
},
    }

#DETECTION THRESHOLDS DERIVED FROM VQAv2 Training Set
det_thresh_dict = {
    0.05:{
    VLMName.SMOLVLM_1_256M:{
        "gait":{0:[1.0390625, 4.59375],1:[1.0859375, 3.859375],2:[1.1246875, 3.549374999999941],5:[1.203125, 3.265625],10:[1.265625, 2.7195312499999957],20:[1.3671875,2.3125]},
        "gait_bounds":{"ldif_lo":-0.46875, "mloss_lo":1.59375, "mloss_hi":8.125, "norm_lo":0.002085372107103467, "norm_hi":0.14538584649562836, "cosim_lo":0.020627484},
        #"pip":{0: ,1: , 2: , 5:, 10:, 20:},
        "pip_occ":{0: 95.59430059124098, 1:63.76638169129412 , 2:57.8876852608649 , 5:41.4497052846029, 10:31.920550641703908, 20:20.144665950240416},
        "mc":{0:0.333598140084933, 1:0.4601206123828888, 2:0.4788713645935059, 5:0.5419402688741684, 10:0.5831245899200439, 20:0.6514887332916259}
        },

      VLMName.SMOLVLM_1_500M:{
        "gait":{0:[0.87109375, 3.4921875 ],1:[1.0078125, 2.703125],2:[1.038984375, 2.227656249999974],5:[1.1171875, 2.125],10:[1.1875, 1.8753906249999979],20:[1.2734375,1.6328125]},
        "gait_bounds":{"ldif_lo":-0.015625, "mloss_lo":1.4375, "mloss_hi":5.4375, "norm_lo":0.001576848328113556, "norm_hi":0.05825292319059372, "cosim_lo":-0.74652416},
        #"pip":{0: ,1: , 2: , 5:, 10:, 20:},
        "pip_occ":{0: 133.28019770618985, 1:72.63934315026646 , 2:59.95085215437579 , 5:38.39318047275097, 10:25.475288747585406, 20:11.97756719032835},#MUST UPDATE
        "mc":{0:0.23525763481205686, 1:0.48910929083824156, 2:0.5091766619682312, 5:0.5699569702148437, 10:0.6152542352676392, 20:0.66209796667099}
        },

      VLMName.SMOLVLM_1_2B:{
        'gait_bounds': {'loss_lo1': 1.046875, 'loss_hi1': 6.0, 'loss_lo2': 1.0, 'loss_hi2': 5.78125, 'loss_lo3': 0.953125, 'loss_hi3': 5.6875, 'loss_lo4': 0.93359375, 'loss_hi4': 5.5, 'loss_lo5': 0.89453125, 'loss_hi5': 5.40625,
        'norm_lo1': 0.0035220442805439234, 'norm_hi1': 0.7069154977798462, 'norm_lo2': 0.003079258603975177, 'norm_hi2': 0.6655638217926025, 'norm_lo3': 0.00294510368257761, 'norm_hi3': 0.7238388061523438, 'norm_lo4': 0.002579149790108204, 'norm_hi4': 0.1873350441455841, 'norm_lo5': 0.0023394692689180374, 'norm_hi5': 0.7220680713653564,
        'cosim_lo1': -0.7636103630065918, 'cosim_hi1': 0.9514346718788147, 'cosim_lo2': -0.781453013420105, 'cosim_hi2': 0.9716395735740662, 'cosim_lo3': -0.8235094547271729, 'cosim_hi3': 0.9562312364578247, 'cosim_lo4': -0.8348919749259949, 'cosim_hi4': 0.9828013181686401, 'cosim_lo5': -0.8508477807044983, 'cosim_hi5': 0.9647858142852783,
        'mloss_lo1': 1.28125, 'mloss_hi1': 7.375, 'mloss_lo2': 1.3046875, 'mloss_hi2': 7.625, 'mloss_lo3': 1.2890625, 'mloss_hi3': 7.875, 'mloss_lo4': 1.3125, 'mloss_hi4': 7.875, 'mloss_lo5': 1.265625, 'mloss_hi5': 7.78125,
        'ldif_lo1': -0.390625, 'ldif_hi1': 4.1875, 'ldif_lo2': -0.25, 'ldif_hi2': 4.6875, 'ldif_lo3': -0.15625, 'ldif_hi3': 5.125, 'ldif_lo4': -0.125, 'ldif_hi4': 5.1875, 'ldif_lo5': -0.1484375, 'ldif_hi5': 5.21875},
        'gait_bounds_percentiles': {'loss_lo1_1': 1.0390234375, 'loss_hi1_1': 4.7190625, 'loss_lo1_2': 1.093515625, 'loss_hi1_2': 4.127812499999997, 'loss_lo1_5': 1.2265625, 'loss_hi1_5': 3.454296874999999, 'loss_lo1_10': 1.327734375, 'loss_hi1_10': 2.96875, 'loss_lo1_20': 1.4609375, 'loss_hi1_20': 2.6578125000000004,
        'ldif_lo1_1': -0.0234375, 'ldif_hi1_1': 2.6265624999999986, 'ldif_lo1_2': 0.007734375000000002, 'ldif_hi1_2': 2.3285937499999996, 'ldif_lo1_5': 0.06230468750000001, 'ldif_hi1_5': 1.7503906249999996, 'ldif_lo1_10': 0.09375, 'ldif_hi1_10': 1.3988281249999996, 'ldif_lo1_20': 0.1484375, 'ldif_hi1_20': 1.1179687500000002},
        'pip': {0: 0.6371895770481049, 1: -0.6459773253668251, 2: -0.751148333734877, 5: -0.9996182914919132, 10: -0.9997461742545448, 20: -1.000143484259567},
        'ns': {0: -33.5, 1: -37.2475, 2: -38.0, 5: -39.75, 10: -41.224999999999994, 20: -42.25},
        'mc': {0: 0.3143019676208496, 1: 0.36999326169490815, 2: 0.3974292266368866, 5: 0.4424226820468903, 10: 0.4919244468212128, 20: 0.542752480506897},
        },

    VLMName.QWEN_2p5_VL_3B:{
        'gait_bounds': {'loss_lo1': 0.011474609375, 'loss_hi1': 1.734375, 'loss_lo2': 0.009033203125, 'loss_hi2': 1.75, 'loss_lo3': 0.00732421875, 'loss_hi3': 1.671875, 'loss_lo4': 0.0068359375, 'loss_hi4': 1.6484375, 'loss_lo5': 0.0079345703125, 'loss_hi5': 1.6484375,
        'norm_lo1': 0.0014281999319791794, 'norm_hi1': 8.242098808288574, 'norm_lo2': 0.0018891672370955348, 'norm_hi2': 3.472264289855957, 'norm_lo3': 0.001064190175384283, 'norm_hi3': 4.147993087768555, 'norm_lo4': 0.0010057855397462845, 'norm_hi4': 2.8153858184814453, 'norm_lo5': 0.001086949952878058, 'norm_hi5': 1.471091866493225,
        'cosim_lo1': -0.8684901595115662, 'cosim_hi1': 0.7603474855422974, 'cosim_lo2': -0.9425919055938721, 'cosim_hi2': 0.9121933579444885, 'cosim_lo3': -0.952944815158844, 'cosim_hi3': 0.8764663338661194, 'cosim_lo4': -0.8742227554321289, 'cosim_hi4': 0.7891747951507568, 'cosim_lo5': -0.9456203579902649, 'cosim_hi5': 0.8510007262229919,
        'mloss_lo1': 0.0120849609375, 'mloss_hi1': 2.40625, 'mloss_lo2': 0.0145263671875, 'mloss_hi2': 2.40625, 'mloss_lo3': 0.0125732421875, 'mloss_hi3': 2.421875, 'mloss_lo4': 0.0189208984375, 'mloss_hi4': 2.484375, 'mloss_lo5': 0.016357421875, 'mloss_hi5': 2.640625,
        'ldif_lo1': -0.140625, 'ldif_hi1': 2.140625, 'ldif_lo2': -0.12060546875, 'ldif_hi2': 2.21875, 'ldif_lo3': -0.0908203125, 'ldif_hi3': 2.046875, 'ldif_lo4': -0.1171875, 'ldif_hi4': 2.09375, 'ldif_lo5': -0.072265625, 'ldif_hi5': 1.7890625},

        'gait_bounds_percentiles': {'loss_lo1_1': 0.0302313232421875, 'loss_hi1_1': 1.453203125, 'loss_lo1_2': 0.0410009765625, 'loss_hi1_2': 1.406328125, 'loss_lo1_5': 0.0658935546875, 'loss_hi1_5': 1.34375,
                                    'loss_lo1_10': 0.089306640625, 'loss_hi1_10': 1.28125, 'loss_lo1_15': 0.11376953125, 'loss_hi1_15': 1.25, 'loss_lo1_20': 0.13466796875, 'loss_hi1_20': 1.2109375,
                                    'ldif_lo1_1': -0.03226806640625, 'ldif_hi1_1': 1.1487109374999998, 'ldif_lo1_2': -0.0283203125, 'ldif_hi1_2': 0.8437890625, 'ldif_lo1_5': -0.01563720703125, 'ldif_hi1_5': 0.6725585937499994,
                                    'ldif_lo1_10': -4.8828124999997224e-05, 'ldif_hi1_10': 0.5546875, 'ldif_lo1_15': 0.01171875, 'ldif_hi1_15': 0.4921875, 'ldif_lo1_20': 0.018505859375000003, 'ldif_hi1_20': 0.43125000000000036},

        'pip': {0: 1.0279900059843876, 1: -0.1902945701021105, 2: -0.6528739637833839, 5: -0.999638271331301, 10: -0.999792379768563, 15: -1.0000260241195966, 20: -1.0002664754975903},
        'ns': {0: -132.0, 1: -158.98000000000002, 2: -161.0, 5: -166.0, 10: -169.0, 15: -171.85000000000002, 20: -173.0},
        'mc': {0: 0.2687864303588867, 1: 0.36935904264450076, 2: 0.40826908111572263, 5: 0.4627618551254272, 10: 0.5008865475654602, 15:0.52, 20: 0.5544941067695618},#15perc is fake

        'nips_gait_bounds': {'loss_lo1': 0.265625, 'loss_hi1': 1.6953125, 'norm_lo1': 0.0039877984672784805, 'norm_hi1': 1.5066347122192383, 'cosim_lo1': -0.5576407313346863, 'cosim_hi1': 0.8331049084663391, 'mloss_lo1': 0.2578125, 'mloss_hi1': 2.234375, 'ldif_lo1': -0.03125, 'ldif_hi1': 0.90625},
        'nips_gait_bounds_percentiles': {'loss_lo1_0.5': 0.280126953125, 'loss_hi1_0.5': 1.6895117187500004, 'loss_lo1_1': 0.29462890625, 'loss_hi1_1': 1.6837109375, 'loss_lo1_2': 0.3236328125, 'loss_hi1_2': 1.6721093750000002, 'loss_lo1_10': 0.40283203125, 'loss_hi1_10': 1.4765625, 'ldif_lo1_0.5': -0.030283203125, 'ldif_hi1_0.5': 0.9043164062500001,
                                        'ldif_lo1_1': -0.02931640625, 'ldif_hi1_1': 0.9023828125, 'ldif_lo1_2': -0.0273828125, 'ldif_hi1_2': 0.898515625, 'ldif_lo1_10': -0.0078125, 'ldif_hi1_10': 0.4490234374999998},
        'coco_gait_bounds': {'loss_lo1': 0.353515625, 'loss_hi1': 1.0546875, 'norm_lo1': 0.004341160412877798, 'norm_hi1': 1.5741714239120483, 'cosim_lo1': -0.6287882924079895, 'cosim_hi1': 0.6293045282363892, 'mloss_lo1': 0.48828125, 'mloss_hi1': 2.34375, 'ldif_lo1': -0.1328125, 'ldif_hi1': 1.4375},
        'coco_gait_bounds_percentiles': {'loss_lo1_0.5': 0.3588330078125, 'loss_hi1_0.5': 1.0411523437500008, 'loss_lo1_1': 0.364150390625, 'loss_hi1_1': 1.0276171874999998, 'loss_lo1_2': 0.37478515625, 'loss_hi1_2': 1.0005468750000004, 'loss_lo1_10': 0.44296875, 'loss_hi1_10': 0.9039062499999999, 'ldif_lo1_0.5': -0.114443359375, 'ldif_hi1_0.5': 1.4336328125000002,
                                        'ldif_lo1_1': -0.09607421875, 'ldif_hi1_1': 1.429765625, 'ldif_lo1_2': -0.0593359375, 'ldif_hi1_2': 1.42203125, 'ldif_lo1_10': -0.0078125, 'ldif_hi1_10': 1.2214843749999997},
        'dsvq_test_gait_bounds': {'loss_lo1': 0.050048828125, 'loss_hi1': 1.609375, 'norm_lo1': 0.005769415758550167, 'norm_hi1': 3.1961231231689453, 'cosim_lo1': -0.5634750723838806, 'cosim_hi1': 0.9879628419876099, 'mloss_lo1': 0.05029296875, 'mloss_hi1': 1.96875, 'ldif_lo1': -0.0830078125, 'ldif_hi1': 0.91796875},
        'dsvq_test_gait_bounds_percentiles': {'loss_lo1_0.5': 0.0612274169921875, 'loss_hi1_0.5': 1.5823046875000013, 'loss_lo1_1': 0.072406005859375, 'loss_hi1_1': 1.5552343749999995, 'loss_lo1_2': 0.09476318359375, 'loss_hi1_2': 1.5010937500000006, 'ldif_lo1_0.5': -0.08276611328125, 'ldif_hi1_0.5': 0.855126953125003, 'ldif_lo1_1': -0.0825244140625, 'ldif_hi1_1': 0.7922851562499988, 'ldif_lo1_2': -0.082041015625, 'ldif_hi1_2': 0.6666015625000012},
        'flickr_gait_bounds': {'loss_lo1': 0.40234375, 'loss_hi1': 1.4296875, 'norm_lo1': 0.008455130271613598, 'norm_hi1': 2.7934703826904297, 'cosim_lo1': -0.8864620923995972, 'cosim_hi1': 0.8363902568817139, 'mloss_lo1': 0.494140625, 'mloss_hi1': 1.828125, 'ldif_lo1': -0.0546875, 'ldif_hi1': 1.296875},
        'flickr_gait_bounds_percentiles': {'loss_lo1_0.5': 0.405244140625, 'loss_hi1_0.5': 1.369746093750003, 'loss_lo1_1': 0.40814453125, 'loss_hi1_1': 1.309804687499999, 'loss_lo1_2': 0.4139453125, 'loss_hi1_2': 1.1899218750000014, 'ldif_lo1_0.5': -0.049853515625, 'ldif_hi1_0.5': 1.2427343750000026, 'ldif_lo1_1': -0.04501953125, 'ldif_hi1_1': 1.188593749999999, 'ldif_lo1_2': -0.0353515625, 'ldif_hi1_2': 1.0803125000000011},

        'nips_pip': {0: 1.3031207661379418, 0.5: 1.1348389507548011, 1: 0.9665571353716652, 2: 0.8757499367082859, 10: -0.10916038532475576},
        'coco_pip': {0: 0.07065558389989368, 0.5: -0.019434648224102763, 1: -0.10952488034809661, 2: -0.18754272962295146, 10: -0.694315146022629},
        'dsvq_test_pip': {0: 1.4784767631956872, 0.5: 1.186977853246678, 1: 0.8954789432976772, 2: 0.6881114074572153},
        'flickr_pip': {0: 1.1673314467393294, 0.5: 0.7463615381548352, 1: 0.32539162957035295, 2: 0.29015254140966856},

        'nips_ns': {0: -154.0, 0.5: -154.99, 1: -155.98, 2: -164.82000000000005, 10: -173.0},
        'coco_ns': {0: -103.0, 0.5: -103.7425, 1: -104.48499999999999, 2: -104.5, 10: -122.35},
        'dsvq_test_ns': {0: -154.0, 0.5: -157.96000000000004, 1: -161.91999999999996, 2: -162.0},
        'flickr_ns': {0: -113.0, 0.5: -115.97000000000003, 1: -118.93999999999997, 2: -121.45000000000002},

},

    VLMName.LLAVA_ONEVISION_0p5B:{
        "gait":{0:[0.240234375, 0.9140625],1:[0.6484375, 0.71875],2:[1.0463671875, 0.6250781249999982],5:[1.1328125, 0.53125],10:[1.2578125, 0.421875],20:[1.4375,0.296875]},
        "gait_bounds":{"ldif_lo":-0.5625, "mloss_lo":0.2021484375, "mloss_hi":7.03125, "norm_lo":0.0034393167588859797, "norm_hi":0.47303998470306396, "cosim_lo":-0.57869124},
        "pip":{0:1.1100336993646858 ,1:0.08395642597740757 , 2:-0.10599537016669928 , 5:-0.38438058064755803, 10:-0.6043401998844391, 20:-0.8536051289005471},#NEED TO GET THIS...
        #"pip_occ":{0: 51.301055580858865, 1:19.267027182756117 , 2:13.502515665199022 , 5:7.672095183515811, 10:4.736566877174392, 20:2.1043503963161583},
        "mc":{0:0.35057264903951685, 1:0.4546985065937042, 2:0.49397273778915407, 5:0.5304794460535049, 10:0.574992710351944, 20:0.6171177864074707}
        },

    VLMName.INTERNVL_3_2B:{
        'gait_bounds': {'loss_lo1': 0.01385498046875, 'loss_hi1': 1.828125, 'loss_lo2': 0.01251220703125, 'loss_hi2': 1.7578125, 'loss_lo3': 0.00921630859375, 'loss_hi3': 1.78125, 'loss_lo4': 0.008056640625, 'loss_hi4': 1.75, 'loss_lo5': 0.007720947265625, 'loss_hi5': 1.71875,
        'norm_lo1': 0.002027104375883937, 'norm_hi1': 0.5198320746421814, 'norm_lo2': 0.0018553799018263817, 'norm_hi2': 0.2729620337486267, 'norm_lo3': 0.001190596492961049, 'norm_hi3': 0.3635440766811371, 'norm_lo4': 0.0008308031246997416, 'norm_hi4': 0.20241601765155792, 'norm_lo5': 0.0008880632231011987, 'norm_hi5': 0.43482592701911926,
        'cosim_lo1': -0.8012385368347168, 'cosim_hi1': 0.8019676208496094, 'cosim_lo2': -0.6058996319770813, 'cosim_hi2': 0.9085567593574524, 'cosim_lo3': -0.6947848796844482, 'cosim_hi3': 0.6986855268478394, 'cosim_lo4': -0.8306132555007935, 'cosim_hi4': 0.6777468919754028, 'cosim_lo5': -0.7420363426208496, 'cosim_hi5': 0.7211511135101318,
        'mloss_lo1': 0.00579833984375, 'mloss_hi1': 3.671875, 'mloss_lo2': 0.01287841796875, 'mloss_hi2': 3.875, 'mloss_lo3': 0.0101318359375, 'mloss_hi3': 3.96875, 'mloss_lo4': 0.0081787109375, 'mloss_hi4': 3.84375, 'mloss_lo5': 0.00848388671875, 'mloss_hi5': 3.734375,
        'ldif_lo1': -0.390625, 'ldif_hi1': 2.90625, 'ldif_lo2': -0.349609375, 'ldif_hi2': 3.15625, 'ldif_lo3': -0.3125, 'ldif_hi3': 3.296875, 'ldif_lo4': -0.365234375, 'ldif_hi4': 3.203125, 'ldif_lo5': -0.1845703125, 'ldif_hi5': 3.09375},
        'gait_bounds_percentiles': {'loss_lo1_1': 0.0225762939453125, 'loss_hi1_1': 1.5705078124999998, 'loss_lo1_2': 0.027926025390625, 'loss_hi1_2': 1.46875, 'loss_lo1_5': 0.0412353515625, 'loss_hi1_5': 1.3755859374999995, 'loss_lo1_10': 0.05517578125, 'loss_hi1_10': 1.2816406249999996, 'loss_lo1_20': 0.0830078125, 'loss_hi1_20': 1.125,
        'ldif_lo1_1': -0.134814453125, 'ldif_hi1_1': 1.6800390624999997, 'ldif_lo1_2': -0.09375, 'ldif_hi1_2': 1.375, 'ldif_lo1_5': -0.03516845703125, 'ldif_hi1_5': 1.0548828124999998, 'ldif_lo1_10': -0.002514648437499996, 'ldif_hi1_10': 0.8517578124999998, 'ldif_lo1_20': 0.026269531250000006, 'ldif_hi1_20': 0.5941406250000001},
        'pip': {0: 1.3752939935684931, 1: 0.37829227170387425, 2: 0.11867384140120597, 5: -0.27563621729772914, 10: -0.6364944945590955, 20: -0.9996234377245719},
        'ns': {0: -61.25, 1: -80.5, 2: -86.49000000000001, 5: -93.5, 10: -98.5, 20: -104.5},
        'mc': {0: 0.2514841556549072, 1: 0.37395972073078154, 2: 0.394701122045517, 5: 0.4521953910589218, 10: 0.4956304430961609, 20: 0.5479042291641235},
    }
    },
    0.025:{
    VLMName.QWEN_2p5_VL_3B:{
        "gait":{0:[0.28515625, 1.1640625],1:[0.3046484375, 0.7462109374999999],2:[0.31052734375, 0.6875781249999999],5:[0.391455078125, 0.5859375],10:[0.8388671875, 0.5235351562499999],20:[0.933203125, 0.3914062500000002]},
        "gait_bounds":{"ldif_lo":-0.0546875, "mloss_lo":0.25390625, "mloss_hi":2.140625, "norm_lo":0.0013182901311665773, "norm_hi":1.555273413658142, "cosim_lo":-0.8270035982131958},
        "pip":{0:0.2763563433199534 ,1:-0.5918927960800316 , 2:-0.8079542437125616 , 5:-0.9996385997628311, 10:-0.9997248250231398, 20:-0.9999670221567097},
        #"pip_occ":{0: 46.335923265250685, 1:37.861092019514984 , 2:31.906905492132736 , 5:23.488855241797463, 10:18.088539998632463, 20:11.380382072488855},
        "mc":{0:0.3255233793918484, 1:0.46928912937641143, 2:0.5107188951969147, 5:0.5603158265352249, 10:0.6149698078632355, 20:0.6799704074859619},
        "ns":{0: -152.00000009979988, 1: -161.99, 2: -168.0, 5: -172.0, 10: -174.0, 20: -176.0}
    }
    },
    0.1:{
    VLMName.QWEN_2p5_VL_3B:{
        "gait":{0:[0.298828125, 1.5078125],1:[0.309521484375, 1.179765625],2:[0.339609375, 1.1021093749999995],5:[0.5046386718750001, 0.9494140624999998],10:[0.8708984375, 0.8828125],20:[0.94140625, 0.7421875]},
        "gait_bounds":{"ldif_lo":-0.05078125, "mloss_lo":0.314453125, "mloss_hi":2.546875, "norm_lo":0.0009021197911351919, "norm_hi":3.4451279640197754, "cosim_lo":-0.8712633848190308},
        "pip":{0:0.2763563433199534 ,1:-0.5918927960800316 , 2:-0.8079542437125616 , 5:-0.9996385997628311, 10:-0.9997248250231398, 20:-0.9999670221567097},
        #"pip_occ":{0: 46.335923265250685, 1:37.861092019514984 , 2:31.906905492132736 , 5:23.488855241797463, 10:18.088539998632463, 20:11.380382072488855},
        "mc":{0:0.3255233793918484, 1:0.46928912937641143, 2:0.5107188951969147, 5:0.5603158265352249, 10:0.6149698078632355, 20:0.6799704074859619},
        "ns":{0: -152.00000009979988, 1: -161.99, 2: -168.0, 5: -172.0, 10: -174.0, 20: -176.0}
    }
    },
    0.2:{
    VLMName.QWEN_2p5_VL_3B:{
        "gait":{0:[0.298828125, 1.7890625],1:[0.309521484375, 1.4494921874999998],2:[0.339609375, 1.414140625],5:[0.5046386718750001, 1.296875],10:[0.8708984375, 1.1875],20:[0.94140625, 1.0242187500000002]},
        "gait_bounds":{"ldif_lo":-0.0546875, "mloss_lo":0.34765625, "mloss_hi":2.828125, "norm_lo":0.0009021197911351919, "norm_hi":3.4451279640197754, "cosim_lo":-0.8712633848190308},
        "pip":{0:0.2763563433199534 ,1:-0.5918927960800316 , 2:-0.8079542437125616 , 5:-0.9996385997628311, 10:-0.9997248250231398, 20:-0.9999670221567097},
        #"pip_occ":{0: 46.335923265250685, 1:37.861092019514984 , 2:31.906905492132736 , 5:23.488855241797463, 10:18.088539998632463, 20:11.380382072488855},
        "mc":{0:0.3255233793918484, 1:0.46928912937641143, 2:0.5107188951969147, 5:0.5603158265352249, 10:0.6149698078632355, 20:0.6799704074859619},
        "ns":{0: -152.00000009979988, 1: -161.99, 2: -168.0, 5: -172.0, 10: -174.0, 20: -176.0}
    }
    },
    0.5:{
    VLMName.QWEN_2p5_VL_3B:{
        "gait":{0:[0.298828125, 2.0390625],1:[0.309521484375, 1.6604296874999998],2:[0.339609375, 1.6015625],5:[0.5046386718750001, 1.4925781249999996],10:[0.8708984375, 1.3363281249999996],20:[0.94140625, 1.1574218750000003]},
        "gait_bounds":{"ldif_lo":-0.078125, "mloss_lo":0.3046875, "mloss_hi":3.078125, "norm_lo":0.0009021197911351919, "norm_hi":3.4451279640197754, "cosim_lo":-0.8712633848190308},
        "pip":{0:0.2763563433199534 ,1:-0.5918927960800316 , 2:-0.8079542437125616 , 5:-0.9996385997628311, 10:-0.9997248250231398, 20:-0.9999670221567097},
        #"pip_occ":{0: 46.335923265250685, 1:37.861092019514984 , 2:31.906905492132736 , 5:23.488855241797463, 10:18.088539998632463, 20:11.380382072488855},
        "mc":{0:0.3255233793918484, 1:0.46928912937641143, 2:0.5107188951969147, 5:0.5603158265352249, 10:0.6149698078632355, 20:0.6799704074859619},
        "ns":{0: -152.00000009979988, 1: -161.99, 2: -168.0, 5: -172.0, 10: -174.0, 20: -176.0}
    }
    }

}


@dataclass
class AttackConfig:
    model_name_clf: str
    model_name_vlm: str
    n_gradient_steps: int
    d_gradient_steps: int
    lr: float
    lambda_clf: float
    lambda_vlm: float
    d_lambda_clf: float
    d_lambda_vlm: float
    user_query: str
    target_vlm_answer: str
    target_clf_idx: int
    max_perturbation: float
    d_max_perturbation: float

    def to_dict(self,):
        return asdict(self)

def launch_attack(
    image: torch.tensor,
    #
    vlmi: VLM,
    config: AttackConfig,
    print_every: int,
    device: str,
    morevlm=None,#attack additional victim VLMs at once (not used for paper)
    untargeted=False,
    patch_attack=False,
    npatches=1,
    pipdef=None,
    mcdef=None,
    nsdef=None,
    dpdef=None,
    nmc=10,
    naib=False,
    refsims=None,
    rmask=False,
    perz=0,
    exclude=True,
    topk=0.05,
    ablate=None,
    cw=False,
    maskadapt=None,
    gait_lr=8/255,
    stubborn=False,
    dspz=None,
    adam=False,
    sigsharp=10,
    orth=False,
):

    vlms=[vlmi]
    if morevlm is not None:
        vlms = vlms + morevlm
    """
    launches the attack
    """
    user_query = config.user_query

    target_vlm_answer = config.target_vlm_answer
    lambda_vlm = config.lambda_vlm
    n_gradient_steps = config.n_gradient_steps
    d_gradient_steps = config.d_gradient_steps
    lr = config.lr
    max_perturbation_pixels = config.max_perturbation if not patch_attack else 255


    initial_image = image.clone().float() if device=="cuda" else image.clone()
    #roco = initial_image.clone().detach()
    if lambda_vlm:
        ftvs, tts=[],[]
        for vlm in vlms:
        #mock_images = [initial_image]
            full_text_vlm_prompt, target_tokens = vlm.get_training_prompt(user_query, target_vlm_answer)
            ftvs.append(full_text_vlm_prompt)
            tts.append(target_tokens)


    loss_vlm = torch.tensor([0]).to(device)

    if patch_attack:
        dx,dy=int(( (config.max_perturbation/npatches)**0.5)*image.shape[1]), int(( (config.max_perturbation/npatches)**0.5)*image.shape[2])
        mask= torch.zeros_like(image)
        #mask_s = torch.zeros_like(image)
        check=mask[0].clone()
        check[-dx:,:] = 1.0
        check[:, -dy:] = 1.0
        for n in range(npatches):

            fail=True
            print("looking for patch placement")
            while fail:
                opt = torch.where(check==0)

                #xmsk, ymsk = torch.randint(low=dx, high=image.shape[1]-dx, size=(1,1)).item(), torch.randint(low=dy, high=image.shape[2]-dy, size=(1,1)).item()

                #xmsk, ymsk = , torch.randint(low=0, high=image.shape[2]-dy, size=(1,1)).item()
                xmsk=torch.randint(low=0, high=opt[0].shape[0], size=(1,1)).item()
                xmsk, ymsk = opt[0][xmsk], opt[1][xmsk]

                #if xmsk < dx or xmsk > image.shape[1] - dx or ymsk < dy or ymsk > image.shape[2] - dy:# or torch.sum(check[xmsk:xmsk+dx, ymsk:ymsk+dy])>0.0:
                #    continue
                if True:#not n:
                    tcheck = check.clone()
                    check[max(0,xmsk-dx):xmsk+dx, max(0,ymsk-dy):ymsk+dy]=1.0
                    if n+1<npatches and check.sum() >= image.shape[2]*image.shape[1]:# - check.sum() < dx*dy:
                        print(f"not possible {check.sum().item()}")
                        check = tcheck#mask[0].clone()
                        del tcheck
                        continue
                    mask[:, xmsk:xmsk+dx, ymsk:ymsk+dy]=1.0
                #else:
                #    mask_s[:, xmsk:xmsk+dx, ymsk:ymsk+dy]=1.0
                fail=False
                print(f"done ({n}): {xmsk}, {ymsk} to {xmsk+dx}, {ymsk+dy}")
        image=image*(1-mask) + (mask*0.5*255).to(torch.uint8).to(torch.float)
        #image=image*(1-mask_s) + mask_s*0.5*255

    best_loss, best_det, best_dist, best_total, best_full_det = torch.inf, torch.inf, -torch.inf, torch.inf, torch.inf
    best, soon, fifty = None, None, None
    best_big, stealth_1, stealth_2, stealth_big = None, None, None, None
    best_esc, best_big_esc, stealth_esc, stealth_big_esc = None, None, None, None
    best_exit, best_exit_det, best_exit_big, best_exit_det_big = torch.inf, torch.inf, torch.inf, torch.inf
    times={"success":-1, "lowest":-1, "lowest_glob":-1, "lowest_und":-1, "lowest_glob_und":-1}
    ims=[]


    itertimes=[]
    totime=0
    target_success = False#BE CAREFUL... WE ARE ASSUMING THE ATTACK TARGET HAS NOT BEEN ACHIEVED BEFORE THE ATTACK BEGINS!
    if mcdef is not None:
        print(f"Generating {nmc} MirrorCheck examples")
        starter = time.time()
        mcimgs = []
        for mci in range(nmc):
            mcimgs.append(mcdef.generate(target_vlm_answer))
        totime += time.time() - starter
        print("ready")


    bounds=det_thresh_dict[topk][vlmi.name]["gait_bounds"]
    if perz:
        bounds["loss_lo1"] = det_thresh_dict[topk][vlmi.name]["gait_bounds_percentiles"][f"loss_lo1_{perz}"]
        bounds["ldif_hi1"] = det_thresh_dict[topk][vlmi.name]["gait_bounds_percentiles"][f"ldif_hi1_{perz}"]
        if dspz is not None:
            bounds=det_thresh_dict[topk][vlmi.name][f"{dspz}_gait_bounds"]
            bounds["loss_lo1"] = det_thresh_dict[topk][vlmi.name][f"{dspz}_gait_bounds_percentiles"][f"loss_lo1_{perz}"]
            bounds["ldif_hi1"] = det_thresh_dict[topk][vlmi.name][f"{dspz}_gait_bounds_percentiles"][f"ldif_hi1_{perz}"]

    if gait_lr != 8/255:
        #input(gait_lr)
        lrkey = int(gait_lr*255)
        bounds = det_thresh_dict_glr[lrkey][vlmi.name]["gait_bounds"]
        if perz:
            bounds["loss_lo1"] = det_thresh_dict_glr[lrkey][vlmi.name]["gait_bounds_percentiles"][f"loss_lo1_{perz}"]
            bounds["ldif_hi1"] = det_thresh_dict_glr[lrkey][vlmi.name]["gait_bounds_percentiles"][f"ldif_hi1_{perz}"]

    pipth=det_thresh_dict[topk][vlmi.name]["pip"][perz]
    mcth=det_thresh_dict[topk][vlmi.name]["mc"][perz]
    nsth=det_thresh_dict[topk][vlmi.name]["ns"][perz]
    if dspz is not None:
        pipth=det_thresh_dict[topk][vlmi.name][f"{dspz}_pip"][perz]
        nsth=det_thresh_dict[topk][vlmi.name][f"{dspz}_ns"][perz]

    if ablate is not None:
        for fst in ablate:
            for i in range(5):
                bounds[f"{fst}_lo{i}"] = -1000
                bounds[f"{fst}_hi{i}"] = 1000
    bounds = {k: torch.Tensor([v]).to(torch.bfloat16).cuda() for k, v in bounds.items()}

    masker = torch.nn.MaxPool2d(kernel_size = 3, stride=1, padding=1).cuda()

    exit=False
    momentum=None
    if adam:
        adam_m, adam_v, adam_t = torch.zeros_like(image),torch.zeros_like(image),0
    for i in range(n_gradient_steps):
        starter = time.time()
        image.requires_grad = True
        tracker = image.clone().detach().requires_grad_(True)
        image = tracker
        if cw:
            cw_grad = torch.autograd.grad(0.01*torch.norm(image - initial_image, p=2), image)[0]
        if dpdef is not None:
            purified = purify(dpdef, image.unsqueeze(0)).squeeze(0)
            loss_vlm=0
            for full_text_vlm_prompt, target_tokens, vlm in zip(ftvs, tts, vlms):
                out = vlm.forward(purified, full_text_vlm_prompt, overwrite=True)
                #olout=out
                loss_vlm += vlm.compute_gen_loss(out, target_tokens)
        else:
            defclfs=False

            if lambda_vlm > 0:
                loss_vlm=0
                for full_text_vlm_prompt, target_tokens, vlm in zip(ftvs, tts, vlms):
                    out = vlm.forward(image, full_text_vlm_prompt, overwrite=True)
                    #olout=out
                    loss_vlm += vlm.compute_gen_loss(out, target_tokens)


        #if bet or untargeted:
        #    total_loss = -lambda_clf * loss_clf - lambda_vlm * loss_vlm
        #else:
        total_loss = lambda_vlm * loss_vlm

        #input(total_loss)
        if i==0 or (i+1)%print_every==0 or dpdef is not None:
            print(f"Iter {i+1:4d}/{n_gradient_steps}, RAM usage -> {get_memory_consumption(device):.2f} GB, Losses -> VLM: {loss_vlm.item():.8f}, Total: {total_loss.item():.8f}")


        obj_grads = torch.autograd.grad(total_loss, image, create_graph=not naib and pipdef is None and mcdef is None and dpdef is None and nsdef is None, retain_graph=not naib and pipdef is None and mcdef is None and dpdef is None and nsdef is None)[0]
        #perz% FPR in TRAINING DATA

        if (pipdef !=None or nsdef != None):
            vlm_answer_b = vlmi.generate_e2e(image.clone().detach(), user_query, temperature=0)[0]
            print(vlm_answer_b)

        #adaptive attack for PIP
        if pipdef is not None:
            acro=[]
            ini_curr = image.clone().detach()

            inatt = pipdef.get_attention(image/255).flatten()
            #FOR RBF KERNEL:
            """
            gamma = pipdef.svm._gamma
            svec, scoef, sbias = torch.tensor(pipdef.svm.support_vectors_).cuda(), torch.tensor(pipdef.svm.dual_coef_).cuda(), torch.tensor(pipdef.svm.intercept_).cuda()
            K = torch.exp(-gamma * torch.linalg.norm(inatt - svec, dim=-1)**2)
            piploss = (K * scoef).sum() + sbias
            """
            #FOR LINEAR KERNEL (like in PIP paper)
            scoef, sbias = torch.tensor(pipdef.svm.coef_).cuda(), torch.tensor(pipdef.svm.intercept_).cuda()
            piploss = (scoef*inatt).sum() + sbias
            acro.append(piploss.item())

            #FOR A BINARY SVM (like in PIP paper)
            piploss = torch.clip(piploss - pipth, min=0)

            adapt_grads = torch.autograd.grad(piploss, image)[0]

        #adaptive attack for MirrorCheck
        elif mcdef is not None:
            ini_curr = image.clone().detach()
            zimz=torch.cat([mcdef.similarity(genimg, image/255).unsqueeze(0) for genimg in mcimgs])
            mcloss = torch.min(zimz)
            acro=[mcloss.item()]
            mcloss = torch.clip(mcth - mcloss, min=0)
            adapt_grads = torch.autograd.grad(mcloss, image)[0]

        #adaptive attack for nearside
        elif nsdef is not None:
            ini_curr = image.clone().detach()
            _, detns = nsdef.detect(image, user_query = user_query)
            acro=[detns.item()]
            nsloss = torch.clip(detns - nsth, min=0)
            adapt_grads = torch.autograd.grad(nsloss, image)[0]

        #adaptive attack for diffpure
        elif dpdef is not None:
            ini_curr = image.clone().detach()
            adapt_grads=0#the "adapt grads" are already the normal grads for the image with purification




        #adaptive attack against AdverStep
        elif not naib:
            if i==0 or (i+1)%print_every==0 or orth:
                vlm_answer_b = vlmi.generate_e2e(image.clone().detach(), user_query, temperature=0)[0]
                print(vlm_answer_b)
            metagrads = 0
            if n_gradient_steps <= 1:
                tracker = tracker.clone().detach()
            current = image
            ini_curr = image.clone().detach()
            acro = []
            bows = 0
            for stepoton in range(d_gradient_steps):#N - 2
                acri = []
                bound_loss = 0
                if not stepoton:
                    grads=obj_grads
                    loss = total_loss#obj_loss

                if n_gradient_steps > 1:
                    loss_loss = torch.clip(bounds[f"loss_lo{stepoton+1}"] - loss, min=0)#torch.clip(threshold[stepoton][0] - loss, min=0)#loss as detection score
                    if not stepoton and not orth:#for OPGD we do not want to minimize the loss simultaneously ever...
                        loss_loss = loss_loss + torch.clip(loss - bounds[f"loss_lo{stepoton+1}"], min=0)#+ torch.clip(loss - threshold[stepoton][0], min=0)#the loss at first step is important, as an attacker, we want to minimize it



                grad_loss = torch.linalg.norm(grads.sum(0))#grad norm as detection score
                acri.append(loss.item())
                acri.append(grad_loss.item())

                if n_gradient_steps > 1:
                    grad_loss = torch.clip(bounds[f"norm_lo{stepoton+1}"] - grad_loss, min=0) + torch.clip(grad_loss - bounds[f"norm_hi{stepoton+1}"], min=0)
                    bound_loss = bound_loss + loss_loss + grad_loss

                lastgrads=grads.sum(0)#None
                ini_curr1 = current.clone().detach()
                current = attack_step_pgd(current, grads, gait_lr, 255, ini_curr)
                ini_curr2 = current.clone().detach()

                out = vlmi.forward(current, full_text_vlm_prompt, overwrite=True)
                nloss = vlmi.compute_gen_loss(out, target_tokens)
                grads = torch.autograd.grad(nloss, current, create_graph=True, retain_graph=True)[0]


                cosim_un = lastgrads*grads.sum(0)
                cosim=cosim_un/(torch.linalg.norm(lastgrads) * torch.linalg.norm(grads.sum(0)))#cosine sim. as detection score
                acri.append(cosim.sum().item())

                if n_gradient_steps > 1:
                    cosim_loss = torch.clip(bounds[f"cosim_lo{stepoton+1}"] - cosim.sum(), min= 0)#only matters if cosine similarity is too low
                    bound_loss = bound_loss + cosim_loss
                else:
                    grads = grads.clone().detach()


                dmask_trick = torch.sigmoid(sigsharp * (cosim_un - cosim_un.quantile(0.95)))#FOR BPDA

                dmask, _ = torch.sort((cosim_un).flatten())
                dmask=dmask[-int(224*224*topk)]
                dmask = 1.0*(cosim_un >= dmask)

                dmask = masker(dmask.unsqueeze(0)).squeeze(0)
                dmask_trick = masker(dmask_trick.unsqueeze(0)).squeeze(0)

                dmask = dmask_trick + (dmask - dmask_trick).detach()#STE + BPDA FOR MASK

                ris = ini_curr1.clone().detach() - current.clone().detach()
                current+=ris#ini_curr.clone().detach()
                current*=(1-dmask)# + dmask*0.5*255
                current+= (dmask*0.5*255)#.to(torch.uint8).to(torch.float)

                dout = vlmi.forward(current, full_text_vlm_prompt, overwrite=True)
                dloss = vlmi.compute_gen_loss(dout, target_tokens)#masked loss

                if n_gradient_steps > 1:
                    bound_loss += torch.clip(bounds[f"mloss_lo{stepoton+1}"] - dloss, min=0) + torch.clip(dloss - bounds[f"mloss_hi{stepoton+1}"], min=0)#masked loss should be within bounds (AGG) (QWEN)#remoed lasttttt

                ldif = dloss - loss
                acri.append(dloss.item())
                acri.append(ldif.item())
                if n_gradient_steps > 1:
                    #bound_loss += torch.clip(ldif - threshold[stepoton][1], min=0) + torch.clip(bounds[f"ldif_lo{stepoton + 1}"] - ldif, min=0)
                    bound_loss += torch.clip(ldif - bounds[f"ldif_hi{stepoton + 1}"], min=0) + torch.clip(bounds[f"ldif_lo{stepoton + 1}"] - ldif, min=0)

                if n_gradient_steps > 1:
                    if maskadapt == 'top':
                        metagrads += torch.autograd.grad(torch.clip(cosim_un - dmask, min = 0).sum(), tracker, retain_graph=True)[0]
                    elif maskadapt == 'std':
                        metagrads += torch.autograd.grad(torch.std(cosim_un), tracker, retain_graph=True)[0]
                    metagrads += torch.autograd.grad(bound_loss, tracker, retain_graph=stepoton < d_gradient_steps-1)[0]

                #input([loss, nloss])
                loss = nloss
                ris = ini_curr2.clone().detach() - current.clone().detach()
                current += ris
                acro.append(acri)
                if n_gradient_steps > 1:
                    bows += torch.Tensor([bound_loss.item()]).to(torch.bfloat16).cuda()
                else:
                    current = current.clone().detach()
                    current.requires_grad = True
                #print(acri)
                #input("go and check")

            adapt_grads = metagrads
            image=ini_curr.clone().detach()

            if not orth:
                bows = bows - torch.clip(torch.Tensor([acro[0][0]]).to(torch.bfloat16).cuda() - bounds[f"loss_lo1"] , min=0)#np.clip(torch.Tensor([acro[0][0]]).to(torch.bfloat16).cuda() - bounds[f"loss_lo1"] , min=0)#np.clip(acro[0][0] - threshold[0][0], min=0)#THE BIG CORRECTION! (WE TAKE THE CLIPPED LOSS OFF THE BOUND CHECKER, NOT THE UNCLIPPED LOSS, WHICH OVERESTIMATES STEALTH)

        else:
            adapt_grads = 0

        if naib:
            ini_curr = image.clone().detach()

        with torch.no_grad():
            if patch_attack:
                obj_grads=obj_grads*mask
                adapt_grads=adapt_grads*mask#_s

            momentum = obj_grads if ((naib or pipdef !=None or mcdef != None or dpdef != None or nsdef != None) and not orth) else 0#if momentum is None else momentum*0.5 + grads
            if orth:
                if not target_success:
                    adapt_grads = obj_grads - project(obj_grads, adapt_grads)
                else:
                    adapt_grads = adapt_grads - project(adapt_grads, obj_grads)
            if cw:
                momentum = momentum + cw_grad
            if n_gradient_steps <= 1:
                momentum = torch.zeros_like(image)
            if adam:#(only for gait atm)
                adam_t+=1
                adam_m = 0.9 * adam_m + (1 - 0.9) * adapt_grads
                adam_v = 0.999 * adam_v + (1 - 0.999) * (adapt_grads ** 2)

                m_hat = adam_m / (1 - 0.9 ** adam_t)
                v_hat = adam_v / (1 - 0.999 ** adam_t)
                adapt_grads = m_hat / (torch.sqrt(v_hat) + 1e-8)

            image = attack_step_pgd(image, momentum + adapt_grads, lr, max_perturbation_pixels, initial_image)
        itertimes.append(time.time() - starter)




        print_this=False
        und=False
        if naib or dpdef is not None:#naive attacks por purifier-only defense...
            und=True
            to_see = purified if dpdef is not None else ini_curr
            if loss_vlm < best_loss:
                print_this=True
                #vlm_answer_b = vlmi.generate_e2e(to_see.clone().detach(), user_query, temperature=0)[0]
                best=ini_curr.clone().detach()
                best_loss  = loss_vlm
            if i==0 or (i+1)%print_every==0 or dpdef is not None:
                vlm_answer_b = vlmi.generate_e2e(to_see.clone().detach(), user_query, temperature=0)[0]
                print(vlm_answer_b)

        if not naib and pipdef is None and mcdef is None and dpdef is None and nsdef is None and n_gradient_steps > 1:# and bound_loss <= 0:# and loss_large <=0:

            if acro[0][0] < best_loss:
                print("lowest loss!")
                vlm_answer_b = vlmi.generate_e2e(ini_curr.clone().detach(), user_query, temperature=0)[0]
                print(vlm_answer_b)
                best = ini_curr.detach().clone().cpu()
                best_loss = loss_vlm
                print_this=True
                times["lowest"] = i
            if bows <=0 and acro[0][0] < best_det:#det_score < best_det:
                print("lowest loss! (undetectable only)")
                vlm_answer_b = vlmi.generate_e2e(ini_curr.clone().detach(), user_query, temperature=0)[0]
                print(vlm_answer_b)
                stealth_1 = ini_curr.detach().clone().cpu()
                best_det = acro[0][0]#loss_vlm#det_score
                print_this=True
                times["lowest_und"] = i
                und = True
            if bows <=0 and (bows + acro[0][0]) < best_full_det:#det_score < best_det:
                print("best overall! (undetectable only)")
                vlm_answer_b = vlmi.generate_e2e(ini_curr.clone().detach(), user_query, temperature=0)[0]
                print(vlm_answer_b)
                stealth_2 = ini_curr.detach().clone().cpu()
                best_full_det = bows + acro[0][0]#bound_loss#det_score
                print_this=True
                times["lowest_glob_und"] = i
                und = True
            if (bows + acro[0][0]) < best_total:#det_score - dist_score - loss_large < best_total:
                print("best overall!")
                vlm_answer_b = vlmi.generate_e2e(ini_curr.clone().detach(), user_query, temperature=0)[0]
                print(vlm_answer_b)
                best_big = ini_curr.detach().clone().cpu()
                best_total = bows + acro[0][0]#det_score + bound_loss# - dist_score - loss_large#full_det_score + loss_vlm
                print_this=True
                times["lowest_glob"] = i
            if print_this:
                #print("way out heah")
                #input(torch.prod(chek==ini_curr))
                print(f'obj/self loss: {acro[0][0]}')#obj_loss.item()}')
                print(f'normloss: {acro[0][1]}')#cosloss.item()}')
                print(f'cosloss: {acro[0][2]}')#cosloss.item()}')
                #print(f'self loss: {loss.item()}')
                print(f'masked loss: {acro[0][3]}')#dloss.item()}')
                print(f'dif: {acro[0][4]}')#ldif.item()}')
                print(f'BOWS: {bows}')
            und = bows<=0#things can be undetectable without achieving the best lost so far...




        if pipdef is not None or mcdef is not None or nsdef is not None:
            #print_this=False
            if pipdef is not None:
                detloss = piploss
            elif mcdef is not None:
                 detloss = mcloss
            elif nsdef is not None:
                detloss = nsloss
            if total_loss < best_loss:
                print("lowest loss!")
                vlm_answer_b = vlmi.generate_e2e(ini_curr.clone().detach(), user_query, temperature=0)[0]
                print(vlm_answer_b)
                best = ini_curr.detach().clone().cpu()
                best_loss = total_loss
                print_this=True
                times["lowest"] = i


            if total_loss + detloss < best_total:#det_score - dist_score - loss_large < best_total:
                print("best overall!")
                vlm_answer_b = vlmi.generate_e2e(ini_curr.clone().detach(), user_query, temperature=0)[0]
                print(vlm_answer_b)
                best_big = ini_curr.detach().clone().cpu()
                best_total = total_loss + detloss
                print_this=True
                times["lowest_glob"] = i

            if total_loss < best_det and detloss<=0:
                print("lowest loss! (undetectable)")
                vlm_answer_b = vlmi.generate_e2e(ini_curr.clone().detach(), user_query, temperature=0)[0]
                print(vlm_answer_b)
                stealth_1 = ini_curr.detach().clone().cpu()
                best_det = total_loss
                print_this=True
                times["lowest_und"] = i
                und = True

            if total_loss + detloss < best_full_det and detloss<=0:#det_score - dist_score - loss_large < best_total:
                print("best overall! (undetectable)")
                vlm_answer_b = vlmi.generate_e2e(ini_curr.clone().detach(), user_query, temperature=0)[0]
                print(vlm_answer_b)
                stealth_2 = ini_curr.detach().clone().cpu()
                best_full_det = total_loss + detloss
                print_this=True
                times["lowest_glob_und"] = i
                und = True

            if print_this:
                #print("way out heah")
                #input(torch.prod(chek==ini_curr))
                print(f'obj loss: {total_loss.item()}')
                print(f'pip/mc loss: {detloss.item()}')
            und = detloss<=0

        target_success = target_vlm_answer.lower().translate(str.maketrans('', '', string.punctuation)) == vlm_answer_b.lower().translate(str.maketrans('', '', string.punctuation))

        if und and target_success and times["success"] == -1:
            times["success"] = i
            best_esc = ini_curr.detach().clone().cpu()
            if naib or dpdef is not None:
                stealth_1 = ini_curr.detach().clone().cpu()
            if not stubborn:
                break

        del ini_curr

    if n_gradient_steps <= 1:
        return acro, 0
    print("ALL OVER")
    #input()
    for k in times.keys():
        adon=-1
        if times[k] != -1:
            adon = sum(itertimes[:times[k]])
        itertimes.append(adon)
    itertimes.append(totime)
    return [image.clone(), best, best_big, stealth_1, stealth_2, stealth_big, best_esc, best_big_esc, stealth_esc, stealth_big_esc], itertimes##soon, fifty




def attack_step_pgd(
    image: torch.tensor,
    grads: torch.tensor,
    lr: float,
    max_perturbation_pixels: int,
    initial_image: torch.tensor,
    additive=False
):
    """
    which direction should we go, given the computed loss gradient
    """
    # take step
    image = image - lr * torch.sign(grads)
    if not additive:
        # clip according to attack budget
        if max_perturbation_pixels < 255:
            image = torch.clip(image, min=initial_image-max_perturbation_pixels, max=initial_image+max_perturbation_pixels)#, out=image)
        # clip to make sure we stay within allowed RGB values
        image = torch.clip(image, min=0, max=255)#, out=image)
    else:
        # clip according to attack budget
        #input(max_perturbation_pixels)
        torch.clip(image, min=initial_image-max_perturbation_pixels*2/255, max=initial_image+max_perturbation_pixels*2/255, out=image)
        # clip to make sure we stay within allowed RGB values
        torch.clip(image, min=-1, max=1, out=image)

    return image

def dif_entropy(x, window=torch.tensor(1)):
    x=torch.sort(x).values
    m=window#window_length
    n=torch.tensor(x.shape[-1])
    dif=x[1:] - x[:-1]
    term1 = 1/(n-m) * torch.sum(torch.log((n+1)/m * dif), axis=-1)
    k = torch.arange(m, n+1, dtype=term1.dtype)
    return term1 + torch.sum(1/k) + torch.log(m) - torch.log(n+1)
