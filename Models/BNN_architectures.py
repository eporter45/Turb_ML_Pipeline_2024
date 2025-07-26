
bl = 'branch_layers'
tl = 'trunk_layers'

architectures = {
    'kaggle':          {'branch_layers': [20],'trunk_layers': [20, 20]},
    'b64x2_t128x2_64': {'branch_layers': [64, 64], 'trunk_layers': [128, 128, 64]},
    'b20x2_t10x2':     {'branch_layers': [20],'trunk_layers': [10, 10]},
    'b5x2_t10x2':      {'branch_layers': [5,5], 'trunk_layers': [10, 10]},
    'b20_t10_30':      {'branch_layers': [20],'trunk_layers': [10, 30]},
    'b20_t30_10':      {'branch_layers': [20], 'trunk_layers': [30, 10]},
    'name' :           {'branch_layers': [5, 10], 'trunk_layers': [40, 10]},
    }


architectures2 = {
    'kaggle': {'branch_layers': [20, 20], 'trunk_layers': [40, 40, 20]},
    'Wide': {'branch_layers': [64, 64], 'trunk_layers': [128, 128, 64]},
    'NarrowDeep': {'branch_layers': [16, 16], 'trunk_layers': [32, 32, 16]},
    'BranchWide_TrunkNarrow': {'branch_layers': [16, 32], 'trunk_layers': [128, 32]},
    'BranchNarrow_TrunkWide': {'branch_layers': [32, 16], 'trunk_layers': [ 32, 128]}
}

