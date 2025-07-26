TRIALS = {

    'single_bump': {'train': ['BUMP_h26'],
        'test': ['BUMP_h26']
    },
    'single_phill': {'train': ['PHLL_case_1p2'],
                    'test': ['PHLL_case_1p2']
                    },
    'bump_extrap': {
        'train': ['BUMP_h20', 'BUMP_h26', 'BUMP_h38'],
        'test': ['BUMP_h42']
    },
    'bump_inter': {
        'train': ['BUMP_h20', 'BUMP_h26', 'BUMP_h38', 'BUMP_h42'],
        'test': ['BUMP_h31']
    },
    'phill_inter': {
        'train': ['PHLL_case_0p5', 'PHLL_case_0p8', 'PHLL_case_1p5'],
        'test': ['PHLL_case_1p0']
    },
    'phill_extrap': {
        'train': ['PHLL_case_0p5', 'PHLL_case_0p8', 'PHLL_case_1p0', 'PHLL_case_1p5'],
        'test': ['PHLL_case_1p2']
    },
    'full_inter': {
        'train': ['BUMP_h20', 'BUMP_h26', 'BUMP_h38', 'PHLL_case_0p5', 'PHLL_case_0p8', 'PHLL_case_1p5'],
        'test': ['BUMP_h42', 'PHLL_case_1p0']
    },
    'full_extrap': {
        'train': ['BUMP_h20', 'BUMP_h26', 'BUMP_h38', 'BUMP_h42',
                  'PHLL_case_0p5', 'PHLL_case_0p8', 'PHLL_case_1p0', 'PHLL_case_1p5'],
        'test': ['BUMP_h31', 'PHLL_case_1p2']
    },
    'all': {
        'train': [
            'DUCT_1250', 'DUCT_1300', 'DUCT_1350', 'DUCT_1400', 'DUCT_1600',
            'DUCT_1800', 'DUCT_2000', 'DUCT_2205', 'DUCT_2600', 'DUCT_2900',
            'DUCT_3200', 'BUMP_h20', 'BUMP_h26', 'BUMP_h38', 'BUMP_h42',
            'PHLL_case_0p5', 'PHLL_case_1p5', 'PHLL_case_0p8', 'PHLL_case_1p2'
        ],
        'test': [
            'DUCT_1100', 'DUCT_3500', 'DUCT_1500', 'DUCT_2400',
            'BUMP_h31', 'PHLL_case_1p0', 'CNDV_12600', 'CNDV_20580'
        ]
    }
}
