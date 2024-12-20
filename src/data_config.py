# static params
data_args = {
    'mr':
        {'train_size': 7108,
         'test_size': 3554,
         'valid_size': 711,
         "num_classes": 2
         },
    'ohsumed':
        {'train_size': 3357,
         'test_size': 4043,
         'valid_size': 336,
         "num_classes": 23
         },
    '20ng':
        {'train_size': 11314,
         'test_size': 7532,
         'valid_size': 1131,
         "num_classes": 20
         },
    'R8':
        {'train_size': 5485,
         'test_size': 2189,  # 2189 (nopunct), 2190 (restore)
         'valid_size': 548,
         "num_classes": 8
         },
    'R52':
        {'train_size': 6532,  # 6532 (nopunct), 6530 (restore)
         'test_size': 2568,  # 2568 (nopunct), 2570 (restore)
         'valid_size': 653,
         "num_classes": 52
         },
    'AGNews':
        {'train_size': 6000,
         'test_size': 3000,
         'valid_size': 600,
         "num_classes": 4
         },
    'dblp':
        {'train_size': 12000,
         'test_size': 3000,
         'valid_size': 1200,
         'num_classes': 6
         },
    'TREC':
        {'train_size': 5452,
         'test_size': 500,
         'valid_size': 545,
         'num_classes': 6
         }
}
