class Args:
    # 纯蒙古文优化 空格
    train_path = 'kongge_MN_data_modify/train_process1.json'
    test_path = 'kongge_MN_data_modify/test_process1.json'
    seq_labels_path = './kongge_MN_data_modify/intents.txt'
    token_labels_path = './kongge_MN_data_modify/slots.txt'
    bert_dir = './cino-bert/'
    save_dir = './kongge_MN_data_modify/checkpoints/'
    save_name = 'cino_slot.pt'
    load_dir = './kongge_MN_data_modify/checkpoints/cino_slot.pt'
    domain_labels_path = './kongge_MN_data_modify/domains.txt'

    # 训练
    do_train = True
    do_eval = True
    do_test = True
    do_save = True
    do_predict = True
    load_model = False
    #预测
    # do_train = False
    # do_eval = True
    # do_test = True
    # do_save = False
    # do_predict = True
    # load_model = True

    device = None
    seqlabel2id = {}
    id2seqlabel = {}
    with open(seq_labels_path, 'r') as fp:
        seq_labels = fp.read().split('\n')
        for i, label in enumerate(seq_labels):
            seqlabel2id[label] = i
            id2seqlabel[i] = label

    tokenlabel2id = {}
    id2tokenlabel = {}
    with open(token_labels_path, 'r') as fp:
        token_labels = fp.read().split('\n')
        for i, label in enumerate(token_labels):
            tokenlabel2id[label] = i
            id2tokenlabel[i] = label

    domainlabel2id = {}
    id2domainlabel = {}
    with open(domain_labels_path, 'r') as fp:
        domain_labels = fp.read().split('\n')
        for i, label in enumerate(domain_labels):
            domainlabel2id[label] = i
            id2domainlabel[i] = label

    tmp = ['O']
    for label in token_labels:
        B_label = 'B-' + label
        I_label = 'I-' + label
        tmp.append(B_label)
        tmp.append(I_label)
    nerlabel2id = {}
    id2nerlabel = {}
    for i,label in enumerate(tmp):
        nerlabel2id[label] = i
        id2nerlabel[i] = label

    hidden_size = 768
    seq_num_labels = len(seq_labels)
    token_num_labels = len(tmp)
    domain_num_labels = len(domain_labels)
    # max_len = 152#50   MN_DATA
    # max_len = 200 #50    MN_DATA_Modify
    max_len = 155 #50    MN_DATA_Modify
    batchsize = 16#64
    lr = 2e-5
    epoch = 40
    hidden_dropout_prob = 0.1

if __name__ == '__main__':
    args = Args()
    print(args.seq_labels)
    print(args.seqlabel2id)
    print(args.tokenlabel2id)
    print(args.nerlabel2id)