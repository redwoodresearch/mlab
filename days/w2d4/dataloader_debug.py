    data_train_gen, data_test_gen = torchtext.datasets.IMDB(
        root=".data", split=("train", "test")
    )
    data_train = list(data_train_gen)
    data_test = list(data_test_gen)

    small_indices = t.randint(0, len(data_train), size=(256,))
    small_train = [data_train[i] for i in small_indices]
    small_test = [data_test[i] for i in small_indices]
    
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
    collate_fn = get_imdb_collate_fn(512, tokenizer, device="cuda")

    dl_train_small = DataLoader(
        small_train,
        batch_size=train_batch_size,
        collate_fn=collate_fn,
        shuffle=True,
    )