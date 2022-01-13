import mlab.hpsearch

def main():
    mlab.hpsearch.hpsearch(
        "sentiment_model",
        "days.w2d4.sentiment.train",
        "days/w2d4/sentiment.gin",
        {"train.lr": [1e-3, 1e-4, 1e-5], "train.max_seq_len": [128, 256]},
        comet_key="vABV7zo6pqS7lfzZBhyabU2Xe",
        local=False
    )

if __name__ == '__main__':
    main()