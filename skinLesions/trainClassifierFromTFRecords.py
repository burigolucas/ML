# depdencies

def build_model(config):
    model = None
    return model

def train(model,config,dataloader):
    return model

def test(model,dataloader):
    return True

def main():

    config = {}
    dataloader = None

    model = build_model(config = config)
    model = train(
        model = model,
        dataloader = dataloader,
        config = config)

    test(model,dataloader)

if __name__ == '__main__':
    main()