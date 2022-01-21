import random, torch
from prettytable import PrettyTable

def count_parameters_require_grads(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():

        if not parameter.requires_grad:
            continue

        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def set_random(config):
    if config.manualSeed is None:
        config.manualSeed = random.randint(1, 10000)
    print("Random Seed: {}".format(config.manualSeed))
    random.seed(config.manualSeed)
    torch.manual_seed(config.manualSeed)
    torch.cuda.manual_seed_all(config.manualSeed)


def chunks(length, step_size):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(length), step_size):
        yield length[i:i + step_size]

