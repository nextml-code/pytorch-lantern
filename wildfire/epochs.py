

def Epochs(max_epochs):
    for epoch in range(max_epochs):
        print(f'------ epoch: {epoch + 1} / {max_epochs} ------')
        yield epoch
