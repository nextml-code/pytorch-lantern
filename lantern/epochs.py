def Epochs(max_epochs):
    """Simple generator that prints the current epoch"""
    for epoch in range(max_epochs):
        print(f"------ epoch: {epoch + 1} / {max_epochs} ------")
        yield epoch
