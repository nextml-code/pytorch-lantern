def Epochs(max_epochs):
    """Simple generator that prints the current epoch"""
    for epoch in range(1, max_epochs + 1):
        print(f"------ epoch: {epoch} / {max_epochs} ------")
        yield epoch
