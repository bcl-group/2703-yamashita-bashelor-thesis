import numpy as np

from .losses import mean_squared_error


def train(model, optimizer, x_train, t_train, x_val, t_val,
          epochs=1000, batch_size=100, verbose_every=50, early_stopping=None):
    n_train = len(x_train)
    hist = {'loss': [], 'val_loss': []}

    for epoch in range(epochs):
        perm = np.random.permutation(n_train)
        x_ = x_train[perm]
        t_ = t_train[perm]

        train_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, batch_size):
            end = start + batch_size
            x_batch = x_[start:end]
            t_batch = t_[start:end]

            y = model.forward(x_batch)

            loss = mean_squared_error(t_batch, y)

            grads = model.backward(t_batch)

            optimizer.step(model, grads)

            train_loss += loss
            n_batches += 1

        train_loss /= n_batches

        val_loss = mean_squared_error(t_val, model.forward(x_val))

        hist['loss'].append(train_loss)
        hist['val_loss'].append(val_loss)

        if verbose_every and ((epoch + 1) % verbose_every == 0 or epoch == 0):
            print('epoch: {:4d}, loss: {:.6f}, val_loss: {:.6f}'.format(
                epoch + 1, train_loss, val_loss))

        if early_stopping is not None and early_stopping(val_loss):
            if verbose_every:
                print('epoch: {:4d}, loss: {:.6f}, val_loss: {:.6f}'.format(
                    epoch + 1, train_loss, val_loss))
            break

    return hist
